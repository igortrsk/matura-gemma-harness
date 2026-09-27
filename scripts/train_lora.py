"""QLoRA fine-tune of Gemma 4 12B (QAT unquantized weights) on chat-format data. Runs on the GPU machine.

The base is loaded in 4-bit (bitsandbytes NF4); only LoRA weights on the language model's linear layers are
trained. Loss is computed on the assistant answer only. The PEFT adapter is saved to --out; convert it with
llama.cpp's convert_lora_to_gguf.py to use it with the Q4_0 GGUF.

  /workspace/ft-venv/bin/python train_lora.py --data sft/pilot-test/train.jsonl --out adapters/pilot-test --epochs 3
"""

import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TARGETS = r"^(?!.*(vision|audio)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


END_OF_TURN = "<turn|>"


def encode(tok, messages, max_len):
    """Token ids of the whole chat + labels masked (-100) everywhere except the assistant answer.

    Built as inference sees it with thinking OFF: the generation prompt ends with an empty thought block
    ("<|channel>thought\\n<channel|>"), which the template drops when rendering a finished conversation,
    so the sequence is prompt + answer + end-of-turn rather than the rendered full chat."""
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(messages[-1]["content"] + END_OF_TURN, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/models/gemma-4-12b-qat-unq")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation (batch size 1)")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", help="held-out chat data (default: val.jsonl next to --data, if present)")
    ap.add_argument("--evals", type=int, default=4, help="validation checks (+ adapter snapshot) during training")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    rows = [json.loads(l) for l in open(args.data)]
    data = [encode(tok, r["messages"], args.max_len) for r in rows]
    print(f"{len(data)} examples, tokens: mean {sum(len(d[0]) for d in data) / len(data):.0f}, "
          f"max {max(len(d[0]) for d in data)}; answer tokens {sum(sum(l != -100 for l in d[1]) for d in data)}")
    print("sample prompt end:", repr(tok.decode(data[0][0][:len(data[0][0]) - sum(l != -100 for l in data[0][1])][-80:])))

    val_path = args.val or os.path.join(os.path.dirname(args.data), "val.jsonl")
    val = [encode(tok, json.loads(l)["messages"], args.max_len) for l in open(val_path)] if os.path.exists(val_path) else []
    print(f"validation: {len(val)} held-out examples from {val_path if val else '-'}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(data) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(  # short warmup, then cosine to 10%
        opt, lambda s: min(1, (s + 1) / max(1, steps // 10)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))))
    order = []
    while len(order) < steps * args.accum:
        epoch = list(range(len(data)))
        random.Random(args.seed + len(order)).shuffle(epoch)
        order += epoch
    order = order[:steps * args.accum]

    def val_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for ids, labels in val[:80]:
                losses.append(model(input_ids=torch.tensor([ids], device=0), labels=torch.tensor([labels], device=0)).loss.item())
        model.train()
        return sum(losses) / len(losses)

    eval_at = {round(steps * (i + 1) / args.evals) for i in range(args.evals)} if val else set()
    if val:
        print(f"val loss before training: {val_loss():.4f}", flush=True)
    model.train()
    t0, running = time.time(), []
    for step in range(steps):
        for i in order[step * args.accum:(step + 1) * args.accum]:
            ids, labels = data[i]
            out = model(input_ids=torch.tensor([ids], device=0), labels=torch.tensor([labels], device=0))
            (out.loss / args.accum).backward()
            running.append(out.loss.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        loss = sum(running) / len(running)
        running = []
        print(f"step {step + 1}/{steps}  loss {loss:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
              f"{time.time() - t0:.0f}s  mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        if step + 1 in eval_at:
            v = val_loss()
            snap = os.path.join(args.out, f"ckpt-{step + 1}")
            model.save_pretrained(snap)
            print(f"== step {step + 1}/{steps}  VAL LOSS {v:.4f}  (snapshot {snap})", flush=True)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
