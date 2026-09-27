"""Turn synthetic records into clean chat-format fine-tuning sets (text only).

Prompts are built by run_exam.exam_prompt — the same function the harness uses for exam-format items (source text,
"Zadanie (N pkt)", question, the exam's answer_format) with SYSTEM_EXAM — so training = exam day. Closed answers
are trained in the exam syntax ("1: P\n2: F\n3: P", "C", "A: 3\nB: 2"), nothing else. Raw synthetic files are never
modified; all cleanup happens here:
  - answers: drop "Przykładowa odpowiedź:" prefixes, "Przykładowe uzasadnienie:" -> "Uzasadnienie:";
    "rozstrzygnij" tasks get "Rozstrzygnięcie: … / Uzasadnienie: …"; closed answers become exam syntax only.
  - old-paper questions: drop layout leftovers ("Uzyskana liczba pkt", task headers, GPT instructions).
  - P/F balance: GPT favours "P F P"; at most --max-pfp of those are kept.
  - essays: exam format (three topics, pick one; answer starts with "Temat nr N."); essays on the 2023–25
    practice topics (Karol Wielki; American vs French revolution) are left out to keep the essay eval fair.
  - duplicates (same prompt) dropped.

  python scripts/build_sft.py --name v1-essay --kinds essay synthetic/pilot/essays.jsonl \
      synthetic/test2/essays_new.jsonl synthetic/v1/essays_new.jsonl
  python scripts/build_sft.py --name v1-short --kinds open,closed synthetic/v1/source_tasks.jsonl \
      synthetic/v1/old_items.jsonl

--retrieval adds offline Wikipedia passages to open/closed prompts, exactly as run_exam.py --retrieval shows them
(RETRIEVED_HEADER + format_hits), so the short-answer specialist trains on exam-day prompts. Needs kb_server.py
(--kb-url, default the (local machine) forward localhost:8090). Query = names_query (names/years in the task; the harness
also adds a model-written query, which needs the model server). The Wikipedia article a source task was written
from is excluded from the hits: exam sources are not Wikipedia, so the passages must not simply contain the source.

Output: $DATA_DIR/sft/<name>/train.jsonl — {"messages": [system, user, assistant], "kind", "src"} per line,
plus stats.json.
"""

import argparse
import json
import random
import re
from urllib.parse import unquote

from build_package import ESSAY_FMT, FREE_TEXT, answer_format
from run_exam import DATA, SYSTEM_EXAM, exam_prompt, format_hits, kb_search, names_query, parse_marks

ESSAY_HEADER = ("Zadanie zawiera trzy tematy. Wybierz jeden z nich do opracowania. "
                "Twoja wypowiedź powinna liczyć minimum 300 wyrazów.")
PRACTICE_ESSAY_TOPICS = re.compile(r"Karol\w* Wielk|rewolucj\w* amerykańsk.*francusk|rewolucj\w* francusk.*amerykańsk",
                                   re.I | re.S)
LAYOUT_LINE = re.compile(r"^\s*(Uzyskana liczba pkt|Wypełnia egzaminator.*|Nr zadania.*|Maks\. liczba pkt.*"
                         r"|Zadanie \d+(\.\d+)?\. \(\d+ pkt\)|Podaj tylko odpowiedź \(bez powtarzania polecenia\)\.)\s*$")


def user_text(context, prompt, points, kind, fmt, retrieved=""):
    return exam_prompt({"context": context, "prompt": prompt, "max_points": points, "type": kind, "answer_format": fmt,
                        "retrieved": retrieved})


def source_title(rec):
    """Title of the Wikipedia article a source task was written from ('' for old-paper questions)."""
    url = (rec.get("meta") or {}).get("url", "")
    return unquote(url.rsplit("/wiki/", 1)[1]).replace("_", " ") if "/wiki/" in url else ""


def make_retriever(args):
    def retrieve(rec, context, prompt):
        """Top --kb-k passages for the names/years in the task, minus the task's own source article."""
        names = names_query({"context": context, "prompt": prompt})
        if not names:
            return "", []
        skip = source_title(rec)
        hits = [h for h in kb_search(args, names, args.kb_k + 4) if h["title"] != skip][:args.kb_k]
        return format_hits(hits), [h["title"] for h in hits]
    return retrieve


def clean_question(q):
    lines = [l for l in q.splitlines() if not LAYOUT_LINE.match(l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_answer(a):
    a = re.sub(r"^\s*Przykładowa odpowiedź:\s*", "", a.strip())
    a = re.sub(r"Przykładowe (uzasadnienie|uzasadnienia):", "Uzasadnienie:", a)
    a = re.sub(r"Przykładowe (wyjaśnienie|wyjaśnienia):", "Wyjaśnienie:", a)
    return a.strip()


def with_decision(a):
    """'Nie. Bo …' -> 'Rozstrzygnięcie: Nie.\\nUzasadnienie: Bo …' (for rozstrzygnij tasks without labels)."""
    if "Rozstrzygnięcie" in a:
        return a
    m = re.match(r"^(.+?[.!])\s+(.+)$", a, re.S)
    return f"Rozstrzygnięcie: {m.group(1)}\nUzasadnienie: {m.group(2)}" if m else a


def closed_marks(a):
    """Canonical marks ("1 – P, 2 – F", "C") for letter / P-F / matching answers, or None if not recognisable."""
    pf = re.findall(r"(?:^|[\s;,/])(\d+)\s*[.):]?\s*[–—\-:]?\s*([PF])\b", a)
    if len(pf) >= 2 and len({n for n, _ in pf}) == len(pf):
        return ", ".join(f"{n} – {v}" for n, v in pf)
    pairs = re.findall(r"(?:^|[\s;,/])([A-F])\s*[.):]?\s*[–—\-:]?\s*(\d+)\b", a)
    if len(pairs) >= 2 and len({l for l, _ in pairs}) == len(pairs):
        return ", ".join(f"{l} – {n}" for l, n in pairs)
    if m := re.match(r"^\s*([A-F])\b(?![’'])[.)]?(\s|$)", a):
        return m.group(1)
    return None


def closed_exam_answer(a, points):
    """(answer_format, answer in exam syntax) for a closed answer, or None if its marks cannot be read."""
    marks = closed_marks(a)
    if marks is None:
        return None
    fmt = answer_format(marks, points)
    got = parse_marks(marks)
    if fmt == FREE_TEXT or not got:
        return None
    if "_" in got:
        return fmt, got["_"]
    return fmt, "\n".join(f"{k}: {v}" for k, v in got.items())


def short_example(rec, retrieve=None):
    """Source task / old-paper question -> (kind, user, answer, retrieved titles) or None."""
    if rec.get("skipped") or not rec.get("answer"):
        return None
    kind = rec.get("type") if rec.get("type") in ("open", "closed") else "open"
    answer = clean_answer(rec["answer"])
    if "rozstrzygnij" in (rec.get("task_type") or "") or re.match(r"^\s*Rozstrzygnij", rec.get("prompt") or ""):
        answer = with_decision(answer)
    fmt = FREE_TEXT
    if kind == "closed":
        closed = closed_exam_answer(answer, rec.get("max_points", 1))
        if closed is None:
            kind = "open"  # free-text matching etc.: answered as text, like the organisers' format does
        else:
            fmt, answer = closed
    if "context" in rec:  # task written around a real source excerpt
        context, prompt, points = rec["context"], rec["prompt"], rec["max_points"]
    else:  # old-format question (sources are inside the question text)
        context, prompt, points = "", clean_question(rec["question"]), rec.get("max_points", 1)
    retrieved, titles = retrieve(rec, context, prompt) if retrieve else ("", [])
    return kind, user_text(context, prompt, points, kind, fmt, retrieved), answer, titles


def essay_examples(recs, rng):
    topics = [re.sub(r"^\s*Teza:\s*", "", r["topic"]).strip() for r in recs]
    for i, r in enumerate(recs):
        others = rng.sample([t for j, t in enumerate(topics) if j != i and recs[j].get("era") != r.get("era")], 2)
        pos = rng.randrange(3)
        three = others[:pos] + [topics[i]] + others[pos:]
        prompt = ESSAY_HEADER + "\n" + "\n".join(f"{n}. {t}" for n, t in enumerate(three, 1))
        yield "essay", user_text("", prompt, 15, "essay", ESSAY_FMT), f"Temat nr {pos + 1}.\n\n{r['essay'].strip()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kinds", default="essay,open,closed", help="comma-separated: essay,open,closed")
    ap.add_argument("--max-pfp", type=int, default=30, help="keep at most this many 'P F P' true/false answers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05, help="held-out share for validation (never trained on)")
    ap.add_argument("--retrieval", action="store_true", help="add offline Wikipedia passages to open/closed prompts")
    ap.add_argument("--kb-url", default="http://localhost:8090")
    ap.add_argument("--kb-k", type=int, default=4, help="passages per task (harness default 4)")
    ap.add_argument("files", nargs="+", help="synthetic jsonl files, relative to $DATA_DIR")
    args = ap.parse_args()
    args.kinds = args.kinds.split(",")
    rng = random.Random(args.seed)

    essays, examples, stats = [], [], {"dropped": {}}
    retrieve = make_retriever(args) if args.retrieval else None
    titles_of = {}  # user text -> retrieved titles (saved per row for inspection)

    def drop(reason):
        stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1

    for f in args.files:
        for line in open(DATA / f):
            rec = json.loads(line)
            if rec.get("essay"):
                if len(rec["essay"].split()) < 350:
                    drop("essay under 350 words")
                elif PRACTICE_ESSAY_TOPICS.search(rec["topic"]):
                    drop("essay on a 2023–25 practice topic")
                else:
                    essays.append(rec)
                continue
            ex = short_example(rec, retrieve)
            if ex is None:
                drop("no answer")
            else:
                examples.append((rec.get("src"), *ex[:3]))
                titles_of[ex[1]] = ex[3]
    examples += [(r.get("src"), *ex) for r, ex in zip(essays, essay_examples(essays, rng))]

    pfp = [e for e in examples if e[1] == "closed" and e[3] == "1: P\n2: F\n3: P"]
    for e in rng.sample(pfp, max(0, len(pfp) - args.max_pfp)):
        examples.remove(e)
        drop("P/F rebalance (P F P)")

    rows, seen = [], set()
    for src, kind, user, answer in examples:
        if kind not in args.kinds:
            continue
        if user in seen:
            drop("duplicate prompt")
            continue
        seen.add(user)
        extra = {"retrieved": titles_of[user]} if retrieve and kind != "essay" else {}
        rows.append({"kind": kind, "src": src, **extra, "messages": [
            {"role": "system", "content": SYSTEM_EXAM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]})
    rng.shuffle(rows)
    n_val = round(len(rows) * args.val_frac)
    val, rows = rows[:n_val], rows[n_val:]
    out = DATA / "sft" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "val.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val))
    stats["val"] = len(val)
    stats["kinds"] = {k: sum(r["kind"] == k for r in rows) for k in args.kinds}
    stats["total"] = len(rows)
    stats["files"] = args.files
    if retrieve:
        with_hits = [r for r in rows + val if r.get("retrieved")]
        stats["retrieval"] = {"rows_with_passages": len(with_hits), "rows_without": len(rows) + len(val) - len(with_hits),
                              "kb_k": args.kb_k}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2), "\n->", out / "train.jsonl")


if __name__ == "__main__":
    main()
