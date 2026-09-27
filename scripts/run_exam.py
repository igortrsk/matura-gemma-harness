"""Have a model sit the practice papers and save its answers.

Talks to any OpenAI-compatible server (llama.cpp's llama-server, vLLM, LM Studio, ...).
Answers are appended to $DATA_DIR/runs/<run>/answers.jsonl as they arrive, so an
interrupted run resumes where it stopped. Closed questions with letter / P-F keys
are scored automatically at the end; open answers and essays wait for the grader.

  python scripts/run_exam.py --run gemma4-12b-qat-base --years 2023 2024 2025 --mode image
  python scripts/run_exam.py --run pkg2023-base --exam exams/history-2023-mock-v1 --key-year 2023 --think

--exam reads an organisers' package (exam.json + cropped images, the format of the real exam); --key-year attaches
our copy of the official answer key for that paper so closed items are auto-scored and the grader can work.
Turn the answers into the organisers' answers.json with write_answers.py.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.parse import urlencode
from urllib.request import urlopen

from openai import OpenAI

DATA = Path(os.environ.get("DATA_DIR") or ("/data" if Path("/data").is_dir() else "data"))

SYSTEM = (
    "Jesteś uczniem zdającym egzamin maturalny z historii na poziomie rozszerzonym. "
    "Odpowiadasz po polsku, rzeczowo i dokładnie na polecenie, korzystając ze źródeł "
    "i własnej wiedzy historycznej. Załączone strony arkusza mogą zawierać także inne zadania – "
    "odpowiadaj wyłącznie na wskazane zadanie."
)
FORMAT_HINT = {
    "closed": "Na końcu podaj odpowiedź w osobnej linii w formacie „ODPOWIEDŹ: …”, "
              "np. „ODPOWIEDŹ: C” albo „ODPOWIEDŹ: 1 – P, 2 – F, 3 – P” albo „ODPOWIEDŹ: A – 3, B – 2”.",
    "open": "Odpowiedz zwięźle, dokładnie tak, jak wymaga polecenie.",
    "essay": "Wybierz jeden temat i napisz wypracowanie (co najmniej 300 słów): wstęp ze stanowiskiem, "
             "rozwinięcie z argumentami popartymi faktami, zakończenie z wnioskiem. "
             "Zacznij od numeru wybranego tematu w formie „Temat nr N.”.",
}


# --- exam format (organisers' package: text + cropped images + answer_format) ---------------------------
# One prompt builder for the harness AND the fine-tuning data (build_sft.py), so training = exam day.
SYSTEM_EXAM = (
    "Jesteś uczniem zdającym egzamin maturalny z historii na poziomie rozszerzonym. "
    "Odpowiadasz po polsku, rzeczowo i dokładnie na polecenie, korzystając ze źródeł "
    "i własnej wiedzy historycznej. Ilustracje do zadania są załączone jako obrazy; w tekście "
    "ich miejsce oznaczono „[Obraz: …]”."
)


def exam_hint(kind, answer_format):
    if kind == "closed":
        return ("Forma odpowiedzi – podaj tylko odpowiedź, dokładnie w tej składni "
                "(wzór pokazuje tylko formę, nie poprawną odpowiedź):\n" + answer_format)
    if kind == "essay":
        return (ESSAY_GUIDE if ESSAY["guide"] else FORMAT_HINT["essay"]) + "\nForma odpowiedzi: " + answer_format
    return "Forma odpowiedzi: " + answer_format + "\n" + FORMAT_HINT["open"]


DESCRIBED_HEADER = ("Opisy ilustracji przygotowane przez pomocniczy model rozpoznawania obrazu "
                    "(mogą zawierać błędy – rozstrzygające są same ilustracje):")
RETRIEVED_HEADER = ("Fragmenty z polskiej Wikipedii wyszukane automatycznie do tego zadania "
                    "(mogą być nietrafne – korzystaj tylko z tego, co rzeczywiście dotyczy zadania):")


def exam_prompt(item):
    """User text for an exam-format item: source text, [retrieved Wikipedia passages], the task, the answer format."""
    return "\n\n".join(part for part in (
        item["context"].strip(),
        (DESCRIBED_HEADER + "\n" + item["described"]) if item.get("described") else "",
        (RETRIEVED_HEADER + "\n" + item["retrieved"]) if item.get("retrieved") else "",
        f"Zadanie ({item['max_points']} pkt)\n{item['prompt'].strip()}",
        exam_hint(item["type"], item["answer_format"]),
    ) if part)


def load_items(years):
    items = []
    for y in years:
        items += [json.loads(line) for line in open(DATA / "practice" / str(y) / "items.jsonl")]
    return items


PKG_FREE_TEXT = "Tekst po polsku"


def load_package(exam_dir, key_year=None):
    """Organisers' exam package -> harness items. Type comes from answer_format (free text / essay / closed)."""
    exam_dir = DATA / exam_dir if not Path(exam_dir).is_absolute() else Path(exam_dir)
    exam = json.loads((exam_dir / "exam.json").read_text(encoding="utf-8"))
    keys = {}
    if key_year:
        keys = {json.loads(l)["uid"]: json.loads(l) for l in open(DATA / "practice" / str(key_year) / "items.jsonl")}
    items = []
    for it in exam.get("items") or exam.get("questions") or []:  # tolerant: a new package may omit optional fields
        fmt = it.get("answer_format") or PKG_FREE_TEXT
        kind = "essay" if "wypracowanie" in fmt.lower() else "open" if fmt.startswith(PKG_FREE_TEXT) else "closed"
        key = keys.get(f"{key_year}-{it['id']}", {})
        items.append({
            "uid": f"{key_year}-{it['id']}" if key_year else it["id"], "id": it["id"], "year": key_year,
            "exam_id": exam.get("exam_id", ""), "type": kind, "max_points": it.get("max_points", 1),
            "context": it.get("source_text") or "", "prompt": it.get("question") or "", "answer_format": fmt,
            "answer": key.get("answer", ""), "rules": key.get("rules", ""),
            "image_count": len(it.get("images") or []),
            "image_files": [str(exam_dir / (im["path"] if isinstance(im, dict) else im)) for im in it.get("images") or []],
        })
    return items


def image_part(path):
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


ESSAY = {"guide": False}  # --essay-guide: detailed CKE checklist instead of FORMAT_HINT["essay"]
ESSAY_GUIDE = ("Wybierz temat, o którym znasz najwięcej konkretnych i pewnych faktów. Napisz wypracowanie (500–650 słów) "
               "zgodnie z kryteriami CKE:\n"
               "1) Wstęp: jednoznaczne stanowisko wobec tezy (zgadzam się / nie zgadzam się / zgadzam się częściowo) "
               "już w pierwszym akapicie.\n"
               "2) Rozwinięcie: KAŻDY z trzech elementów tematu (aspektów lub okresów) omów w osobnym akapicie – po 3–4 "
               "konkretne fakty (daty, postacie, wydarzenia, dokumenty, nazwy) i przy każdym wyjaśnij, jak wspiera "
               "stanowisko. Żadnego elementu nie traktuj ogólnikowo.\n"
               "3) Porównanie z inną sytuacją lub okresem albo kontrargument i jego ocena.\n"
               "4) Zakończenie: wniosek wynikający z argumentów, zgodny ze stanowiskiem.\n"
               "Podawaj tylko fakty, których jesteś pewien – każdy błąd merytoryczny obniża ocenę. "
               "Zacznij od numeru wybranego tematu w formie „Temat nr N.”.")
ZOOM = {"min_side": 0}  # --zoom-tiles N: pictures whose longer side is >= N px also get two zoomed halves (0 = off)
HALF = {True: ("lewa połowa", "prawa połowa"), False: ("górna połowa", "dolna połowa")}


def zoom_parts(path):
    """Two overlapping halves (along the longer side) of a large picture, at full resolution: the vision encoder
    shrinks every image to a fixed size, so small labels on wide maps/charts become unreadable in the whole view."""
    import pymupdf
    px = pymupdf.Pixmap(str(path))
    if max(px.width, px.height) < ZOOM["min_side"]:
        return []
    page = pymupdf.open(str(path))[0]
    r, wide = page.rect, px.width >= px.height
    scale = px.width / r.width
    parts = []
    for i, name in enumerate(HALF[wide]):
        if wide:
            clip = pymupdf.Rect(r.x0 + i * 0.45 * r.width, r.y0, r.x0 + (0.55 + i * 0.45) * r.width, r.y1)
        else:
            clip = pymupdf.Rect(r.x0, r.y0 + i * 0.45 * r.height, r.x1, r.y0 + (0.55 + i * 0.45) * r.height)
        b64 = base64.b64encode(page.get_pixmap(clip=clip, matrix=pymupdf.Matrix(scale, scale)).tobytes("png")).decode()
        parts += [{"type": "text", "text": f"[Powiększenie: images/{Path(path).name} – {name}]"},
                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
    return parts


def build_messages(item, mode):
    if "answer_format" in item:  # exam format
        content = [{"type": "text", "text": exam_prompt(item)}]
        for p in item["image_files"]:
            content += [{"type": "text", "text": f"[Obraz: images/{Path(p).name}]"}, image_part(Path(p))]
            if ZOOM["min_side"]:
                content += zoom_parts(Path(p))
        return [{"role": "system", "content": SYSTEM_EXAM}, {"role": "user", "content": content}]
    text = "\n\n".join(part for part in (
        item["context"],
        f"Zadanie {item['id']}. ({item['max_points']} pkt)\n{item['prompt']}",
        FORMAT_HINT[item["type"]],
    ) if part)
    content = [{"type": "text", "text": text}]
    if mode == "image" and item["image_count"]:
        base = DATA / "practice" / str(item["year"])
        content += [image_part(base / p) for p in item["page_images"]]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


# --- retrieval: offline Wikipedia KB (kb_server.py on the GPU box) -----------------------------------------
QUERY_PROMPT = ("Napisz jedno zapytanie do wyszukiwarki w polskiej Wikipedii, które pomoże odpowiedzieć na to zadanie: "
                "5–10 słów kluczowych (nazwy własne, wydarzenia, daty, pojęcia; także to, co rozpoznajesz na "
                "ilustracjach). Nie odpowiadaj na zadanie. Zwróć tylko zapytanie, w jednej linii.")


ADAPTERS = {}  # OpenAI base URL -> ids of the LoRA adapters that llama-server has loaded


def load_adapters(base_url):
    try:
        with urlopen(base_url.rstrip("/").removesuffix("/v1") + "/lora-adapters", timeout=10) as r:
            ADAPTERS[base_url] = [a["id"] for a in json.load(r)]
    except Exception as e:  # not a llama-server (or none loaded): nothing to switch
        print(f"  (no LoRA adapter list from {base_url}: {e})")
        ADAPTERS[base_url] = []
    return ADAPTERS[base_url]


def lora_body(base_url, on=None, scale=1.0):
    """{"lora": [...]} with an explicit scale for EVERY loaded adapter (1.0 for id `on`, else 0.0). llama-server applies a
    loaded adapter to requests that omit "lora", even with --lora-init-without-apply (verified), so plain-model
    requests must switch adapters off explicitly."""
    ids = ADAPTERS[base_url] if base_url in ADAPTERS else load_adapters(base_url)
    return {"lora": [{"id": i, "scale": scale if i == on else 0.0} for i in ids]} if ids else {}


def kb_search(args, query, k):
    url = f"{args.kb_url}/search?" + urlencode({"q": query, "k": k})
    with urlopen(url, timeout=30) as r:
        return json.load(r)["results"]


CAPWORD = re.compile(r"(?<![\w])(?:[A-ZŁŚŻŹĆŃÓĘĄ][\wąćęłńóśźż-]{2,}|1\d{3}|[5-9]\d{2})(?![\w])")
NOT_NAMES = {"Źródło", "Źródła", "Fragment", "Fragmenty", "Zadanie", "Podaj", "Wyjaśnij", "Rozstrzygnij", "Oceń",
             "Przedstaw", "Wymień", "Określ", "Uzasadnij", "Odpowiedź", "Rozstrzygnięcie", "Uzasadnienie", "Zaznacz",
             "Ilustracja", "Mapa", "Tekst", "Na", "Po", "Przy", "Przez", "Dla", "Ten", "Ta", "To", "Jego", "Jej",
             "Oto", "Jest", "Był", "Była", "Było", "Obraz", "Warszawa", "Kraków", "Wrocław", "Poznań", "Łódź", "Wiek"}


def names_query(item, limit=12):
    """Proper names and years from the sources + question (what the task is about, whatever the model believes).
    Skips citation lines, image labels and capitalised words that merely start a sentence or line."""
    text = item["context"] + "\n" + item["prompt"]
    lines = [l for l in text.splitlines()
             if not re.search(r"(s\. \d|\b(19|20)\d{2}\)?,? s\.|https?://|oprac\.|red\.|\[Obraz)", l)]
    seen = []
    for line in lines:
        for m in CAPWORD.finditer(line):
            w, before = m.group(), line[:m.start()].rstrip()
            if w[0].isupper() and (not before or before[-1] in ".!?:„\"(–-"):
                continue  # sentence/line start: probably not a name
            if w in NOT_NAMES or re.fullmatch(r"Z\d\d.*", w) or w in seen:
                continue
            seen.append(w)
    return " ".join(seen[:limit])


def retrieve(client, args, item):
    """Model-written query + the question's own wording -> merged top passages (text for exam_prompt)."""
    msgs = build_messages(item, args.mode)
    msgs[1]["content"][0]["text"] += "\n\n" + QUERY_PROMPT
    r = client.chat.completions.create(model=args.model, messages=msgs, temperature=0, max_tokens=60,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False},
                                                   **lora_body(args.base_url)})
    query = (r.choices[0].message.content or "").strip().splitlines()[0][:200] if r.choices[0].message.content else ""
    hits, seen = [], set()
    names = names_query(item)
    lists = [kb_search(args, query, args.kb_k) if query else [], kb_search(args, names, args.kb_k) if names else []]
    for pair in zip(*[l + [None] * (args.kb_k - len(l)) for l in lists]):  # interleave: model query first
        for h in pair:
            if h and h["id"] not in seen and len(hits) < args.kb_k:
                seen.add(h["id"])
                hits.append(h)
    return format_hits(hits), {"query": query, "names": names, "titles": [h["title"] for h in hits]}


def format_hits(hits):
    """Passages as they appear under RETRIEVED_HEADER (shared with build_sft.py --retrieval)."""
    return "\n".join(f"[{i}] {h['title']}: {' '.join(h['text'].split()[:120])}" for i, h in enumerate(hits, 1))


def lora_id(args, item):
    """Adapter id switched on for this item's type (--lora/--lora-types, --lora-for TYPE=ID), or None = plain model."""
    return args.lora_map.get(item["type"])


def ask_once(client, args, item, think, extra_text="", max_tokens=None):
    if (lora_id(args, item) is not None and not args.lora_think) or item["type"] in args.no_think_types:
        think = False  # adapters are trained in the thinking-OFF format
    if max_tokens is None:
        max_tokens = 3000 if item["type"] == "essay" else 2000  # 1000 cut off verbose no-thinking answers
    extra = {"chat_template_kwargs": {"enable_thinking": think},
             **lora_body(args.base_url, lora_id(args, item), args.lora_scale)}
    if think:  # thinking shares the token limit; too small a budget leaves an empty answer
        max_tokens += args.essay_think_budget if item["type"] == "essay" and args.essay_think_budget else args.think_budget
    msgs = build_messages(item, args.mode)
    if extra_text:
        msgs[1]["content"][0]["text"] += "\n\n" + extra_text
    start = time.time()
    r = client.chat.completions.create(
        model=args.model, messages=msgs,
        temperature=args.temperature, max_tokens=max_tokens, extra_body=extra,
    )
    msg = r.choices[0].message
    return {
        "uid": item["uid"], "answer": (msg.content or "").strip(),
        "reasoning": getattr(msg, "reasoning_content", None),
        "finish_reason": r.choices[0].finish_reason,
        "tokens": r.usage.completion_tokens if r.usage else None,
        "seconds": round(time.time() - start, 1), "mode": args.mode, "think": think,
        **({"lora": lora_id(args, item)} if lora_id(args, item) is not None else {}),
    }


PLAN_PROMPT = ("Zanim napiszesz wypracowanie, przygotuj jego plan (nie pisz jeszcze wypracowania). W punktach: "
               "1) numer wybranego tematu; 2) jednoznaczne stanowisko wobec tezy; 3) dla KAŻDEGO z trzech elementów "
               "tematu: argument i 3–4 konkretne fakty (daty, postacie, wydarzenia, dokumenty) z wyjaśnieniem, jak "
               "wspierają stanowisko; 4) porównanie z alternatywą albo kontrargument i jego ocena; 5) wniosek. "
               "Używaj tylko pewnych faktów.")
WRITE_FROM_PLAN = ("Twój plan wypracowania:\n{plan}\n\nNapisz teraz wypracowanie zgodnie z tym planem: każdy z trzech "
                   "elementów tematu w osobnym, rozbudowanym akapicie z faktami z planu, porównanie lub kontrargument, "
                   "zakończenie z wnioskiem (co najmniej 450 słów). Zacznij od „Temat nr N.”.")


DESCRIBE_PROMPT = ("To jest ilustracja ze źródła do zadania maturalnego z historii. Podpis i kontekst w arkuszu:\n{caption}\n\n"
                   "Opisz ilustrację dokładnie i rzeczowo, po polsku: rodzaj; wszystkie widoczne napisy, podpisy, liczby i "
                   "legendę (przepisz dosłownie); kto i co jest przedstawione (osoby, gesty, atrybuty, symbole, układ; na "
                   "mapie obszary, granice, strzałki, miejscowości). Nie zgaduj ponad to, co widać.")
DESCRIPTIONS = {}  # image path -> description (shared images are described once)
DESC_LOCK = Lock()


def describe_images(args, item):
    """Descriptions of the item's pictures by the helper vision model (e.g. Qwen3.5-9B on another llama-server)."""
    client = OpenAI(base_url=args.describer_url, api_key="none", timeout=600)
    out = []
    for p in item["image_files"]:
        with DESC_LOCK:
            cached = DESCRIPTIONS.get(p)
        if cached is None:
            name = Path(p).name
            lines = item["context"].splitlines()
            idx = next((i for i, l in enumerate(lines) if name in l), 0)
            caption = "\n".join(l for l in lines[max(0, idx - 2):idx + 3] if "[Obraz" not in l)[:600] or "(brak)"
            r = client.chat.completions.create(
                model="describer", temperature=0, max_tokens=900,
                messages=[{"role": "user", "content": [{"type": "text", "text": DESCRIBE_PROMPT.format(caption=caption)},
                                                       image_part(Path(p))]}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, **lora_body(args.describer_url)})
            cached = (r.choices[0].message.content or "").strip()
            with DESC_LOCK:
                DESCRIPTIONS[p] = cached
        out.append(f"[Obraz: images/{Path(p).name}] {cached}")
    return "\n\n".join(out)


def ask(client, args, item):
    if args.describer_url and item.get("image_files"):
        item = {**item, "described": describe_images(args, item)}
    if args.essay_plan and item["type"] == "essay":
        plan = ask_once(client, args, item, args.think, extra_text=PLAN_PROMPT, max_tokens=1500)
        rec = ask_once(client, args, item, args.think, extra_text=WRITE_FROM_PLAN.format(plan=plan["answer"]))
        rec["plan"] = plan["answer"]
        rec["seconds"] = round(rec["seconds"] + plan["seconds"], 1)
        return rec
    info = None
    if args.retrieval and item["type"] != "essay" and "answer_format" in item:
        text, info = retrieve(client, args, item)
        item = {**item, "retrieved": text}
    rec = ask_once(client, args, item, args.think)
    if info:
        rec["retrieval"] = info
    if args.think and args.no_think_fallback and not rec["answer"]:
        # thinking looped until the token limit: answer the same question again without thinking
        retry = ask_once(client, args, item, False)
        retry["fallback"] = {k: rec[k] for k in ("finish_reason", "tokens", "seconds")}
        retry["seconds"] = round(retry["seconds"] + rec["seconds"], 1)
        if info:
            retry["retrieval"] = info
        return retry
    return rec


# --- automatic scoring of closed questions with letter / P-F keys -------------------

PAIR = re.compile(r"\b([A-Z]|\d+)\s*\.?\s*[–—\-:=)]\s*([A-Z]|\d+)\b")
SINGLE = re.compile(r"^\s*([A-D])\s*\.?\s*$")


def parse_marks(text):
    """'1 – F, 2 – P' -> {'1': 'F', '2': 'P'};  'C' -> {'_': 'C'};  otherwise None."""
    text = text.replace("\n", ", ")
    if m := SINGLE.match(text):
        return {"_": m.group(1)}
    pairs = dict(PAIR.findall(text))
    return pairs or None


def final_line(answer):
    m = re.findall(r"ODPOWIEDŹ\s*:\s*(.+)", answer, re.I)
    return m[-1] if m else answer.strip().splitlines()[-1] if answer.strip() else ""


EXAM_LINE = re.compile(r"^\s*([A-Z]|\d+)\s*[.)]?\s*[:–—-]\s*([A-Z]|\d+)\s*[.,;]?\s*$")


def answer_marks(answer):
    """Marks from an answer: exam-syntax lines ("1: P"), a lone letter line, else the ODPOWIEDŹ/final line."""
    lines = [l for l in answer.strip().splitlines() if l.strip()]
    pairs = [m.groups() for l in lines if (m := EXAM_LINE.match(l.replace("*", "")))]
    if pairs:
        return dict(pairs)
    if lines and (m := SINGLE.match(lines[-1].replace("*", ""))):
        return {"_": m.group(1)}
    return parse_marks(final_line(answer)) or {}


def score_closed(item, answer):
    """Points for keys made of letters / P-F marks; None when the key needs a human-style grader."""
    key = parse_marks(item["answer"])
    if not key or any(len(v) > 1 and not v.isdigit() for v in key.values()):
        return None
    got = answer_marks(answer)
    if "_" in key and "_" not in got and len(got) == 1:
        got = {"_": next(iter(got.values()))}
    if "_" in key and "_" not in got:  # single choice: accept a lone letter anywhere in the final line
        if m := re.search(r"\b([A-D])\b", final_line(answer)):
            got = {"_": m.group(1)}
    correct = sum(got.get(k, "").upper() == v for k, v in key.items())
    n, top = len(key), item["max_points"]
    if n == top:  # one point per mark
        return correct
    if top == 2 and n == 3:  # CKE: 3 correct -> 2 pts, 2 correct -> 1 pt
        return max(0, correct - 1)
    return top if correct == n else 0  # all-or-nothing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, e.g. gemma4-12b-qat-base")
    ap.add_argument("--years", nargs="+", default=["2023", "2024", "2025"])
    ap.add_argument("--exam", help="organisers' exam package dir (exam.json + images/), used instead of --years")
    ap.add_argument("--key-year", help="attach our official key for this paper (package runs only)")
    ap.add_argument("--mode", choices=["text", "image"], default="image")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="local", help="model name sent to the server")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--think", action="store_true", help="enable the model's thinking mode")
    ap.add_argument("--think-budget", type=int, default=12000, help="extra tokens allowed for thinking")
    ap.add_argument("--no-think-fallback", action=argparse.BooleanOptionalAction, default=True,
                    help="if thinking ends without an answer, ask again with thinking off (default on)")
    ap.add_argument("--retrieval", action="store_true", help="add offline Wikipedia passages (exam-format, non-essay)")
    ap.add_argument("--kb-url", default="http://localhost:8090", help="kb_server.py (tunnel -L 8090 from Docker)")
    ap.add_argument("--kb-k", type=int, default=4, help="passages per question")
    ap.add_argument("--redo-empty", action="store_true", help="drop saved empty/truncated answers and ask again")
    ap.add_argument("--parallel", type=int, default=4, help="concurrent requests (match llama-server -np)")
    ap.add_argument("--only", nargs="*", help="limit to these item uids (smoke tests)")
    ap.add_argument("--skip-essays", action="store_true", help="leave essays out (non-essay tests)")
    ap.add_argument("--describer-url", help="helper vision model (OpenAI-compatible) that describes pictures first")
    ap.add_argument("--images-only", action="store_true", help="only items with pictures (describer tests)")
    ap.add_argument("--essays-only", action="store_true", help="only essays (essay tests)")
    ap.add_argument("--essay-plan", action="store_true", help="essays: write a plan first, then the essay from it")
    ap.add_argument("--lora", type=int, help="llama-server LoRA adapter id to switch on (per request)")
    ap.add_argument("--lora-types", nargs="+", default=["essay"], help="item types that get the adapter")
    ap.add_argument("--lora-for", nargs="+", default=[], metavar="TYPE=ID",
                    help="per-type adapters, e.g. essay=0 open=1 (adds to / overrides --lora)")
    ap.add_argument("--lora-scale", type=float, default=1.0, help="strength of the switched-on adapter")
    ap.add_argument("--lora-think", action="store_true", help="keep thinking ON for adapter items (default: off)")
    ap.add_argument("--essay-think-budget", type=int, help="thinking budget for essays (default: --think-budget)")
    ap.add_argument("--essay-guide", action="store_true", help="detailed CKE essay checklist in the prompt")
    ap.add_argument("--zoom-tiles", type=int, default=0, metavar="PX",
                    help="pictures with longer side >= PX also sent as two zoomed halves (0 = off)")
    ap.add_argument("--no-think-types", nargs="+", default=[], help="item types answered with thinking OFF")
    args = ap.parse_args()
    ZOOM["min_side"] = args.zoom_tiles
    ESSAY["guide"] = args.essay_guide
    args.lora_map = {t: args.lora for t in args.lora_types} if args.lora is not None else {}
    args.lora_map.update({t: int(i) for t, i in (s.split("=") for s in args.lora_for)})
    if ("2026" in args.years or args.key_year == "2026") and not os.environ.get("ALLOW_HELDOUT"):
        raise SystemExit("2026 is the held-out paper; set ALLOW_HELDOUT=1 only for the final evaluation.")

    out = DATA / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    answers_file = out / "answers.jsonl"
    if args.redo_empty and answers_file.exists():
        keep = [l for l in open(answers_file) if json.loads(l)["answer"].strip() and json.loads(l)["finish_reason"] != "length"]
        answers_file.write_text("".join(keep))
    done = {json.loads(l)["uid"] for l in open(answers_file)} if answers_file.exists() else set()

    source = load_package(args.exam, args.key_year) if args.exam else load_items(args.years)
    items = [it for it in source if (not args.only or it["uid"] in args.only)
             and not (args.skip_essays and it["type"] == "essay")
             and not (args.essays_only and it["type"] != "essay")
             and not (args.images_only and not it.get("image_files"))]
    todo = [it for it in items if it["uid"] not in done]
    print(f"{args.run}: {len(items)} items, {len(done & {i['uid'] for i in items})} already answered, {len(todo)} to go")

    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("LLM_API_KEY", "none"), timeout=600)
    for url in {args.base_url, args.describer_url} - {None}:  # before the threads start
        print(f"  LoRA adapters on {url}: {load_adapters(url)} (switched on only for {args.lora_map or 'none'})")
    if missing := set(args.lora_map.values()) - set(ADAPTERS[args.base_url]):
        sys.exit(f"adapters {missing} are not loaded on {args.base_url}")
    lock, t0 = Lock(), time.time()
    with ThreadPoolExecutor(args.parallel) as pool, open(answers_file, "a") as f:
        futures = {pool.submit(ask, client, args, it): it for it in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            it = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # keep going; failed items are retried on the next run
                print(f"  ! {it['uid']}: {e}")
                continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            print(f"  [{i}/{len(todo)}] {rec['uid']:10} {rec['seconds']:6.1f}s  {rec['answer'][:70]!r}")

    answers = {json.loads(l)["uid"]: json.loads(l) for l in open(answers_file)}
    report, rows = {}, []
    for it in items:
        a = answers.get(it["uid"])
        pts = score_closed(it, a["answer"]) if a and it["type"] == "closed" and it["answer"] else None
        rows.append({"uid": it["uid"], "type": it["type"], "max": it["max_points"], "auto_points": pts})
        if pts is not None:
            r = report.setdefault(str(it["year"] or it.get("exam_id")), [0, 0])
            r[0] += pts
            r[1] += it["max_points"]
    (out / "auto_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\ndone in {time.time() - t0:.0f}s — answers: {out / 'answers.jsonl'}")
    for y, (got, top) in sorted(report.items()):
        print(f"  {y}: auto-scored closed questions {got}/{top}")


if __name__ == "__main__":
    main()
