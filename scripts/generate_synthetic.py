"""Generate matura-style training examples with GPT-6 (Forgehand team LLM credits).

Sources are the older-format CKE papers ($DATA_DIR/f2015, built with `build_practice_set.py --old`),
never the Formuła 2023 practice/held-out papers.

  essays: old essay topic -> GPT rewrites it as a Formuła 2023 thesis topic (stance + three elements)
          and writes a model essay aimed at full marks under the CKE rubric.
  open:   old open question (+ its sources) -> concise model answer; the official key is stored
          alongside so answers can be checked before they are used for training.

Output: $DATA_DIR/synthetic/<run>/{essays,open}.jsonl, appended as results arrive (resumable),
plus cost.json with the team's LLM balance before/after.

  python scripts/generate_synthetic.py --run pilot --essays 10 --open 20
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen

from openai import OpenAI

DATA = Path(os.environ.get("DATA_DIR") or ("/data" if Path("/data").is_dir() else "data"))
FH_URL = os.environ.get("FORGEHAND_URL", "https://app.forgehand.app")

ESSAY_SYSTEM = """Jesteś doświadczonym nauczycielem historii i egzaminatorem CKE. Piszesz wzorcowe wypracowania \
maturalne z historii (poziom rozszerzony, Formuła 2023), które otrzymują maksymalną liczbę punktów.

Kryteria CKE (15 pkt):
A. Narracja historyczna (0–12): zdający zajmuje jednoznaczne stanowisko wobec tezy i przekonująco je uzasadnia, \
wykorzystując funkcjonalnie wiedzę historyczną w odniesieniu do TRZECH elementów wskazanych w temacie. \
Maksimum wymaga bogatej argumentacji dla każdego z trzech elementów: rzeczowej, pogłębionej, popartej trafnie \
dobraną, szczegółową faktografią (daty, postacie, wydarzenia, procesy) i adekwatną terminologią. \
Za błędy merytoryczne (chronologia, terminologia, związki przyczynowo-skutkowe) odejmowane są punkty.
B. Spójność wypowiedzi (0–3): co najmniej 300 słów; wstęp ze stanowiskiem – rozwinięcie – zakończenie z wnioskiem \
wynikającym z wywodu; każdy akapit wynika z poprzedniego; bez wątków pobocznych.

Pisz poprawną polszczyzną, jak najlepszy maturzysta (nie jak podręcznik). Nie wymyślaj faktów – używaj tylko \
pewnej wiedzy."""

ESSAY_USER = """Temat z dawnej formuły egzaminu:
{topic}

1. Przeformułuj go na temat w stylu Formuły 2023: teza do oceny + polecenie „Zajmij stanowisko wobec powyższej \
tezy i je uzasadnij, uwzględniając w swojej argumentacji …” z TRZEMA wyraźnie wskazanymi elementami \
(np. aspekty: polityczny, społeczno-gospodarczy, kulturowy; albo trzech władców; albo trzy okresy).
2. Napisz wzorcowe wypracowanie na ten nowy temat (420–550 słów), w akapitach.

Zwróć JSON: {{"topic": "...", "essay": "..."}}"""

OPEN_SYSTEM = """Jesteś najlepszym maturzystą zdającym egzamin z historii (poziom rozszerzony). Odpowiadasz po polsku, \
zwięźle i dokładnie na polecenie: tyle elementów, ile wymaga zadanie, z odwołaniem do źródeł, gdy polecenie tego \
wymaga. Egzaminator przyznaje punkty według klucza CKE – odpowiedź ma trafić w kryteria, bez zbędnych dygresji."""

OPEN_USER = """{context}

Zadanie {id}. ({points} pkt)
{prompt}

Podaj tylko odpowiedź (bez powtarzania polecenia)."""


SOURCE_SYSTEM = """Jesteś autorem zadań CKE do matury z historii (poziom rozszerzony, Formuła 2023). Na podstawie \
prawdziwego fragmentu źródła układasz zadania w stylu arkuszy CKE wraz z kluczem odpowiedzi i zasadami oceniania.

Zasady:
- Ułóż 2–3 zadania RÓŻNYCH typów, wybierając z: rozstrzygnij i uzasadnij (z odwołaniem do źródła); podaj nazwę / \
postać / wydarzenie / datę (identyfikacja wymagająca własnej wiedzy); wyjaśnij (przyczynę, skutek, pojęcie, \
znaczenie); porównaj; oceń prawdziwość trzech zdań (P/F); zadanie wyboru A–D; przyporządkowanie.
- Co najmniej jedno zadanie musi wymagać połączenia treści źródła z własną wiedzą historyczną (jak w CKE).
- Możesz zastąpić w fragmencie nazwy własne znakiem […], jeśli zadanie ma sprawdzać ich rozpoznanie – zwróć wtedy \
zmieniony fragment. Możesz skrócić fragment (zaznacz pominięcia […]).
- Odpowiedzi muszą być PEWNE: wynikać z fragmentu albo z ugruntowanej wiedzy. Nie wymyślaj faktów, dat, cytatów.
- Klucz pisz jak CKE: zwięźle, tylko to, za co przyznaje się punkty; dla zadań otwartych podaj przykładową odpowiedź.
- Jeśli fragment nie nadaje się na źródło historyczne (np. nie dotyczy historii, jest urywkiem bez sensu), zwróć {"tasks": []}."""

SOURCE_USER = """Przykłady prawdziwych zadań CKE (styl, poziom trudności, forma klucza):
{examples}

---
Fragment źródła ({site}, „{title}”{section}):
{text}

Zwróć JSON: {{"source_intro": "np. Fragment opracowania historycznego / Fragment dokumentu", "excerpt": "fragment (ew. zmieniony)", \
"tasks": [{{"task_type": "...", "type": "open|closed", "points": 1, "prompt": "treść polecenia", "answer": "klucz", \
"rules": "1 pkt – …\\n0 pkt – …"}}]}}"""

TOPICS_USER = """Okres: {era}
Ułóż {n} różnorodnych tematów wypracowań maturalnych z historii (poziom rozszerzony, Formuła 2023), dotyczących tego okresu. \
Każdy temat: teza do oceny, a po niej polecenie „Zajmij stanowisko wobec powyższej tezy i je uzasadnij, uwzględniając \
w swojej argumentacji …” z TRZEMA wyraźnie wskazanymi elementami (np. aspekty polityczny, społeczno-gospodarczy i \
kulturowy; trzech władców; trzy wydarzenia; trzy okresy). Tezy mają być dyskusyjne, ale oparte na faktach. \
{avoid}Zwróć JSON: {{"topics": ["..."]}}"""

ESSAY_USER_NEW = """Temat:
{topic}

Napisz wzorcowe wypracowanie na ten temat (420–550 słów), w akapitach. Zwróć JSON: {{"essay": "..."}}"""

# v2: aimed at what our model loses points on (argumentation "zadowalająca/powierzchowna" instead of "bogata")
ESSAY_USER_V2 = """Temat:
{topic}

Napisz wzorcowe wypracowanie na ten temat (480–600 słów) za 15/15 pkt. Wymagania:
1. Wstęp: jednoznaczne stanowisko wobec tezy (zgoda, niezgoda albo zgoda częściowa) i zapowiedź argumentacji.
2. Rozwinięcie: KAŻDY z trzech elementów wskazanych w temacie w osobnym akapicie, rozwinięty równie głęboko. W każdym akapicie co najmniej 2–3 konkretne fakty (daty, postacie, wydarzenia, nazwy dokumentów, procesy) i wyjaśnienie, jak te fakty potwierdzają lub osłabiają tezę – nie samo wyliczenie.
3. Pogłębienie: porównanie z alternatywą albo kontrargument (np. inne państwo, inny władca, inny czynnik, sytuacja wcześniejsza lub późniejsza) i jego ocena.
4. Zakończenie: wniosek wynikający z wywodu, nawiązujący do tezy.
Tylko pewne fakty; poprawna terminologia; bez wątków pobocznych. Zwróć JSON: {{"essay": "..."}}"""

# Essay topics of the 2023–25 practice papers: new topics must not repeat them (keeps the essay eval fair).
PRACTICE_ESSAY_TOPICS = [
    "W życiu politycznym państwa polskiego w okresie XI–XII wieku dominowały tendencje decentralizacyjne.",
    "Rewolucja amerykańska i francuska z końca XVIII wieku miały podobne przyczyny.",
    "Zimna wojna osiągnęła apogeum w latach 50. XX wieku.",
    "Najbardziej udaną próbą odnowienia tradycji imperium rzymskiego w średniowieczu było państwo Karola Wielkiego.",
    "Wojny z Turcją w największym stopniu przyczyniły się do upadku znaczenia Rzeczypospolitej w XVII wieku.",
    "Niepodległość Polska zawdzięczała przede wszystkim przywództwu Józefa Piłsudskiego.",
    "Władysław Jagiełło był najwybitniejszym władcą Polski z dynastii Jagiellonów.",
    "Lata 1871–1914 są niesłusznie określane jako belle époque.",
    "Rok 1956 był przełomem w systemie komunistycznym.",
]

EXAMINER_BOX = re.compile(r"\n(?:Wypełnia|egzaminator|Nr zadania|Maks\. liczba pkt|Uzyskana liczba pkt|[\d. ]+)(?=\n)")


def clean_md(text):
    """Exam answers are plain text: drop markdown emphasis and headings."""
    return re.sub(r"(\*\*|__|^#+\s*)", "", text, flags=re.M).strip()


def llm_client():
    team = os.environ.get("FORGEHAND_TEAM_ID") or dict(
        line.split("=", 1) for line in open(DATA / ".env").read().split() if "=" in line)["FORGEHAND_TEAM_ID"]
    token = os.environ.get("FORGEHAND_TOKEN") or json.load(open(DATA / ".config/forgehand/config.json"))["token"]
    return OpenAI(api_key=token, base_url=f"{FH_URL}/api/v1/teams/{team}/llm/v1", max_retries=2, timeout=300), team, token


def balance(team, token):
    req = Request(f"{FH_URL}/api/v1/teams/{team}/llm/usage", headers={"authorization": f"Bearer {token}"})
    d = json.load(urlopen(req))
    return {"spent_usd": d["spentMicroUsd"] / 1e6, "available_usd": d["availableMicroUsd"] / 1e6}


def old_items():
    for f in sorted((DATA / "f2015").glob("*/items.jsonl")):
        yield from map(json.loads, open(f))


def essay_topics():
    """Split each old essay task into its numbered topics; skip ones that depend on printed source packs."""
    for it in old_items():
        if it["type"] != "essay":
            continue
        text = re.sub(r"\s*\n\s*", " ", it["prompt"])
        text = re.split(r"Wypełnia|Materiały źródłowe|Źródło A\.", text)[0]  # topics come before examiner box / sources
        seen = set()
        for n, topic in re.findall(r"(?:^| )(\d)\. (.+?)(?= \d\. |$)", text):
            if n in seen or re.search(r"źródł|odwołaj się do|wykorzystaj", topic, re.I):
                continue
            seen.add(n)
            yield {"src": f"{it['uid']}-t{n}", "old_topic": topic.strip()}


def open_questions(include_closed=False):
    """Old questions with an official answer and no picture (the API is text-only)."""
    for it in old_items():
        if (it["type"] == "open" or (include_closed and it["type"] == "closed")) and it["answer"] and not it["image_count"]:
            yield it


def gen_essay(client, model, effort, t):
    r = client.chat.completions.create(
        model=model, reasoning_effort=effort, max_completion_tokens=6000,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": ESSAY_SYSTEM},
                  {"role": "user", "content": ESSAY_USER.format(topic=t["old_topic"])}])
    out = json.loads(r.choices[0].message.content)
    return {**t, "topic": out["topic"], "essay": out["essay"], "words": len(out["essay"].split()),
            "model": model, "effort": effort, "usage": r.usage.model_dump(include={"prompt_tokens", "completion_tokens"})}


def gen_open(client, model, effort, it):
    prompt = EXAMINER_BOX.sub("", "\n" + it["prompt"]).strip()
    user = OPEN_USER.format(context=it["context"], id=it["id"], points=it["max_points"], prompt=prompt).strip()
    r = client.chat.completions.create(
        model=model, reasoning_effort=effort, max_completion_tokens=3000,
        messages=[{"role": "system", "content": OPEN_SYSTEM}, {"role": "user", "content": user}])
    return {"src": it["uid"], "question": user, "max_points": it["max_points"],
            "kind": "old_item", "type": it["type"],
            "answer": clean_md(r.choices[0].message.content), "official_answer": it["answer"],
            "official_rules": it["rules"], "model": model, "effort": effort,
            "usage": r.usage.model_dump(include={"prompt_tokens", "completion_tokens"})}


def style_examples(rng, k=3):
    pool = [it for it in old_items() if it["answer"] and not it["image_count"] and it["type"] != "essay"]
    out = []
    for it in rng.sample(pool, k):
        q = EXAMINER_BOX.sub("", "\n" + it["prompt"]).strip()
        out.append(f"[Zadanie ({it['max_points']} pkt)]\n{it['context'][:600]}\n{q}\n[Klucz] {it['answer'][:500]}\n[Zasady] "
                   f"{it['rules'].replace('Zasady oceniania', '').strip()[:300]}")
    return "\n\n".join(out)


def gen_source_tasks(client, model, effort, job):
    ex = job["excerpt"]
    user = SOURCE_USER.format(examples=job["examples"], site=ex["site"], title=ex["title"],
                              section=f", sekcja „{ex['section']}”" if ex.get("section") else "", text=ex["text"])
    r = client.chat.completions.create(
        model=model, reasoning_effort=effort, max_completion_tokens=6000, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SOURCE_SYSTEM}, {"role": "user", "content": user}])
    out = json.loads(r.choices[0].message.content)
    cite = (f"Na podstawie: Wikipedia, hasło „{ex['title']}”" if ex["site"] == "pl.wikipedia.org"
            else f"Źródło: Wikiźródła, „{ex['title']}”")
    context = f"{out.get('source_intro', 'Fragment źródła')}\n{out.get('excerpt') or ex['text']}\n{cite}"
    usage = r.usage.model_dump(include={"prompt_tokens", "completion_tokens"})
    recs = [{"job": job["src"], "src": f"{job['src']}-t{i}", "kind": "source_task", "context": context,
             "prompt": t["prompt"], "answer": clean_md(t["answer"]), "max_points": int(t.get("points", 1)),
             "type": t.get("type"), "task_type": t.get("task_type"), "rules": t.get("rules"),
             "meta": {"url": ex["url"], "revid": ex.get("revid"), "license": ex["license"], "era": ex["era"],
                      "model": model, "effort": effort, "usage": usage if i == 0 else None}}
            for i, t in enumerate(out.get("tasks", []))]
    return recs or [{"job": job["src"], "src": job["src"], "skipped": True, "meta": {"usage": usage}}]


def gen_topics(client, era, n, avoid=()):
    avoid_txt = ("Nie powtarzaj tych tematów ani tematów bardzo do nich podobnych (inna teza, inny problem):\n"
                 + "\n".join(f"- {a}" for a in avoid) + "\n") if avoid else ""
    r = client.chat.completions.create(
        model="gpt-6-luna", reasoning_effort="low", max_completion_tokens=6000, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": TOPICS_USER.format(era=era, n=n, avoid=avoid_txt)}])
    return json.loads(r.choices[0].message.content).get("topics", [])


def gen_essay_new(client, model, effort, t, v2=False):
    r = client.chat.completions.create(
        model=model, reasoning_effort=effort, max_completion_tokens=8000, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": ESSAY_SYSTEM},
                  {"role": "user", "content": (ESSAY_USER_V2 if v2 else ESSAY_USER_NEW).format(topic=t["topic"])}])
    essay = clean_md(json.loads(r.choices[0].message.content)["essay"])
    return {**t, "kind": "essay", "essay": essay, "words": len(essay.split()), "model": model, "effort": effort,
            "usage": r.usage.model_dump(include={"prompt_tokens", "completion_tokens"})}


class Budget:
    """Stops new API calls once the team ledger shows --max-usd spent since the start of this run."""

    def __init__(self, team, token, start_spent, max_usd):
        self.team, self.token, self.start, self.max = team, token, start_spent, max_usd
        self.spent, self.stopped, self.lock = 0.0, False, Lock()

    def check(self):
        if not self.max:
            return
        with self.lock:
            self.spent = balance(self.team, self.token)["spent_usd"] - self.start
            if self.spent >= self.max and not self.stopped:
                self.stopped = True
                print(f"  $ budget reached: ${self.spent:.2f} >= ${self.max:.2f}; no new requests", flush=True)


BUDGET = None


def run(kind, jobs, fn, out_file, args, client):
    done = {(lambda r: r.get("job", r["src"]))(json.loads(l)) for l in open(out_file)} if out_file.exists() else set()
    todo = [j for j in jobs if (j["src"] if "src" in j else j["uid"]) not in done]
    print(f"{kind}: {len(jobs)} requested, {len(jobs) - len(todo)} already done, {len(todo)} to generate", flush=True)
    lock = Lock()
    def guarded(client, model, effort, job):
        if BUDGET and BUDGET.stopped:
            return []
        return fn(client, model, effort, job)

    with ThreadPoolExecutor(args.parallel) as pool, open(out_file, "a") as f:
        futs = {pool.submit(guarded, client, args.model, args.effort, j): j for j in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:  # keep going; rerun to retry failures
                print(f"  ! {kind} {futs[fut].get('src', futs[fut].get('uid'))}: {e}", flush=True)
                continue
            recs = rec if isinstance(rec, list) else [rec]
            if not recs:  # skipped: budget reached
                continue
            if BUDGET and i % 10 == 0:
                BUDGET.check()
            with lock:
                for r_ in recs:
                    f.write(json.dumps(r_, ensure_ascii=False) + "\n")
                f.flush()
            print(f"  [{i}/{len(todo)}] {kind} {recs[0]['src']} -> {len(recs)} record(s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--essays", type=int, default=0)
    ap.add_argument("--open", type=int, default=0)
    ap.add_argument("--old-all", action="store_true", help="all old open+closed text-only questions")
    ap.add_argument("--source-tasks", type=int, default=0, help="number of source excerpts to write tasks for")
    ap.add_argument("--new-essays-per-era", type=int, default=0)
    ap.add_argument("--eras", type=int, default=0, help="limit eras (test runs); 0 = all")
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--effort", default="medium", help="reasoning effort: none/low/medium/high")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=0, help="stop new requests once this much is spent (0 = no cap)")
    ap.add_argument("--essay-v2", action="store_true", help="v2 essay prompt (deeper argumentation checklist)")
    ap.add_argument("--avoid-topics", nargs="*", default=[],
                    help="topics.jsonl files whose topics (per era) new topics must not repeat")
    args = ap.parse_args()

    out = DATA / "synthetic" / args.run
    out.mkdir(parents=True, exist_ok=True)
    client, team, token = llm_client()
    before = balance(team, token)
    global BUDGET
    BUDGET = Budget(team, token, before["spent_usd"], args.max_usd) if args.max_usd else None
    rng = random.Random(args.seed)

    if args.essays:
        topics = list(essay_topics())
        rng.shuffle(topics)
        run("essay", topics[:args.essays], gen_essay, out / "essays.jsonl", args, client)
    if args.open:
        qs = [{**q, "src": q["uid"]} for q in open_questions()]
        rng.shuffle(qs)
        run("open", qs[:args.open], gen_open, out / "open.jsonl", args, client)

    if args.old_all:
        qs = [{**q, "src": q["uid"]} for q in open_questions(include_closed=True)]
        run("old", qs, gen_open, out / "old_items.jsonl", args, client)
    if args.source_tasks:
        excerpts = [json.loads(l) for l in open(DATA / "sources" / "excerpts.jsonl")]
        rng.shuffle(excerpts)
        jobs = [{"src": e["id"], "excerpt": e, "examples": style_examples(rng)} for e in excerpts[:args.source_tasks]]
        run("source", jobs, gen_source_tasks, out / "source_tasks.jsonl", args, client)
    if args.new_essays_per_era:
        from fetch_sources import ERAS
        topics_file = out / "topics.jsonl"
        if not topics_file.exists():
            eras = ERAS[:args.eras] if args.eras else ERAS
            avoid = {}
            for f_ in args.avoid_topics:
                for l in open(DATA / f_):
                    t = json.loads(l)
                    avoid.setdefault(t["era"], []).append(t["topic"])
            always = PRACTICE_ESSAY_TOPICS
            with ThreadPoolExecutor(args.parallel) as pool:
                lists = list(pool.map(lambda e: (e, gen_topics(client, e, args.new_essays_per_era,
                                                               avoid.get(e, []) + always)), eras))
            with open(topics_file, "w") as f:
                for era, ts in lists:
                    for j, t in enumerate(ts):
                        f.write(json.dumps({"src": f"topic-{ERAS.index(era):02d}-{j:02d}", "era": era, "topic": t},
                                           ensure_ascii=False) + "\n")
        topics = [json.loads(l) for l in open(topics_file)]
        rng.shuffle(topics)  # a budget stop then leaves all eras about equally covered
        fn = (lambda c, m, e, t: gen_essay_new(c, m, e, t, v2=True)) if args.essay_v2 else gen_essay_new
        run("essay-new", topics, fn, out / "essays_new.jsonl", args, client)

    time.sleep(3)  # let the ledger settle final usage
    after = balance(team, token)
    cost = {"before": before, "after": after, "this_run_usd": round(after["spent_usd"] - before["spent_usd"], 4),
            "args": vars(args)}
    with open(out / "cost.jsonl", "a") as f:
        f.write(json.dumps(cost) + "\n")
    print(f"\ncost of this run: ${cost['this_run_usd']:.2f}; available: ${after['available_usd']:.2f}")


if __name__ == "__main__":
    main()
