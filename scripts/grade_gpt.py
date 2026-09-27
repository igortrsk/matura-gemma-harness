"""Grade runs with GPT-6 via the Forgehand team API (alternative to grade_run.sh), same rules as scripts/GRADING.md, one item per call.

  python scripts/grade_gpt.py sd-2023 hv1-2023 ...        -> runs/<run>/grades-gpt.jsonl (resumable; grades.jsonl untouched)

Closed items with auto_points keep them. Compare runs only with grades from the SAME grader (both sides graded here).
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic import llm_client  # noqa: E402

DATA = Path("/data")
SYSTEM = """Jesteś egzaminatorem CKE oceniającym jedną odpowiedź z matury z historii (poziom rozszerzony, Formuła 2023).
Przyznaj punkty DOKŁADNIE według oficjalnych zasad oceniania (rules) — liczba całkowita od 0 do max_points. Oficjalne rozwiązanie
jest tylko przykładem: akceptuj każdą odpowiedź merytorycznie poprawną i spełniającą warunki zadania (synonimy, inne sformułowania,
równoważne nazwy). Sprawdzaj rygorystycznie: „rozstrzygnięcie + uzasadnienie” — wymagane oba, poprawne rozstrzygnięcie bez
poprawnego uzasadnienia = 0 pkt; „odwołując się do źródła” — uzasadnienie musi faktycznie korzystać ze źródła; wymagana liczba
elementów — liczą się tylko poprawne, sprzeczne dodatkowe elementy mogą unieważnić punkt; błędy merytoryczne w ocenianej treści
obniżają ocenę. Pusta odpowiedź = 0 pkt. Odpowiedź ucięta (finish_reason = length): oceń to, co jest.
Wypracowanie (15 pkt): kryterium A – narracja historyczna 0–12 według rules (stanowisko wobec tezy; funkcjonalne wykorzystanie
wiedzy w odniesieniu do TRZECH elementów tematu; argumentacja bogata / zadowalająca / powierzchowna; −1 pkt za 1–2 błędy
merytoryczne, −2 za 3–5, −3 za więcej niż 5); kryterium B – spójność 0–3; mniej niż 300 słów ⇒ B = 0 (liczbę słów podano).
Kategorie straty: knowledge, image, source, task, format, reasoning, essay-structure, essay-argument, essay-facts, essay-length,
truncated, other (null, jeśli bez straty).
Zwróć TYLKO JSON: {"points": <int>, "category": <str|null>, "reason": "<jedno krótkie zdanie po polsku>"} — dla wypracowania
dodaj "essay": {"topic": <int>, "A": <int>, "B": <int>, "factual_errors": <int>} i points = A + B."""


def items_for(run):
    cfg = json.loads((DATA / "runs" / run / "config.json").read_text())
    key = cfg.get("key_year") or ""
    if "2026" in str(key) or "2026" in str(cfg.get("years")):
        raise SystemExit(f"{run}: 2026 is held out — not graded here")
    return {json.loads(l)["uid"]: json.loads(l) for l in open(DATA / "practice" / str(key) / "items.jsonl")}


def grade_one(client, args, it, ans):
    words = len(ans.get("answer", "").split())
    user = (f"Polecenie (typ {it['type']}, max_points {it['max_points']}):\n{it['prompt']}\n\n"
            f"Źródła (context):\n{it.get('context', '')[:6000]}\n\nOficjalne zasady oceniania (rules):\n{it.get('rules', '')}\n\n"
            f"Oficjalne przykładowe rozwiązanie:\n{it.get('answer', '')}\n\n"
            f"ODPOWIEDŹ ZDAJĄCEGO (finish_reason = {ans.get('finish_reason')}, słów: {words}):\n{ans.get('answer', '')}")
    r = client.chat.completions.create(model=args.model, reasoning_effort=args.effort, max_completion_tokens=4000,
                                       response_format={"type": "json_object"},
                                       messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    g = json.loads(r.choices[0].message.content)
    mx = int(it["max_points"])
    g["points"] = max(0, min(mx, int(g.get("points", 0))))
    if not ans.get("answer", "").strip():
        g.update(points=0, category="truncated", reason="pusta odpowiedź")
    if it["type"] == "essay":
        g.setdefault("essay", {})["words"] = words
    return g, r.usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    client, team, token = llm_client()
    tok = {"in": 0, "out": 0}
    lock = Lock()
    for run in args.runs:
        d = DATA / "runs" / run
        items = items_for(run)
        answers = {json.loads(l)["uid"]: json.loads(l) for l in open(d / "answers.jsonl")}
        auto = {json.loads(l)["uid"]: json.loads(l).get("auto_points") for l in open(d / "auto_scores.jsonl")} \
            if (d / "auto_scores.jsonl").exists() else {}
        out = d / "grades-gpt.jsonl"
        done = {json.loads(l)["uid"] for l in open(out)} if out.exists() else set()
        todo = [u for u in answers if u not in done]

        def work(u):
            it, ans = items[u], answers[u]
            base = {"uid": u, "type": it["type"], "max": int(it["max_points"]), "image_count": it.get("image_count")}
            if auto.get(u) is not None:
                rec = {**base, "points": auto[u], "source": "auto", "category": None, "reason": "auto"}
            else:
                try:
                    g, usage = grade_one(client, args, it, ans)
                except Exception as e:
                    print(f"  ! {run} {u}: {e}")
                    return
                rec = {**base, **g, "source": "gpt", "grader": args.model}
                with lock:
                    tok["in"] += usage.prompt_tokens
                    tok["out"] += usage.completion_tokens
            with lock, open(out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(args.parallel) as pool:
            list(pool.map(work, todo))
        g = [json.loads(l) for l in open(out)]
        print(f"{run}: {sum(x['points'] for x in g)}/{sum(x['max'] for x in g)} | graded {len(g)}/{len(answers)}")
    cost = tok["in"] / 1e6 * 1.25 + tok["out"] / 1e6 * 10
    print(f"tokens in {tok['in']} out {tok['out']} ≈ ${cost:.2f} (sol prices)")


if __name__ == "__main__":
    main()
