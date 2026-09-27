"""Turn a harness run into the organisers' answers.json and check it against their rules.

  python scripts/write_answers.py --run pkg2023-base --exam exams/history-2023-mock-v1

Reads runs/<run>/answers.jsonl, writes runs/<run>/answers.json:
  {"exam_id": "...", "answers": [{"id": "1", "answer": "..."}, ...]}   (every template id exactly once, strings)
Closed answers are converted from our "ODPOWIEDŹ: 1 – P, 2 – F" line into the item's answer_format syntax
("1: P\\n2: F\\n3: P", "A", "A: 3\\nB: 2"); open answers and the essay are plain text without markdown.
Checks mirror the organisers' README: exam_id, all ids once, strings only, <= 100,000 chars each, <= 1 MiB.
"""

import argparse
import json
import re

from run_exam import DATA, answer_marks, final_line, load_package

MAX_ANSWER, MAX_FILE = 100_000, 1024 * 1024


def plain(text):
    """Exam answers are plain text: drop markdown emphasis/headings and the harness' ODPOWIEDŹ line label."""
    text = re.sub(r"(\*\*|__|^#+\s*)", "", text, flags=re.M)
    return text.strip()


def closed_to_format(answer, fmt):
    """Our answer -> the item's answer_format syntax, or None if the marks cannot be read."""
    marks = answer_marks(answer)
    if not marks:
        return None
    fmt_keys = [line.split(":")[0].strip() for line in fmt.splitlines() if ":" in line]
    if not fmt_keys:  # single choice, format "A"
        letter = marks.get("_") or (next(iter(marks.values())) if len(marks) == 1 else None)
        return letter
    if not all(k in marks for k in fmt_keys):
        return None
    return "\n".join(f"{k}: {marks[k].upper()}" for k in fmt_keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--exam", required=True, help="package dir with exam.json and answers-template.json")
    args = ap.parse_args()

    exam_dir = DATA / args.exam
    exam = json.loads((exam_dir / "exam.json").read_text(encoding="utf-8"))
    template = json.loads((exam_dir / "answers-template.json").read_text(encoding="utf-8"))
    items = {it["id"]: it for it in load_package(args.exam)}
    got = {}
    for line in open(DATA / "runs" / args.run / "answers.jsonl"):
        rec = json.loads(line)
        got[rec["uid"].split("-", 1)[-1] if rec["uid"] not in items else rec["uid"]] = rec["answer"]

    answers, problems = [], []
    for slot in template["answers"]:
        qid, it = slot["id"], items[slot["id"]]
        raw = got.get(qid, "")
        if not raw.strip():
            problems.append(f"{qid}: empty")
            text = ""
        elif it["type"] == "closed":
            text = closed_to_format(raw, it["answer_format"])
            if text is None:
                problems.append(f"{qid}: closed answer not in expected syntax, sent as text: {final_line(raw)[:60]!r}")
                text = plain(final_line(raw)).removeprefix("ODPOWIEDŹ:").strip()
        else:
            text = plain(raw)
            if it["type"] == "essay":
                words = len(text.split())
                if not re.search(r"\bTemat\s*(nr\.?)?\s*\d", text[:200], re.I):
                    problems.append(f"{qid}: essay does not start with the topic number")
                if words < 300:
                    problems.append(f"{qid}: essay has only {words} words (<300 => B = 0)")
        if len(text) > MAX_ANSWER:
            problems.append(f"{qid}: {len(text)} chars > {MAX_ANSWER}")
        answers.append({"id": qid, "answer": text})

    out = {"exam_id": exam["exam_id"], "answers": answers}
    data = json.dumps(out, ensure_ascii=False, indent=2)
    ids = [a["id"] for a in answers]
    assert out["exam_id"] == template["exam_id"]
    assert sorted(ids) == sorted(a["id"] for a in template["answers"]) and len(ids) == len(set(ids))
    assert all(isinstance(a["answer"], str) for a in answers)
    assert len(data.encode()) <= MAX_FILE, "file > 1 MiB"
    path = DATA / "runs" / args.run / "answers.json"
    path.write_text(data + "\n", encoding="utf-8")
    print(f"{path}: {len(answers)} answers, {sum(not a['answer'] for a in answers)} blank, "
          f"{len(data.encode()) / 1024:.0f} KiB")
    for p in problems:
        print("  !", p)


if __name__ == "__main__":
    main()
