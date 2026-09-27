"""Convert a CKE Formuła 2023 paper into the organisers' exam-package format (text + cropped illustrations).

The real exam arrives as exam.json + images/*.png + answers-template.json ("separate-text-and-images-v1").
To test and train in exactly that format we build the same package from our practice papers:
  - text and task split come from the same PDF parsing as build_practice_set.py ("Zadanie N." headers);
  - every raster illustration is cropped from the PDF page at 200 dpi (labels drawn over it are included)
    and an "[Obraz: images/ZNN-k.png]" marker is put into the text where the picture stands;
  - answer_format is derived from the official key (P/F, single choice, matching) — free text otherwise.

  python scripts/build_package.py --year 2024          -> $DATA_DIR/exams/practice-2024/
  python scripts/build_package.py --year 2023 --compare exams/history-2023-mock-v1   (check against organisers)

2026 is the held-out paper: refused unless ALLOW_HELDOUT=1.
"""

import argparse
import hashlib
import json
import os
import re

import pymupdf

from build_practice_set import DATA, HEADER, NOISE
from run_exam import parse_marks

FREE_TEXT = "Tekst po polsku. Podaj wszystkie wymagane elementy odpowiedzi."
ESSAY_FMT = "Jeden tekst: numer wybranego tematu i całe wypracowanie. Minimum 300 wyrazów zgodnie z poleceniem."
DPI = 200


def clean(line):
    line = re.sub(r"[.…]{4,}", " ", line).strip()
    return "" if not line or any(p.match(line) for p in NOISE) else line


def vector_figures(page, rasters, gap=30):
    """Drawn figures (e.g. genealogy charts: boxes joined by connector lines, short labels).

    Tables and framed text boxes are made of rectangles only and stay text (as in the organisers' package);
    examiner boxes (right margin), coloured header bars and full-width rules are ignored."""
    groups = []
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        fill, stroke = d.get("fill"), d.get("color")
        if stroke and max(stroke) - min(stroke) > 0.15:
            continue  # coloured (purple) examiner boxes and header frames; figures are drawn in black/grey
        if (r.width <= 0 and r.height <= 0) or r.x0 > 525 or (r.height < 1.5 and r.width > 400) or \
                (fill and r.width > 300 and not all(c > 0.95 for c in fill) and not all(c < 0.05 for c in fill)):
            continue
        g = {"bbox": pymupdf.Rect(r), "l": sum(i[0] == "l" for i in d["items"]),
             "re": sum(i[0] == "re" for i in d["items"])}
        for h in [h for h in groups if h["bbox"].intersects(r + (-gap, -gap, gap, gap))]:
            g["bbox"].include_rect(h["bbox"]); g["l"] += h["l"]; g["re"] += h["re"]; groups.remove(h)
        groups.append(g)
    area = page.rect.width * page.rect.height
    out = []
    for g in groups:
        b = g["bbox"]
        if b.get_area() / area < 0.03 or g["l"] < 2 or g["re"] < 3:
            continue
        if any((b & r).get_area() > 0.3 * b.get_area() for r in rasters):
            continue  # vector overlay on a photo/map that is cropped anyway
        for l in (l for blk in page.get_text("dict")["blocks"] for l in blk.get("lines", [])):
            text = "".join(sp["text"] for sp in l["spans"]).strip()
            if HEADER.match(text) and b.y0 < l["bbox"][1] < b.y1:
                b = pymupdf.Rect(b.x0, b.y0, b.x1, l["bbox"][1] - 2)  # never swallow a task header
        lines = [l for l in page.get_text("text", clip=b).split("\n") if l.strip()]
        if not lines or sum(map(len, lines)) / len(lines) > 30:
            continue  # prose in a frame, not a figure
        out.append(b + (-4, -4, 4, 4))
    return out


def page_stream(page, min_frac=0.02):
    """Lines of a page in text order, with ('img', rect) entries inserted where each illustration stands."""
    area = page.rect.width * page.rect.height
    imgs = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()
            if pymupdf.Rect(i["bbox"]).get_area() / area >= min_frac]
    imgs = sorted(imgs + vector_figures(page, imgs), key=lambda r: (r.y0, r.x0))
    imgs = [r for r in imgs  # drop pieces lying (mostly) inside a bigger picture
            if not any(o != r and o.get_area() > r.get_area() and (r & o).get_area() > 0.8 * r.get_area() for o in imgs)]
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            text = clean("".join(s["text"] for s in l["spans"]))
            if text and not any(pymupdf.Rect(l["bbox"]) in r for r in imgs):  # labels inside a picture stay in it
                lines.append((l["bbox"][1], text))
    out, pending = [], list(imgs)
    for y, text in lines:
        while pending and pending[0].y0 <= y:
            out.append(("img", pending.pop(0)))
        out.append(("txt", text))
    out += [("img", r) for r in pending]
    return out


def sections(pdf, first_page=1):
    """Split the paper at 'Zadanie N.' headers; each section keeps text lines and illustration entries."""
    secs, cur = [], None
    for pno in range(first_page, pdf.page_count):
        for kind, val in page_stream(pdf[pno]):
            if kind == "txt" and (m := HEADER.match(val)):
                g, sub, _, hi = m.groups()
                cur = {"group": int(g), "sub": int(sub) if sub else None, "max_points": int(hi) if hi else None,
                       "parts": []}
                secs.append(cur)
            elif cur is not None:
                cur["parts"].append((kind, val, pno))
    return secs


def answer_format(key, max_points):
    if max_points >= 10:
        return ESSAY_FMT
    first = key.strip().split("\n\n")[0]
    marks = parse_marks(first) if first else None
    if not marks or any(len(v) > 1 and not v.isdigit() for v in marks.values()):
        return FREE_TEXT
    if "_" in marks:
        return "A"
    vals = list(marks.values())
    if all(v in "PF" for v in vals):
        return "\n".join(f"{k}: {'P' if i % 2 == 0 else 'F'}" for i, k in enumerate(marks))
    return "\n".join(f"{k}: {'1' if v.isdigit() else 'A'}" for k, v in marks.items())


def build(year, out_dir):
    if year == 2026 and not os.environ.get("ALLOW_HELDOUT"):
        raise SystemExit("2026 is the held-out paper; set ALLOW_HELDOUT=1 only for the final evaluation.")
    pdf = pymupdf.open(DATA / "raw/cke/f2023" / f"{year}-arkusz.pdf")
    keys = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "practice" / str(year) / "items.jsonl")}
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    shared, counters, items = {}, {}, []

    def render(parts, group):
        text, images = [], []
        for kind, val, pno in parts:
            if kind == "txt":
                text.append(val)
                continue
            counters[group] = counters.get(group, 0) + 1
            name = f"images/Z{group:02d}-{counters[group]}.png"
            pdf[pno].get_pixmap(clip=val, dpi=DPI).save(out_dir / name)
            images.append({"path": name, "source_page": pno + 1,
                           "sha256": hashlib.sha256((out_dir / name).read_bytes()).hexdigest()})
            text.append(f"[Obraz: {name}]")
        return "\n".join(text).strip(), images

    for s in sections(pdf):
        g = s["group"]
        if s["max_points"] is None:  # group header: shared sources for the sub-tasks
            shared[g] = render(s["parts"], g)
            continue
        item_id = f"{g}.{s['sub']}" if s["sub"] else str(g)
        question, own_images = render(s["parts"], g)
        src_text, src_images = shared.get(g, ("", [])) if s["sub"] else ("", [])
        if s["max_points"] >= 10:
            question = re.split(r"\nWYPRACOWANIE\b", question)[0].strip()
        question = re.sub(r"\nA(\n[B-F])+$", "", question)  # stray picture labels "A", "B"… after the prompt
        key = keys.get(item_id, {})
        items.append({"id": item_id, "group": g, "max_points": s["max_points"], "question": question,
                      "source_text": src_text, "images": src_images + own_images,
                      "answer_format": answer_format(key.get("answer", ""), s["max_points"])})

    exam = {"exam_id": f"practice-{year}-v1", "title": f"Historia — matura rozszerzona, maj {year} (nasza konwersja)",
            "input_format": "separate-text-and-images-v1", "language": "pl",
            "max_points": sum(i["max_points"] for i in items),
            "instructions": "Rozwiąż zadania po polsku. Czytaj question, source_text i wszystkie powiązane images. "
                            "Wzory answer_format pokazują tylko składnię, nie poprawne odpowiedzi.",
            "items": items}
    (out_dir / "exam.json").write_text(json.dumps(exam, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "answers-template.json").write_text(json.dumps(
        {"exam_id": exam["exam_id"], "answers": [{"id": i["id"], "answer": ""} for i in items]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return exam


def compare(ours, theirs_dir):
    theirs = json.loads((DATA / theirs_dir / "exam.json").read_text(encoding="utf-8"))
    t = {i["id"]: i for i in theirs["items"]}
    o = {i["id"]: i for i in ours["items"]}
    print(f"items: ours {len(o)}, theirs {len(t)}; same ids: {sorted(o) == sorted(t)}")
    print(f"images: ours {sum(len(i['images']) for i in o.values())}, theirs {sum(len(i['images']) for i in t.values())}")
    for k in t:
        oi, ti = o.get(k), t[k]
        if not oi:
            print(f"  {k}: missing in ours")
            continue
        diffs = []
        if len(oi["images"]) != len(ti["images"]):
            diffs.append(f"images {len(oi['images'])} vs {len(ti['images'])} "
                         f"(pages ours {[i['source_page'] for i in oi['images']]}, theirs {[i['source_page'] for i in ti['images']]})")
        if oi["answer_format"] != ti["answer_format"]:
            diffs.append(f"format {oi['answer_format']!r} vs {ti['answer_format']!r}")
        if diffs:
            print(f"  {k}: " + "; ".join(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", help="default: exams/practice-<year>")
    ap.add_argument("--compare", help="organisers' package dir to compare with (2023 mock)")
    args = ap.parse_args()
    out = DATA / (args.out or f"exams/practice-{args.year}")
    exam = build(args.year, out)
    print(f"{out}: {len(exam['items'])} items, {exam['max_points']} pts, "
          f"{sum(len(i['images']) for i in exam['items'])} image refs")
    if args.compare:
        compare(exam, args.compare)


if __name__ == "__main__":
    main()
