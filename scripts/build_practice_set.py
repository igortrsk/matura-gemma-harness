"""Split CKE history matura papers into scored items with marking guidance.

Formuła 2023 (practice papers, default):
  Input:  $DATA_DIR/raw/cke/f2023/{year}-arkusz.pdf, {year}-zasady.pdf  (fetch with scripts/fetch_cke.sh)
  Output: $DATA_DIR/practice/{year}/items.jsonl   one line per scored item
          $DATA_DIR/practice/{year}/pages/pNN.png  page renders for image-input runs
Formuła 2015 (older format, training/synthetic-data material only), with --old:
  Input:  $DATA_DIR/raw/cke/f2015/{year}-arkusz.pdf [+ {year}-zasady.pdf when published]
  Output: $DATA_DIR/f2015/{year}/items.jsonl      (no page renders)
DATA_DIR defaults to /data if mounted, else ./data.
"""

import json
import os
import re
import sys
from pathlib import Path

import pymupdf

# Prefer the Docker volume at /data when mounted, so large files stay off the host folder.
DATA = Path(os.environ.get("DATA_DIR") or ("/data" if Path("/data").is_dir() else "data"))
FORMATS = {  # raw input folder, output folder, render page images?
    "f2023": (DATA / "raw/cke/f2023", DATA / "practice", True),
    "f2015": (DATA / "raw/cke/f2015", DATA / "f2015", False),
}
DPI = 150

HEADER = re.compile(r"^Zadanie (\d+)(?:\.(\d+))?\.?\s*(?:\((\d+)\s*[–-]\s*(\d+)\))?\s*$")
NOISE = [
    re.compile(r"^Strona \d+ z \d+$"),
    re.compile(r"^M[HI]{2}P?-R0[-_]\d+.*$|^[EM]HI[P_-].*$"),  # sheet codes: MHIP-R0_100, MHI_1R, EHIP-R0-100
    re.compile(r"^[.…\s•]*$"),  # empty / dotted answer lines
    re.compile(r"^\d+(\.\d+)*\.$"),  # examiner box: "3.2."
    re.compile(r"^\d+\s*[–-]\s*\d+$"),  # examiner box: "0–1"
    re.compile(r"^Egzamin maturalny z historii – .*$"),
    re.compile(r"^Zasady oceniania rozwiązań zadań$"),
    re.compile(r"^Układ graficzny$|^© CKE \d+$"),
]
# Marking-guide block labels. Formuła 2023: "Zasady oceniania" / "Rozwiązanie";
# Formuła 2015: "Schemat punktowania" / "Prawidłowa odpowiedź" (2015 prints the answer before the rules).
LABEL = re.compile(r"^(Zasady oceniania|Schemat punktowania|Przykładowe rozwiązani[ea]|Rozwiązanie"
                   r"|Prawidłowa odpowiedź|Poprawna odpowiedź|Przykładowa odpowiedź|Przykładowe odpowiedzi)\s*$", re.M)
CLOSED_HINTS = re.compile(
    r"Oceń prawdziwość|Zaznacz|Przyporządkuj|Uporządkuj|Uzupełnij tabelę|Wybierz|wpisz literę|P – jeżeli",
    re.I,
)


def clean_lines(text):
    out = []
    for raw in text.splitlines():
        line = re.sub(r"[.…]{4,}", " ", raw).strip()
        if not line or any(p.match(line) for p in NOISE):
            continue
        out.append(line)
    return out


def big_images(page, min_frac=0.02):
    """Raster images covering a meaningful share of the page (skips logos, boxes)."""
    area = page.rect.width * page.rect.height
    return sum(1 for info in page.get_image_info() if pymupdf.Rect(info["bbox"]).get_area() / area >= min_frac)


def walk(pdf, first_page=0):
    """Yield (page_no, line) over the document, 1-based pages."""
    for i in range(first_page, pdf.page_count):
        for line in clean_lines(pdf[i].get_text()):
            yield i + 1, line


def split_sections(pdf, first_page=0):
    """Split into sections at 'Zadanie N.' / 'Zadanie N.M. (0–k)' headers."""
    sections, cur = [], None
    for page, line in walk(pdf, first_page):
        m = HEADER.match(line)
        if m:
            group, sub, lo, hi = m.groups()
            cur = {"group": int(group), "sub": int(sub) if sub else None,
                   "max_points": int(hi) if hi else None, "pages": [page], "lines": []}
            sections.append(cur)
            continue
        if cur is None:
            continue
        if page not in cur["pages"]:
            cur["pages"].append(page)
        cur["lines"].append(line)
    return sections


def parse_paper(pdf):
    """Return scored items, each carrying its group's shared sources as context."""
    items, context, context_pages = [], {}, {}
    for s in split_sections(pdf, first_page=1):  # skip cover; 2026 starts tasks on page 3
        text = "\n".join(s["lines"]).strip()
        if s["max_points"] is None:  # group header: shared sources for its subtasks
            context[s["group"]], context_pages[s["group"]] = text, s["pages"]
            continue
        g = s["group"]
        item_id = f"{g}.{s['sub']}" if s["sub"] else str(g)
        pages = sorted(set(context_pages.get(g, []) + s["pages"]))
        items.append({"id": item_id, "group": g, "max_points": s["max_points"],
                      "context": context.get(g, "") if s["sub"] else "",
                      "prompt": text, "pages": pages})
    return items


def parse_marking(pdf):
    """Map item id -> {'rules', 'answer'} and return the essay rubric pages' text."""
    marking = {}
    for s in split_sections(pdf, first_page=1):
        if s["max_points"] is None:
            continue
        item_id = f"{s['group']}.{s['sub']}" if s["sub"] else str(s["group"])
        text = "\n".join(s["lines"])
        # Keep the scoring-rules and model-answer blocks, in whatever order the guide prints them;
        # drop the curriculum-requirements preamble before the first label.
        blocks = {"rules": [], "answer": []}
        labels = list(LABEL.finditer(text))
        for m, nxt in zip(labels, labels[1:] + [None]):
            kind = "rules" if m.group(1) in ("Zasady oceniania", "Schemat punktowania") else "answer"
            blocks[kind].append(text[m.end():nxt.start() if nxt else len(text)].strip())
        rules = "\n".join(blocks["rules"])
        marking[item_id] = {"rules": f"Zasady oceniania\n{rules}" if rules else "",
                            "answer": "\n".join(blocks["answer"])}
        if s["max_points"] >= 10:  # essay numbering can differ between paper and guide (2025: 25 vs 26)
            marking["essay"] = marking[item_id]
    return marking


def essay_rubric(pdf):
    start = next((i for i in range(pdf.page_count) if "NARRACJA HISTORYCZNA" in pdf[i].get_text()), None)
    if start is None:
        return ""
    return "\n".join(line for _, line in walk(pdf, start))


def classify(item):
    if item["max_points"] >= 10:
        return "essay"
    return "closed" if CLOSED_HINTS.search(item["prompt"]) else "open"


def build(year, fmt="f2023"):
    raw, out_root, render = FORMATS[fmt]
    paper = pymupdf.open(raw / f"{year}-arkusz.pdf")
    guide_path = raw / f"{year}-zasady.pdf"  # some older years have no published guide
    guide = pymupdf.open(guide_path) if guide_path.exists() else None
    items = parse_paper(paper)
    marking, rubric = (parse_marking(guide), essay_rubric(guide)) if guide else ({}, "")

    out = out_root / str(year)
    out.mkdir(parents=True, exist_ok=True)
    if render:
        (out / "pages").mkdir(exist_ok=True)
        for p in sorted({p for it in items for p in it["pages"]}):
            png = out / "pages" / f"p{p:02d}.png"
            if not png.exists():
                paper[p - 1].get_pixmap(dpi=DPI).save(png)

    missing = []
    with open(out / "items.jsonl", "w") as f:
        for it in items:
            mk = marking.get(it["id"]) or (marking.get("essay") if it["max_points"] >= 10 else None)
            if mk is None:
                missing.append(it["id"])
            it.update({
                "uid": f"{fmt}-{year}-{it['id']}" if fmt != "f2023" else f"{year}-{it['id']}",
                "year": year, "format": fmt, "type": classify(it),
                "image_count": sum(big_images(paper[p - 1]) for p in it["pages"]),
                "page_images": [f"pages/p{p:02d}.png" for p in it["pages"]] if render else [],
                "rules": mk["rules"] if mk else "", "answer": mk["answer"] if mk else "",
            })
            if it["type"] == "essay":
                it["rules"] = (it["rules"] + "\n\n" + rubric).strip()
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    total = sum(it["max_points"] for it in items)
    by_type = {t: sum(it["max_points"] for it in items if it["type"] == t) for t in ("closed", "open", "essay")}
    answered = sum(1 for it in items if it["answer"])
    print(f"{fmt} {year}: {len(items)} items, {total} pts {by_type}, with images: "
          f"{sum(1 for it in items if it['image_count'])}, with model answer: {answered}, "
          f"missing marking: {'no guide' if guide is None else (missing or 'none')}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--old"]:
        for y in args[1:] or ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]:
            build(int(y), "f2015")
    else:
        for y in args or ["2023", "2024", "2025", "2026"]:
            build(int(y))
