"""Collect real Polish-language source texts for synthetic matura tasks.

For each period of the history curriculum, GPT-6 Luna proposes key Wikipedia topics and well-known
historical documents; every proposal is resolved against the live sites (search API) and only pages
that exist are kept. Texts are cut into short excerpts, like the sources quoted in CKE papers.

  pl.wikipedia.org  — CC BY-SA 4.0 (derived data must keep CC BY-SA + attribution)
  pl.wikisource.org — public-domain texts

Output: $DATA_DIR/sources/excerpts.jsonl  {id, site, title, url, revid, license, era, text}
Run:    python scripts/fetch_sources.py [--per-era-wiki 18 --per-era-docs 6]
"""

import argparse
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic import DATA, llm_client  # noqa: E402

UA = "warsaw-model-trainers-hackathon/0.1 (matura research; public data only)"
ERAS = [
    "pradzieje i cywilizacje starożytnego Wschodu", "starożytna Grecja", "starożytny Rzym",
    "Bizancjum i świat islamu", "Europa wczesnego średniowiecza", "Polska pierwszych Piastów (X–XII w.)",
    "Europa pełnego i późnego średniowiecza (feudalizm, papiestwo, krucjaty, miasta)",
    "Polska w XIII–XV w. (rozbicie dzielnicowe, zjednoczenie, Jagiellonowie)",
    "wielkie odkrycia geograficzne, renesans i reformacja", "Rzeczpospolita w XVI w. (demokracja szlachecka, unie, elekcje)",
    "Europa w XVII w. (absolutyzm, rewolucja angielska, wojna trzydziestoletnia)",
    "Rzeczpospolita w XVII w. (wojny, potop, kryzys)", "oświecenie oraz rewolucje amerykańska i francuska",
    "Rzeczpospolita w XVIII w. (czasy saskie, stanisławowskie, rozbiory, insurekcja kościuszkowska)",
    "epoka napoleońska i Księstwo Warszawskie, kongres wiedeński",
    "ziemie polskie 1815–1864 (Królestwo Polskie, powstania listopadowe i styczniowe, Wielka Emigracja)",
    "Europa i świat w XIX w. (rewolucja przemysłowa, zjednoczenie Włoch i Niemiec, wojna secesyjna, kolonializm, ideologie)",
    "ziemie polskie 1864–1914 (praca organiczna, rusyfikacja, germanizacja, partie polityczne)",
    "I wojna światowa i odzyskanie niepodległości przez Polskę", "świat w okresie międzywojennym (totalitaryzmy, wielki kryzys)",
    "II Rzeczpospolita (1918–1939)", "II wojna światowa na świecie", "Polska pod okupacją i Polskie Państwo Podziemne (1939–1945)",
    "świat po 1945 r. (zimna wojna, dekolonizacja, integracja europejska)", "Polska Ludowa 1944–1989",
    "upadek komunizmu i przemiany po 1989 r. w Polsce i Europie",
]

PROPOSE = """Okres: {era}
Na potrzeby matury z historii (poziom rozszerzony, Polska) podaj:
1. "wiki": {n_wiki} tytułów haseł polskiej Wikipedii o kluczowych wydarzeniach, procesach, postaciach, pojęciach i dokumentach tego okresu (mieszanka historii Polski i powszechnej, jak w podstawie programowej);
2. "docs": {n_docs} znanych tekstów źródłowych z epoki w języku polskim (dokumenty, akty prawne, traktaty, odezwy, kroniki, pamiętniki, listy, mowy), które mogą być dostępne w domenie publicznej na polskiej Wikiźródłach.
Zwróć JSON: {{"wiki": ["..."], "docs": ["..."]}}"""


def api(site, **params):
    url = f"https://{site}/w/api.php?" + urlencode({**params, "format": "json"})
    for attempt in range(4):
        try:
            return json.load(urlopen(Request(url, headers={"User-Agent": UA}), timeout=30))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {}


def resolve(site, query):
    hits = api(site, action="query", list="search", srsearch=query, srlimit=1).get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def wikipedia_text(title):
    q = api("pl.wikipedia.org", action="query", prop="extracts|revisions", explaintext=1, redirects=1,
            titles=title, rvprop="ids")
    page = next(iter(q.get("query", {}).get("pages", {}).values()), {})
    return page.get("title"), page.get("extract", ""), (page.get("revisions") or [{}])[0].get("revid")


def wikisource_text(title):
    p = api("pl.wikisource.org", action="parse", page=title, prop="text|revid")
    raw = p.get("parse", {}).get("text", {}).get("*", "")
    raw = re.sub(r"<(style|script|table)[^>]*>.*?</\1>", " ", raw, flags=re.S)  # header box, navigation tables
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"\S+\.djvu/\d+|\[\s*\d+\s*\]", " ", text)  # scan page markers
    return re.sub(r"\s+", " ", text).strip(), p.get("parse", {}).get("revid")


def wiki_excerpts(text, max_chunks=2, size=(900, 2600)):
    """Split a Wikipedia article at section headings; keep substantive prose chunks."""
    skip = re.compile(r"(Przypisy|Bibliografia|Linki zewnętrzne|Zobacz też|Uwagi|Literatura|Galeria)", re.I)
    parts = re.split(r"\n=+ (.+?) =+\n", "\n" + text)
    chunks, head = [], "Wstęp"
    for i, part in enumerate(parts):
        if i % 2 == 1:
            head = part
            continue
        body = re.sub(r"\s+", " ", part).strip()
        if skip.search(head) or len(body) < size[0]:
            continue
        chunks.append((head, body[:size[1]].rsplit(". ", 1)[0] + "."))
    return chunks[:max_chunks]


def doc_excerpts(text, max_chunks=2, size=(700, 1800)):
    """Pick passages from the body of a source text (skip the first part, often front matter)."""
    body = text[min(len(text) // 10, 1500):]
    out = []
    for start in range(0, max(len(body) - size[0], 0), max(len(body) // (max_chunks + 1), size[1])):
        piece = body[start:start + size[1]]
        piece = piece[piece.find(" ") + 1:].rsplit(". ", 1)[0] + "."
        if len(piece) >= size[0]:
            out.append(piece)
        if len(out) >= max_chunks:
            break
    return out


def collect_era(client, era, args):
    r = client.chat.completions.create(
        model="gpt-6-luna", reasoning_effort="low", max_completion_tokens=3000,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROPOSE.format(era=era, n_wiki=args.per_era_wiki, n_docs=args.per_era_docs)}])
    prop = json.loads(r.choices[0].message.content)
    found = []
    for q in prop.get("wiki", []):
        t = resolve("pl.wikipedia.org", q)
        if not t:
            continue
        title, text, revid = wikipedia_text(t)
        for head, chunk in wiki_excerpts(text):
            found.append({"site": "pl.wikipedia.org", "title": title, "section": head, "revid": revid,
                          "url": f"https://pl.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                          "license": "CC BY-SA 4.0", "era": era, "text": chunk})
    for q in prop.get("docs", []):
        t = resolve("pl.wikisource.org", q)
        if not t:
            continue
        text, revid = wikisource_text(t)
        for chunk in doc_excerpts(text):
            found.append({"site": "pl.wikisource.org", "title": t, "section": None, "revid": revid,
                          "url": f"https://pl.wikisource.org/wiki/{quote(t.replace(' ', '_'))}",
                          "license": "public domain", "era": era, "text": chunk})
    print(f"  {era[:60]:60} -> {len(found)} excerpts", flush=True)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-era-wiki", type=int, default=18)
    ap.add_argument("--per-era-docs", type=int, default=6)
    ap.add_argument("--eras", type=int, default=len(ERAS), help="limit for a test run")
    args = ap.parse_args()
    client, _, _ = llm_client()
    out = DATA / "sources"
    out.mkdir(exist_ok=True)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda e: collect_era(client, e, args), ERAS[:args.eras]))
    seen, n = set(), 0
    with open(out / "excerpts.jsonl", "w") as f:
        for era_items in results:
            for it in era_items:
                key = it["text"][:200]
                if key in seen:
                    continue
                seen.add(key)
                n += 1
                f.write(json.dumps({"id": f"src{n:05d}", **it}, ensure_ascii=False) + "\n")
    print(f"{n} excerpts -> {out / 'excerpts.jsonl'}")


if __name__ == "__main__":
    main()
