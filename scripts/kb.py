"""Offline knowledge base: Polish Wikipedia + Wikisource passages, BM25 search with tantivy. Runs on the GPU machine.

Two files under --dir (default /workspace/kb):
  passages.sqlite   passage text + title + url (CC BY-SA 4.0 attribution), looked up by id
  tantivy/          BM25 index over stems of title and body (title matches boosted x3)

Build (once; input = HF parquet dumps from download_ft_kb.sh; ~10–15 min on 4 CPUs):
  /workspace/ft-venv/bin/python kb.py build --raw /workspace/kb/raw --dir /scratch/kb && cp -r /scratch/kb/* /workspace/kb/
Search:
  python kb.py search --dir /workspace/kb "bitwa pod Ostrołęką Bem 1831"
From code:
  kb = KB("/workspace/kb"); kb.search("...", k=5) -> [{"id", "title", "url", "text", "score"}, ...]

Polish is heavily inflected and tantivy has no Polish stemmer, so passages and queries are both reduced to
crude stems (first 6 letters of each word: powstanie/powstania/powstaniu -> "powsta").
"""

import argparse
import glob
import os
import re
import sqlite3
import time

import tantivy

STEM_LEN = 6
WORD = re.compile(r"\w+", re.UNICODE)
STOP = set("""a aby ale albo ani aż bez bo by być był była było były czy dla do gdy gdzie go i ich ile im
inne jak jako je jego jej jest jeszcze jeśli już każdy kiedy kto która które którego której który których
ku lub ma mu na nad nie niż o od oraz po pod przez przy się są ta tak także te tego tej ten to tu tych tym
u w we więc właśnie wśród z za ze że żeby np itd tzw r ok""".split())


# Exam-instruction vocabulary: useless (even harmful) as search terms — matches "Podaj.net", "Tekst źródłowy" …
EXAM_WORDS = ("podaj", "wyjaś", "rozst", "uzasa", "odpow", "źródł", "fragm", "tekst", "zadan", "polec", "infor",
              "odwoł", "własn", "wiedz", "przed", "wymie", "okreś", "oceń", "prawd", "stwie", "zazna", "zapis",
              "ilust", "przyp", "porów", "nazwa", "nazwę", "któr", "jeśli", "fałsz", "wskaż", "przyk")


def query_stems(text):
    return [t for t in stems(text) if not t.startswith(EXAM_WORDS)]


def stems(text):
    return [w[:STEM_LEN] for w in WORD.findall(text.lower()) if w not in STOP and (len(w) > 1 or w.isdigit())]


def passages(text, words=150):
    """Split an article into ~`words`-word passages along paragraph boundaries."""
    buf, n = [], 0
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        w = len(para.split())
        if n and n + w > words * 1.4:
            yield "\n".join(buf)
            buf, n = [], 0
        buf.append(para)
        n += w
        if n >= words:
            yield "\n".join(buf)
            buf, n = [], 0
    if buf and (n >= 25 or not text.strip().count("\n")):
        yield "\n".join(buf)


def schema():
    sb = tantivy.SchemaBuilder()
    sb.add_integer_field("pid", stored=True, indexed=False)
    sb.add_text_field("title", tokenizer_name="whitespace", index_option="freq")
    sb.add_text_field("body", tokenizer_name="whitespace", index_option="freq")
    return sb.build()


def build(args):
    import pyarrow.parquet as pq

    os.makedirs(f"{args.dir}/tantivy", exist_ok=True)
    con = sqlite3.connect(f"{args.dir}/passages.sqlite")
    con.executescript("""PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        DROP TABLE IF EXISTS passage;
        CREATE TABLE passage(id INTEGER PRIMARY KEY, source TEXT, title TEXT, url TEXT, text TEXT);""")
    writer = tantivy.Index(schema(), path=f"{args.dir}/tantivy").writer(heap_size=2_000_000_000, num_threads=3)
    files = [("wikipedia", f) for f in sorted(glob.glob(f"{args.raw}/wikipedia/*/*.parquet"))] + \
            [("wikisource", f) for f in sorted(glob.glob(f"{args.raw}/wikisource/*/*.parquet"))]
    pid, t0 = 0, time.time()
    for source, f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["title", "url", "text"]):
            rows = []
            for art in batch.to_pylist():
                if len(art["text"]) < args.min_chars:
                    continue
                title_stems = " ".join(stems(art["title"]))
                for p in passages(art["text"]):
                    pid += 1
                    rows.append((pid, source, art["title"], art["url"], p))
                    writer.add_document(tantivy.Document(pid=pid, title=title_stems, body=" ".join(stems(p))))
            con.executemany("INSERT INTO passage VALUES (?,?,?,?,?)", rows)
        con.commit()
        print(f"{f.split('/')[-1]} ({source}): {pid} passages so far, {time.time() - t0:.0f}s", flush=True)
    writer.commit()
    writer.wait_merging_threads()
    print(f"done: {pid} passages in {time.time() - t0:.0f}s -> {args.dir}")


class KB:
    def __init__(self, path="/workspace/kb"):
        self.index = tantivy.Index.open(f"{path}/tantivy")
        self.searcher = self.index.searcher()
        self.con = sqlite3.connect(f"{path}/passages.sqlite", check_same_thread=False)

    def search(self, query, k=5, title_boost=1.5, per_title=2):
        """Top-k useful passages: BM25 over stems (title x1.5), skipping reference lists / link sections,
        at most `per_title` passages per article."""
        terms = sorted(set(query_stems(query)))
        if not terms:
            return []
        clauses = []
        for t in terms:
            clauses.append((tantivy.Occur.Should, tantivy.Query.term_query(self.index.schema, "body", t)))
            clauses.append((tantivy.Occur.Should, tantivy.Query.boost_query(
                tantivy.Query.term_query(self.index.schema, "title", t), title_boost)))
        hits = self.searcher.search(tantivy.Query.boolean_query(clauses), k * 10).hits
        need = 2 if len(terms) >= 3 else 1  # one rare word alone (e.g. a foreign name) must not decide
        out, per = [], {}
        for score, addr in hits:
            pid = self.searcher.doc(addr)["pid"][0]
            title, url, text = self.con.execute("SELECT title, url, text FROM passage WHERE id=?", (pid,)).fetchone()
            if junk(text) or per.get(title, 0) >= per_title or SKIP_TITLE.match(title):
                continue
            if len(set(terms) & set(stems(title + " " + text))) < need:
                continue
            per[title] = per.get(title, 0) + 1
            out.append({"id": pid, "title": title, "url": url, "text": text, "score": round(score, 2)})
            if len(out) == k:
                break
        return out


SKIP_TITLE = re.compile(r"^(Ulica|Aleja|Aleje|Plac|Rondo|Most|Stacja|Przystanek|Osiedle|Park) |\(ujednoznacznienie\)")
REFS = re.compile(r"^(Przypisy|Bibliografia|Linki zewnętrzne|Zobacz też|Uwagi)\s*$", re.M)


def junk(text):
    """Reference lists, 'see also' and link sections, or passages that are mostly numbers/short lines."""
    if REFS.search(text[:200]) or len(REFS.findall(text)) >= 2:
        return True
    lines = [l for l in text.splitlines() if l.strip()]
    letters = sum(c.isalpha() for c in text)
    return letters < 0.6 * max(1, len(text)) or (len(lines) > 6 and sum(len(l) < 40 for l in lines) > 0.7 * len(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", default="/workspace/kb/raw")
    b.add_argument("--dir", default="/scratch/kb")
    b.add_argument("--min-chars", type=int, default=200)
    s = sub.add_parser("search")
    s.add_argument("--dir", default="/workspace/kb")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("query", nargs="+")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
        return
    kb = KB(args.dir)
    for q in args.query:
        t0 = time.time()
        res = kb.search(q, args.k)
        print(f"######## {q}  ({time.time() - t0:.2f}s)")
        for r in res:
            print(f"[{r['score']}] {r['title']} — {r['url']}\n  {r['text'][:220]}")


if __name__ == "__main__":
    main()
