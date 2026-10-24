"""Collect ~100 recent news/notice articles from the BITS Pilani website.

Usage:  python scripts/collect_campus.py --target 100 --max-pages 20
Output: data/raw/campus/campus_articles.json

Polite crawling: one request per second, a descriptive User-Agent, and only
the public /news/ listing and the article pages it links to.
"""
import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "campus" / "campus_articles.json"
BASE = "https://www.bits-pilani.ac.in/news/"
HEADERS = {"User-Agent": "Mozilla/5.0 (IR course project, BITS Pilani Group 6; polite crawler)"}
DELAY = 1.0


def get(url, session):
    time.sleep(DELAY)
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def listing_links(soup):
    links = []
    for a in soup.find_all("a", href=True):
        h = a["href"].split("#")[0]
        if h.startswith(BASE) and "/page/" not in h and h.rstrip("/") != BASE.rstrip("/"):
            links.append(h)
    return list(dict.fromkeys(links))


def canonical_slug(url):
    """Reposted notices get slugs like '...-2-2-2/'; collapse them so we keep one copy."""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"(-\d)+$", "", slug)


def extract_article(soup):
    for t in soup(["script", "style", "nav", "header", "footer", "noscript", "form"]):
        t.decompose()
    title = soup.find("meta", property="og:title")
    title = title["content"] if title else (soup.title.get_text() if soup.title else "")
    title = re.sub(r"\s*-\s*BITS Pilani\s*$", "", title)
    title = re.sub(r"[^\w\s&|:,.()'\-/–]", "", title).strip()

    # The CMS has no semantic <article>; take the block with the most direct paragraph text.
    best, best_len = None, 0
    for div in soup.find_all(["div", "section"]):
        n = sum(len(p.get_text(strip=True)) for p in div.find_all(["p", "li"], recursive=False))
        if n > best_len:
            best, best_len = div, n
    body = best.get_text(" ", strip=True) if best else ""
    body = re.sub(r"\s+", " ", body)

    date = soup.find("meta", property="article:published_time") or \
        soup.find("meta", property="article:modified_time")
    return title, body, (date["content"][:10] if date else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=25)
    ap.add_argument("--min-words", type=int, default=30)
    args = ap.parse_args()

    session = requests.Session()
    urls, seen_slugs = [], set()
    for page in range(1, args.max_pages + 1):
        listing = BASE if page == 1 else f"{BASE}page/{page}/"
        try:
            links = listing_links(get(listing, session))
        except requests.RequestException as e:
            print(f"listing page {page} failed: {e}")
            break
        for u in links:
            slug = canonical_slug(u)
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                urls.append(u)
        print(f"page {page}: {len(urls)} candidate articles")
        if len(urls) >= args.target * 2:
            break

    articles, hashes = [], set()
    collected = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for u in urls:
        if len(articles) >= args.target:
            break
        try:
            title, body, date = extract_article(get(u, session))
        except requests.RequestException as e:
            print(f"skip {u}: {e}")
            continue
        h = hashlib.md5(body.lower().encode()).hexdigest()
        if len(body.split()) < args.min_words or h in hashes:
            continue
        hashes.add(h)
        articles.append({"doc_id": f"cp-{len(articles) + 1:03d}", "url": u, "title": title, "text": body,
                         "published": date, "collected": collected})
        print(f"[{len(articles):3d}] {len(body.split()):5d} words  {title[:70]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(articles, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(articles)} articles -> {OUT}")


if __name__ == "__main__":
    main()
