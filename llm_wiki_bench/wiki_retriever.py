"""Wiki retriever — implements the two paper-defined tools `wiki_search` and
`wiki_read` over a compiled LLM-Wiki directory.

The retrieval design follows the paper (§3.2):

* **wiki_search(query, limit?)** — searches the Wiki index by *prioritizing
  structured signals such as page names, aliases, tags, and descriptions
  before falling back to page content*. It returns candidate pages and
  metadata (no body text) for subsequent reading and traversal.
* **wiki_read(paths)** — batch-reads directory indices (`_index.md`) or full
  pages. For knowledge pages, the returned content includes inter-page
  wikilinks that serve as traversal affordances for subsequent hops.

Scoring (each query sub-token contributes its single highest field hit; all
sub-token contributions are summed):

    name exact         100
    name substring      80
    alias exact         90
    alias substring     70
    tag  exact/substr   50
    description substr  40

A BM25 score over (name + description + body) is added on top, capped so it
cannot outrank a structural exact match.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_logger = logging.getLogger("llm_wiki.retriever")


# ─── Data structures ──────────────────────────────────────────────────────

@dataclass
class WikiPage:
    """A single compiled Wiki page."""
    name: str                                        # file stem (no .md)
    dir_name: str                                    # directory under wiki/
    rel_path: str                                    # e.g. "entities/Einstein.md"
    text: str                                        # full markdown (with frontmatter)
    body: str                                        # markdown without frontmatter
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ""                            # one-line "> ..." blockquote
    links_to: list[str] = field(default_factory=list)  # [[wikilink]] targets


@dataclass
class SearchResult:
    page: WikiPage
    score: float
    matched_fields: list[str] = field(default_factory=list)


# ─── Retriever ────────────────────────────────────────────────────────────

class WikiRetriever:
    """Loads a compiled Wiki and exposes `search` / `read` tools."""

    # BM25 hyper-parameters
    _BM25_K1 = 1.2
    _BM25_B = 0.75
    _BM25_WEIGHT = 5.0   # scale factor on the BM25 score before adding
    _BM25_CAP = 60.0     # cap on the BM25 contribution (< exact-name score 100)

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, wiki_dir: Path):
        self.wiki_dir = Path(wiki_dir)
        self.pages: dict[str, WikiPage] = {}
        self.dir_indexes: dict[str, str] = {}        # dir_name → _index.md content
        self._loaded = False

        # BM25 inverted index
        self._inv: dict[str, dict[str, int]] = {}     # token → {rel_path: tf}
        self._doc_len: dict[str, int] = {}            # rel_path → doc length
        self._avg_dl: float = 0.0
        self._idf: dict[str, float] = {}

    # ── loading ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Walk the wiki directory and build all indices (idempotent)."""
        if self._loaded:
            return
        if not self.wiki_dir.exists():
            _logger.warning("wiki dir does not exist: %s", self.wiki_dir)
            self._loaded = True
            return

        for md in self.wiki_dir.rglob("*.md"):
            rel = md.relative_to(self.wiki_dir)
            rel_str = str(rel)

            if md.name in ("index.md", "overview.md", "log.md"):
                continue

            if md.name == "_index.md":
                dir_name = str(rel.parent) if str(rel.parent) != "." else ""
                if dir_name:
                    try:
                        self.dir_indexes[dir_name] = md.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        pass
                continue

            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            page = self._parse_page(md.stem, rel, text)
            self.pages[rel_str] = page

        self._build_bm25()
        self._loaded = True
        _logger.info(
            "loaded wiki: %d pages, %d directory indices",
            len(self.pages), len(self.dir_indexes),
        )

    def _parse_page(self, stem: str, rel: Path, text: str) -> WikiPage:
        dir_name = str(rel.parent) if str(rel.parent) != "." else ""

        aliases: list[str] = []
        tags: list[str] = []
        description = ""
        body = text

        fm = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if fm:
            body = text[fm.end():]
            block = fm.group(1)
            am = re.search(r"aliases:\s*\[(.+?)\]", block)
            if am:
                aliases = [a.strip().strip("'\"") for a in am.group(1).split(",") if a.strip()]
            tm = re.search(r"tags:\s*\[(.+?)\]", block)
            if tm:
                tags = [t.strip().strip("'\"") for t in tm.group(1).split(",") if t.strip()]

        for line in body.strip().split("\n"):
            line = line.strip()
            if line.startswith("> ") and not line.startswith("> Source"):
                description = line[2:].strip()
                break

        links_to = re.findall(r"\[\[(.+?)\]\]", body)

        return WikiPage(
            name=stem,
            dir_name=dir_name,
            rel_path=str(rel),
            text=text,
            body=body,
            aliases=aliases,
            tags=tags,
            description=description,
            links_to=links_to,
        )

    # ── BM25 ──────────────────────────────────────────────────────────────

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """ASCII tokenizer (lowercase, ≥2 chars). Suitable for English corpora."""
        return [m.group() for m in cls._TOKEN_RE.finditer(text.lower()) if len(m.group()) >= 2]

    def _build_bm25(self) -> None:
        inv: dict[str, dict[str, int]] = {}
        doc_len: dict[str, int] = {}
        total = 0
        for rp, page in self.pages.items():
            doc_text = f"{page.name.replace('-', ' ')} {page.description} {page.body}"
            toks = self._tokenize(doc_text)
            doc_len[rp] = len(toks)
            total += len(toks)
            for tok, tf in Counter(toks).items():
                inv.setdefault(tok, {})[rp] = tf
        self._inv = inv
        self._doc_len = doc_len
        self._avg_dl = total / len(self.pages) if self.pages else 1.0
        N = len(self.pages)
        self._idf = {
            tok: math.log((N - len(post) + 0.5) / (len(post) + 0.5) + 1.0)
            for tok, post in inv.items()
        }

    def _bm25_score(self, query_tokens: Iterable[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        k1, b, avgdl = self._BM25_K1, self._BM25_B, self._avg_dl
        for tok in query_tokens:
            idf = self._idf.get(tok, 0.0)
            if idf <= 0:
                continue
            postings = self._inv.get(tok)
            if not postings:
                continue
            for rp, tf in postings.items():
                dl = self._doc_len.get(rp, 0)
                num = tf * (k1 + 1)
                den = tf + k1 * (1 - b + b * dl / avgdl)
                scores[rp] = scores.get(rp, 0.0) + idf * num / den
        return scores

    # ── search ────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search by structured fields first, then fall back to page content (BM25)."""
        self.load()
        query = (query or "").strip()
        if not query:
            return []

        sub_words = [w for w in query.split() if w]
        bm25 = self._bm25_score(self._tokenize(query))

        results: list[SearchResult] = []
        scored_paths: set[str] = set()

        for rp, page in self.pages.items():
            total = 0.0
            matched: set[str] = set()
            for w in sub_words:
                s, fields = self._score_word(w, page)
                if s > 0:
                    total += s
                    matched.update(fields)

            bm25_s = bm25.get(rp, 0.0)
            if bm25_s > 0:
                total += min(bm25_s * self._BM25_WEIGHT, self._BM25_CAP)
                matched.add("content")

            if total > 0:
                results.append(SearchResult(page=page, score=total, matched_fields=sorted(matched)))
                scored_paths.add(rp)

        # BM25-only hits (no structured match)
        for rp, bm25_s in bm25.items():
            if rp in scored_paths or bm25_s <= 0:
                continue
            page = self.pages.get(rp)
            if page is None:
                continue
            results.append(SearchResult(
                page=page,
                score=min(bm25_s * self._BM25_WEIGHT, self._BM25_CAP),
                matched_fields=["content"],
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    @staticmethod
    def _score_word(word: str, page: WikiPage) -> tuple[float, set[str]]:
        """Score a single sub-token against one page (highest field wins)."""
        w = word.lower()
        best_score = 0.0
        best_field: str | None = None

        # name
        name = page.name.lower()
        name_norm = name.replace("-", " ")
        if w == name or w == name_norm:
            best_score, best_field = 100.0, "name"
        elif w in name or w in name_norm:
            best_score, best_field = 80.0, "name"

        # aliases
        for alias in page.aliases:
            al = alias.lower()
            if w == al and 90.0 > best_score:
                best_score, best_field = 90.0, "aliases"
                break
            if w in al and 70.0 > best_score:
                best_score, best_field = 70.0, "aliases"

        # tags
        for tag in page.tags:
            tl = tag.lower()
            if (w == tl or w in tl) and 50.0 > best_score:
                best_score, best_field = 50.0, "tags"
                break

        # description
        if page.description and w in page.description.lower() and 40.0 > best_score:
            best_score, best_field = 40.0, "description"

        return best_score, ({best_field} if best_field else set())

    # ── read ──────────────────────────────────────────────────────────────

    def read(self, paths: list[str]) -> list[dict]:
        """Batch-read directory indices or page files.

        * `"/"`            → list of available top-level directories
        * `"entities"`     → directory `_index.md` (or page list)
        * `"entities/X.md"` → full page text + metadata
        """
        self.load()
        out: list[dict] = []
        for raw in paths:
            p = (raw or "").strip()
            if p == "/":
                out.append({"path": "/", "type": "root", "dirs": sorted(self.dir_indexes.keys())})
                continue

            if p.endswith(".md"):
                page = self.pages.get(p)
                if page is None:
                    out.append({"path": p, "type": "error", "error": "not found"})
                    continue
                out.append({
                    "path": p,
                    "type": "file",
                    "name": page.name,
                    "text": page.text,
                    "meta": {
                        "aliases": page.aliases,
                        "tags": page.tags,
                        "description": page.description,
                        "links_to": page.links_to,
                    },
                })
                continue

            # directory
            idx = self.dir_indexes.get(p)
            if idx:
                out.append({"path": p, "type": "directory", "name": p, "text": idx})
                continue

            prefix = p + "/"
            pages_in_dir = [pg.name for rel, pg in self.pages.items() if rel.startswith(prefix)]
            if pages_in_dir:
                out.append({"path": p, "type": "directory", "name": p, "pages": pages_in_dir})
            else:
                out.append({"path": p, "type": "error", "error": "not found"})
        return out

    # ── helpers ───────────────────────────────────────────────────────────

    def wiki_map(self) -> str:
        """A short, agent-facing overview of the Wiki layout."""
        self.load()
        counts: dict[str, int] = {}
        for page in self.pages.values():
            counts[page.dir_name] = counts.get(page.dir_name, 0) + 1
        lines = ["# Wiki Map\n"]
        for d in sorted(counts):
            lines.append(f"- **{d or '/'}/** ({counts[d]} pages)")
            idx = self.dir_indexes.get(d, "")
            if idx:
                entries = re.findall(r"\[\[(.+?)\]\]", idx)
                for e in entries[:5]:
                    lines.append(f"  - {e}")
                if len(entries) > 5:
                    lines.append(f"  - ... ({len(entries) - 5} more)")
        return "\n".join(lines)

    # ── tool dispatch (matches paper tool names) ──────────────────────────

    def execute_tool(self, name: str, arguments: dict) -> str:
        """Run a single tool call and return a JSON string."""
        if name == "wiki_search":
            results = self.search(
                arguments.get("query", ""),
                limit=min(int(arguments.get("limit", 10)), 50),
            )
            payload = {
                "matched": len(results),
                "results": [
                    {
                        "path": r.page.rel_path,
                        "name": r.page.name,
                        "score": round(r.score, 3),
                        "matched_fields": r.matched_fields,
                        "meta": {
                            "aliases": r.page.aliases,
                            "tags": r.page.tags,
                            "description": r.page.description,
                        },
                    }
                    for r in results
                ],
            }
            return json.dumps(payload, ensure_ascii=False)

        if name == "wiki_read":
            return json.dumps(self.read(arguments.get("paths") or arguments.get("dirs") or []),
                              ensure_ascii=False)

        return json.dumps({"error": f"unknown tool: {name}"})


# ─── OpenAI tool schemas exposed to the agent ─────────────────────────────

WIKI_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": (
                "Search the Wiki index by prioritizing structured signals such as page "
                "names, aliases, tags, and descriptions before falling back to page "
                "content. Returns candidate pages and metadata only (no body text)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {"type": "integer", "description": "Max results to return (default 10)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_read",
            "description": (
                "Batch-read directory indices (_index.md) or full pages. For knowledge "
                "pages, the returned content includes inter-page wikilinks that serve "
                "as traversal affordances for subsequent hops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of paths. Use '/' for the root, 'entities' for a "
                            "directory index, 'entities/X.md' for a full page."
                        ),
                    }
                },
                "required": ["paths"],
            },
        },
    },
]
