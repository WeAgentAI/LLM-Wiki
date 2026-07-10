"""Wiki agent — Retrieval-as-Reasoning loop over a compiled LLM-Wiki.

The agent composes `wiki_search` and `wiki_read` calls based on intermediate
observations, iteratively searching, reading, following links, and checking
sufficiency until it gathers enough evidence to answer (paper §3.2).

Paper-faithful hyper-parameters (overridable):

* ``t_max = 15``       — maximum tool-call budget per question
* ``patience = 3``     — stop after this many consecutive empty searches
* ``select_pages = 5`` — at most k pages selected per search

The agent terminates when all reasoning chains have been traced, the tool-call
budget is reached, or consecutive empty searches exceed the patience
threshold. *At least one* ``wiki_read`` call is required before answering.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from wiki_retriever import WIKI_TOOL_SCHEMAS, WikiPage, WikiRetriever

_logger = logging.getLogger("llm_wiki.agent")


# ─── System prompt ────────────────────────────────────────────────────────
#
# Wording is intentionally aligned with the paper (§3.2). The prompt is fully
# domain-agnostic: it does not mention any specific dataset, entity type, or
# corpus. It only describes the two tools and the traversal contract.

_AGENT_SYSTEM_PROMPT_TEMPLATE = """You are a Wiki retrieval agent operating under the Retrieval-as-Reasoning paradigm.
Your job is to answer a question by composing wiki_search and wiki_read calls based on intermediate observations,
iteratively searching, reading, following links, and checking evidence sufficiency until you have gathered enough
evidence to answer.

## Wiki Map
{wiki_map}

## Tools
- **wiki_search(query, limit?)** — Searches the Wiki index by prioritizing structured signals such as page names,
  aliases, tags, and descriptions before falling back to page content. Returns candidate pages and metadata only;
  it does NOT return page body text.
- **wiki_read(paths)** — Batch-reads directory indices (_index.md) or full pages. For knowledge pages, the returned
  content includes inter-page wikilinks (links_to) that serve as traversal affordances for subsequent hops.

## Traversal strategies
Adaptively choose a strategy based on the question:
- **Direct access**: For known entities, read pages directly, or first search and then read the top results.
- **Bridge queries (A → B → answer)**: Read page A, identify entity B through inter-page links, and traverse to
  page B, reducing complex reasoning to iterative link traversal.
- **Exploratory browsing**: For open-ended queries, read directory indices to obtain a structured overview, then
  selectively read promising pages.

## Contract
1. wiki_search returns metadata only — you cannot answer based on search output alone.
2. After every wiki_read, assess evidence sufficiency. If any sub-question is unresolved, continue traversing
   (search a new entity, read a more specific page, or follow a links_to pointer).
3. You MUST call wiki_read at least once before producing a final answer.
4. Stop when all reasoning chains have been traced, the tool-call budget is reached, or consecutive empty searches
   exceed the patience threshold. When you stop, output ONLY the final answer text — no tool calls, no preamble.
"""


# ─── Result container ─────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    pages: list[tuple[str, str]] = field(default_factory=list)   # [(rel_path, name)]
    pages_text: dict[str, str] = field(default_factory=dict)     # rel_path → full md
    pages_meta: dict[str, WikiPage] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)               # human-readable log
    tool_calls: list[dict] = field(default_factory=list)         # raw call log
    total_calls: int = 0
    search_top_results: list[str] = field(default_factory=list)  # top search-hit paths (for fallbacks)


# ─── Agent ────────────────────────────────────────────────────────────────

class WikiAgent:
    """Tool-calling agent that traverses a compiled Wiki to gather evidence."""

    def __init__(
        self,
        retriever: WikiRetriever,
        *,
        call_llm_with_tools: Callable[..., dict | None],
        model: str | None = None,
        t_max: int = 15,
        patience: int = 3,
        select_pages: int = 5,
        verbose: bool = False,
    ):
        self.retriever = retriever
        self.call_llm_with_tools = call_llm_with_tools
        self.model = model
        self.t_max = t_max
        self.patience = patience
        self.select_pages = select_pages
        self.verbose = verbose

    # ── public API ────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> RetrievalResult:
        """Run the agent on a single question and return what it gathered."""
        self.retriever.load()

        system_prompt = _AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            wiki_map=self.retriever.wiki_map(),
        )
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    "Plan the entities or facts you need, then traverse the Wiki to gather them."
                ),
            },
        ]

        result = RetrievalResult()
        consecutive_empty = 0

        # Two extra rounds let the agent emit a final no-tool message after t_max.
        for _ in range(self.t_max + 2):
            if result.total_calls >= self.t_max:
                if self.verbose:
                    print(f"  [agent] reached tool-call budget T_max={self.t_max}")
                break

            assistant_msg = self.call_llm_with_tools(
                messages,
                tools=WIKI_TOOL_SCHEMAS,
                model=self.model,
                temperature=0.0,
            )
            if assistant_msg is None:
                _logger.warning("agent LLM call failed; stopping")
                break

            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []

            if not tool_calls:
                # Agent decided to answer. Enforce the "at least one wiki_read" contract:
                # only nudge if it has search hits it could still read.
                if not result.pages and result.search_top_results and result.total_calls < self.t_max:
                    if self.verbose:
                        print("  [agent] reminder: must call wiki_read before answering")
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have not read any pages yet. wiki_search returns metadata only. "
                            "Call wiki_read on the most relevant search results before answering."
                        ),
                    })
                    continue
                break

            for tc in tool_calls:
                if result.total_calls >= self.t_max:
                    break
                self._execute_one(tc, messages, result)

            consecutive_empty = self._consecutive_empty_searches(result)
            if consecutive_empty >= self.patience:
                if self.verbose:
                    print(f"  [agent] {consecutive_empty} consecutive empty searches; stopping")
                break

        # ── Fallback 1: agent stopped without reading any page, but search had hits.
        # Auto-read the top search results so the answer step is not left empty.
        if not result.pages and result.search_top_results:
            fallback_paths = result.search_top_results[:3]
            if self.verbose:
                print(f"  [agent] fallback: read top search results {fallback_paths}")
            result.trace.append(f"[fallback] auto-read top search results: {', '.join(fallback_paths)}")
            for rp in fallback_paths:
                self._add_page(rp, result)

        # ── Fallback 2: only one page read for a likely multi-hop question.
        # Follow the first page's links_to (preferring pages that also appeared in
        # search hits) to auto-read a second hop.
        elif len(result.pages) == 1 and self._is_likely_multihop(question):
            first_page = next(iter(result.pages_meta.values()), None)
            linked_paths: set[str] = set()
            if first_page is not None:
                for link in first_page.links_to:
                    link_path = link if link.endswith(".md") else link + ".md"
                    page = self.retriever.pages.get(link_path)
                    if page is None or link_path in result.pages_text:
                        continue
                    if not str(getattr(page, "dir_name", "")).startswith("sources"):
                        linked_paths.add(link_path)

            overlap = [rp for rp in result.search_top_results if rp in linked_paths]
            fallback_candidates = overlap if overlap else list(linked_paths)[:3]
            if not fallback_candidates:
                fallback_candidates = [
                    rp for rp in result.search_top_results
                    if rp not in result.pages_text
                    and not str(getattr(self.retriever.pages.get(rp), "dir_name", "")).startswith("sources")
                ][:2]

            if fallback_candidates:
                if self.verbose:
                    print(f"  [agent] fallback (2nd hop): read {fallback_candidates[:2]}")
                result.trace.append(f"[fallback-2hop] auto-read for 2nd hop: {', '.join(fallback_candidates[:2])}")
                for rp in fallback_candidates[:2]:
                    self._add_page(rp, result)

        if self.verbose:
            print(
                f"  [agent] done: {result.total_calls} tool calls, "
                f"{len(result.pages)} pages read"
            )
        return result

    # ── internal helpers ──────────────────────────────────────────────────

    def _execute_one(self, tool_call: dict, messages: list[dict], result: RetrievalResult) -> None:
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}

        # Enforce paper's k = select_pages cap on wiki_search.
        if name == "wiki_search":
            limit = int(args.get("limit", self.select_pages))
            args["limit"] = min(limit, self.select_pages)

        if self.verbose:
            print(f"  [agent] [{result.total_calls + 1}/{self.t_max}] {name}({json.dumps(args, ensure_ascii=False)[:120]})")

        result_str = self.retriever.execute_tool(name, args)
        result.total_calls += 1
        result.tool_calls.append({"step": result.total_calls, "tool": name, "arguments": args, "result": result_str})

        # Post-process for tracking.
        if name == "wiki_search":
            try:
                payload = json.loads(result_str)
                matched = payload.get("matched", 0)
                if matched == 0:
                    result.trace.append(f'search "{args.get("query", "")}" → 0 results')
                else:
                    top = ", ".join(r["name"] for r in payload.get("results", [])[:3])
                    result.trace.append(f'search "{args.get("query", "")}" → {matched} results (top: {top})')
                    # Track top hit paths for the post-loop fallbacks.
                    for r in payload.get("results", [])[:3]:
                        rp = r.get("path") or r.get("dir", "")
                        if rp and rp not in result.search_top_results:
                            result.search_top_results.append(rp)
            except json.JSONDecodeError:
                pass
        elif name == "wiki_read":
            try:
                read_payload = json.loads(result_str)
                names = []
                for item in read_payload:
                    if item.get("type") == "file" and "text" in item:
                        rp = item["path"]
                        if rp not in result.pages_text:
                            page = self.retriever.pages.get(rp)
                            result.pages.append((rp, item.get("name", rp)))
                            result.pages_text[rp] = item["text"]
                            if page is not None:
                                result.pages_meta[rp] = page
                        names.append(item.get("name", rp))
                if names:
                    result.trace.append(f"read: {', '.join(names)}")
            except json.JSONDecodeError:
                pass

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.get("id", f"call_{result.total_calls}"),
            "content": result_str,
        })

    def _consecutive_empty_searches(self, result: RetrievalResult) -> int:
        """Re-derive the consecutive-empty count from the last few search traces."""
        count = 0
        for entry in reversed(result.trace):
            if entry.startswith("search "):
                if entry.endswith("\u2192 0 results"):
                    count += 1
                else:
                    break
        return count

    def _add_page(self, rel_path: str, result: RetrievalResult) -> None:
        """Add a page (by rel_path) into the result if present and not already read."""
        page = self.retriever.pages.get(rel_path)
        if page is None or rel_path in result.pages_text:
            return
        result.pages.append((rel_path, page.name))
        result.pages_text[rel_path] = page.text
        result.pages_meta[rel_path] = page

    def _is_likely_multihop(self, question: str) -> bool:
        """Heuristically decide whether a question likely needs 2+ entity pages.

        Pattern-based (no LLM): comparison keywords, nested bridge patterns, or
        the presence of 2+ distinct proper-noun phrases.
        """
        q = question.lower()

        comparison_keywords = [
            "both", "same country", "same nationality", "same city",
            "which is older", "which is younger", "which is larger",
            "which is smaller", "who is older", "who is younger",
            "do both", "are both", "were both", "did both",
        ]
        if any(kw in q for kw in comparison_keywords):
            return True

        bridge_patterns = [
            r"the \w+ of the \w+ of",
            r"the \w+ of the \w+ that",
            r"the \w+ of the \w+ who",
            r"where .+ was born",
            r"who directed the film",
            r"the \w+ where .+ (was|is|were)",
            r"the \w+ that .+ (directed|wrote|starred|produced|founded)",
        ]
        if any(re.search(pat, q) for pat in bridge_patterns):
            return True

        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        unique_nouns = {pn.lower() for pn in proper_nouns}
        noise_words = {"what", "where", "when", "who", "which", "how",
                       "does", "did", "are", "is", "was", "were"}
        unique_nouns -= noise_words
        return len(unique_nouns) >= 2
