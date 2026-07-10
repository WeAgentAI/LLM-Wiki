#!/usr/bin/env python3
"""Benchmark ingest adapter — bridges bench_config with the ingest engine.

Responsibilities:
  1. Provide English prompts for Step 1 (page selection) and Step 2 (page synthesis).
  2. Reuse the parsing, writing and validation routines from the original ingest engine.
  3. Disable benchmark-irrelevant features (error book, mirror writes, periodic
     maintenance, etc.).

Usage:
    python bench_ingest.py --dataset hotpotqa --limit 50
    python bench_ingest.py --dataset musique --batch-size 5
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Path setup ───
BENCH_DIR = Path(__file__).parent

# Use bench_config in place of the engine config.
import bench_config as config

# Import the LLM client first (it reads bench_config and has full retry logic).
from llm_client import call_llm, call_llm_json

# Error-book module.
import bench_error_book as error_book

# ─── Feature toggles ───
# Set ABLATION_ERROR_BOOK=0 to disable the error book (constraint injection, LLM repair, logging).
_ENABLE_ERROR_BOOK = os.environ.get("ABLATION_ERROR_BOOK", "1") != "0"

# ─── Logging ───
_logger = logging.getLogger("bench_ingest")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _logger.addHandler(logging.StreamHandler())


# ─── Utility functions ───

def _read_file_safe(path: Path, max_len: int = 5000) -> str:
    if path and path.exists():
        try:
            text = path.read_text(encoding="utf-8")
            return text[:max_len] if len(text) > max_len else text
        except (OSError, UnicodeDecodeError):
            return "(read failed)"
    return "(empty)"


# ─── Index smart-truncation constants ───
MAX_INDEX_TOTAL_CHARS = 100000  # ~25K tokens; leaves room for the rest of the prompt.

# ─── Prompt total-length safety valve ───
MAX_TOTAL_PROMPT_CHARS = 500000  # ~125K tokens; safe for models with a 200K-token context window.


def _apply_prompt_safety_valve(system: str, user: str) -> tuple[str, str]:
    """Total-prompt safety valve: when system+user exceeds MAX_TOTAL_PROMPT_CHARS,
    progressively trim the largest sections of the user prompt so we stay within
    the model context window.

    Trim priority (lowest value first):
      1. Existing Page Names      — drop trailing directories.
      2. Existing Page Content    — drop trailing pages.
      3. Existing Directory Indexes — drop trailing indexes.
    """
    total = len(system) + len(user)
    if total <= MAX_TOTAL_PROMPT_CHARS:
        return system, user

    overflow = total - MAX_TOTAL_PROMPT_CHARS
    print(f"  ⚠️ Prompt safety valve triggered: {total:,} chars > {MAX_TOTAL_PROMPT_CHARS:,} (overflow: {overflow:,})")

    marker_names = "## Existing Page Names"
    marker_content = "## Existing Page Content"
    marker_indexes = "## Existing Directory Indexes"

    idx_names = user.find(marker_names)
    idx_content = user.find(marker_content)
    idx_indexes = user.find(marker_indexes)

    def _find_section_end(text: str, start: int) -> int:
        """Find the end of the section starting at ``start`` (i.e. the line before the next ``## ``)."""
        line_end = text.find("\n", start)
        if line_end == -1:
            return len(text)
        next_section = text.find("\n## ", line_end)
        if next_section == -1:
            return len(text)
        return next_section

    sections_to_trim = []
    if idx_names != -1:
        sections_to_trim.append(("Existing Page Names", idx_names, _find_section_end(user, idx_names)))
    if idx_content != -1:
        sections_to_trim.append(("Existing Page Content", idx_content, _find_section_end(user, idx_content)))
    if idx_indexes != -1:
        sections_to_trim.append(("Existing Directory Indexes", idx_indexes, _find_section_end(user, idx_indexes)))

    for section_name, sec_start, sec_end in sections_to_trim:
        if overflow <= 0:
            break

        section_text = user[sec_start:sec_end]
        section_len = len(section_text)

        min_keep = max(200, int(section_len * 0.2))
        can_trim = section_len - min_keep

        if can_trim <= 0:
            continue

        trim_amount = min(overflow, can_trim)
        new_section = section_text[:section_len - trim_amount]
        last_newline = new_section.rfind("\n")
        if last_newline > 0:
            new_section = new_section[:last_newline]
        new_section += f"\n... ({section_name} truncated by safety valve, {trim_amount:,} chars removed)\n"

        user = user[:sec_start] + new_section + user[sec_end:]
        overflow -= trim_amount
        print(f"    ✂️ Trimmed '{section_name}': -{trim_amount:,} chars")

    final_total = len(system) + len(user)
    print(f"  📏 Final prompt size: {final_total:,} chars (target: ≤{MAX_TOTAL_PROMPT_CHARS:,})")
    return system, user


# ─── Level-2 helper: candidate entity-name extraction regex ───
_CAND_CJK_RE = re.compile(r"[\u4e00-\u9fa5]{2,15}")
_CAND_LATIN_RE = re.compile(r"\b[A-Z][A-Za-z.\-]{1,30}\b")


def _compute_index_relevance(dir_name: str, content: str, title_keywords: set) -> float:
    """Score the relevance between a directory and an article title (normalized density + name match)."""
    score = 0.0
    content_lower = content.lower()
    dir_name_lower = dir_name.lower()

    for kw in title_keywords:
        if kw in dir_name_lower:
            score += 10.0

    content_len = max(len(content), 1)
    for kw in title_keywords:
        count = content_lower.count(kw)
        density = count / (content_len / 10000)
        score += density

    return score


def _get_all_index_content(source_title: str = "") -> dict[str, str]:
    """Read every directory's _index.md; when their total exceeds the budget,
    trim the lower-relevance ones first.

    source_title: title of the current article, used to score directory relevance
    when trimming is required.    """
    indexes = {}
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources/"):
            continue
        idx_path = dir_path / "_index.md"
        if idx_path.exists():
            try:
                indexes[dir_name] = idx_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

    total_chars = sum(len(v) for v in indexes.values())

    if total_chars <= MAX_INDEX_TOTAL_CHARS:
        return {f"wiki/{k}/_index.md": v for k, v in indexes.items()}

    title_keywords = set(w.lower() for w in source_title.split() if len(w) > 2)

    scored = []
    for dir_name, content in indexes.items():
        score = _compute_index_relevance(dir_name, content, title_keywords)
        scored.append((dir_name, content, score, len(content)))

    scored.sort(key=lambda x: x[2], reverse=True)

    result = {}
    remaining_budget = MAX_INDEX_TOTAL_CHARS

    for dir_name, content, score, size in scored:
        if remaining_budget <= 0:
            page_count = content.count("\n- [[")
            result[f"wiki/{dir_name}/_index.md"] = f"({page_count} pages, index too large to display)"
            continue

        if size <= remaining_budget:
            result[f"wiki/{dir_name}/_index.md"] = content
            remaining_budget -= size
        else:
            lines = content.split("\n")
            truncated = []
            chars = 0
            for line in lines:
                if chars + len(line) > remaining_budget:
                    break
                truncated.append(line)
                chars += len(line) + 1
            truncated.append(f"\n... (truncated, {size} chars total)")
            result[f"wiki/{dir_name}/_index.md"] = "\n".join(truncated)
            remaining_budget = 0

    print(f"  📋 Index trimmed: {total_chars:,} → {sum(len(v) for v in result.values()):,} chars (budget {MAX_INDEX_TOTAL_CHARS:,})")
    return result


def _expand_dirs_from_selected(selected_pages: list[str] | None) -> set[str] | None:
    """Derive the directory whitelist that Step 2 must fully expand,
    based on the pages selected in Step 1.

    Returns None when ``selected_pages`` is empty/None (no trimming, expand all).
    """
    if not selected_pages:
        return None
    dirs = {p.strip("[]").split("/", 1)[0] for p in selected_pages if "/" in p.strip("[]")}
    for page_name in selected_pages:
        clean_name = page_name.strip("[]")  # strip the [[...]] double brackets
        if "/" not in clean_name:
            for dir_name, dir_path in config.get_page_dirs().items():
                if dir_name.startswith("sources"):
                    continue
                if (dir_path / f"{clean_name}.md").exists():
                    dirs.add(dir_name)
                    break
    if not dirs:
        return None
    return dirs


def _extract_candidate_names(articles: list[dict]) -> set[str]:
    """Extract candidate proper nouns from the title and body, used for Level-2 alias pruning.

    - 2-15 character runs of CJK characters.
    - Capitalized Latin words (allowing . and -).

    Returns the candidate set, used by ``_get_existing_page_names`` to filter aliases via ``hot_names``.
    """
    if not articles:
        return set()
    blobs = []
    for a in articles:
        blobs.append(str(a.get("title", "")))
        blobs.append(str(a.get("content", ""))[:5000])  # cap at 5000 chars to avoid blow-up
    text = "\n".join(blobs)
    names: set[str] = set()
    names.update(_CAND_CJK_RE.findall(text))
    names.update(_CAND_LATIN_RE.findall(text))
    return names


def _get_existing_page_names(expand_dirs: set[str] | None = None,
                             hot_names: set[str] | None = None) -> str:
    """Return the list of existing page names with three-level trimming support.

    expand_dirs: directories to fully expand (Level 1).
                 Directories outside this whitelist are summarised as
                 "directory_name + page_count" only.
                 Pass ``None`` to disable trimming and expand everything.
    hot_names:   set of candidate entity names extracted from this batch (Level 2).
                 Only pages whose primary name or any alias hits this set will
                 expand their aliases; other pages list only the primary name,
                 which significantly compresses the prompt.
                 Pass ``None`` to keep all aliases.    """
    lines = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        pages = [f.stem for f in dir_path.glob("*.md") if f.name != "_index.md"]
        if not pages:
            continue

        if expand_dirs is not None and dir_name not in expand_dirs:
            lines.append(f"【{dir_name}】({len(pages)} pages, not selected)")
            continue

        lines.append(f"【{dir_name}】")
        for page_name in sorted(pages):
            page_path = dir_path / f"{page_name}.md"
            aliases = ""
            try:
                text = page_path.read_text(encoding="utf-8")
                result = config.split_frontmatter(text)
                if result:
                    _, fm, _ = result
                    am = re.search(r'aliases:\s*\[(.+?)\]', fm)
                    if am:
                        aliases = am.group(1).strip()
            except (OSError, UnicodeDecodeError):
                pass

            if aliases and hot_names is not None:
                alias_list = [a.strip() for a in aliases.split(",")]
                probe = {page_name, *alias_list}
                mentioned = bool(probe & hot_names)
                if mentioned:
                    lines.append(f"{page_name} | {aliases}")
                else:
                    lines.append(page_name)
            elif aliases:
                lines.append(f"{page_name} | {aliases}")
            else:
                lines.append(page_name)
    return "\n".join(lines) if lines else "(no existing pages)"


# ─── Step 1 prompt: page selection + noise filtering ───

def build_select_pages_prompt(source_title: str, source_content: str) -> tuple[str, str]:
    """Step 1: Let LLM read the full source paragraph + directory indexes,
    select which existing Wiki pages need to be viewed/updated."""

    index_contents = _get_all_index_content(source_title=source_title)
    index_text = ""
    for path, content in index_contents.items():
        index_text += f"\n### {path}\n{content}\n"
    if not index_text:
        index_text = "(no indexes yet)"

    purpose_content = _read_file_safe(config.get_purpose_file(), 2000)

    if len(source_content) > config.INGEST_MAX_CONTENT_LEN:
        source_content = source_content[:config.INGEST_MAX_CONTENT_LEN] + "\n...(content truncated)"

    system = "You are a knowledge base maintenance expert. Based on the research direction, source paragraph, and existing directory indexes, determine if the source has knowledge value and which existing Wiki pages need to be viewed/updated."

    user = f"""## Research Direction
{purpose_content}

## Source Paragraph to Process
Title: {source_title}
Full text:
{source_content}

## Existing Directory Indexes (_index.md)
{index_text}

## Task

This Wikipedia paragraph is about to be ingested into the knowledge base. First judge content quality, then select pages to view.

### Step 1: Should this paragraph be skipped?

Skip ONLY if the paragraph is:
- Extremely short (< 2 sentences) with no useful information
- A disambiguation page with no actual content
- A redirect stub

Most Wikipedia paragraphs contain useful knowledge and should NOT be skipped.

### Step 2: If the paragraph has knowledge value, select existing pages to view

1. Which existing pages cover entities/concepts mentioned in this paragraph? (need to update)
2. Which existing pages should be checked to avoid duplication or contradiction?

### Output JSON format

If the paragraph should be skipped:
{{
  "skip": true,
  "skip_reason": "reason for skipping",
  "pages_to_view": [],
  "reasoning": ""
}}

If the paragraph has knowledge value:
{{
  "skip": false,
  "pages_to_view": ["page_name_1", "page_name_2", ...],
  "reasoning": "Brief explanation of why these pages need to be viewed (no more than 100 chars)"
}}

Notes:
- Only select pages **directly and deeply related** to the source paragraph
- **Limit: at most 10 pages**, prioritize the most important ones
- Page names must exactly match those in the indexes
- Match aliases: if index shows "[[Bach]] (J.S. Bach)" and source mentions "J.S. Bach" → select "Bach"
- The reasoning field MUST be concise, no more than 100 characters. Just one sentence summarizing why. Do NOT analyze each page individually.
"""

    return _apply_prompt_safety_valve(system, user)


# ─── Step 1 prompt (batch mode): page selection ───

def build_select_pages_batch_prompt(articles: list[dict]) -> tuple[str, str]:
    """Step 1 batch version: multiple paragraphs at once."""

    combined_title = " ".join(art.get("title", "") for art in articles)
    index_contents = _get_all_index_content(source_title=combined_title)
    index_text = ""
    for path, content in index_contents.items():
        index_text += f"\n### {path}\n{content}\n"
    if not index_text:
        index_text = "(no indexes yet)"

    purpose_content = _read_file_safe(config.get_purpose_file(), 2000)

    articles_text = ""
    for i, art in enumerate(articles):
        content = art["content"]
        if len(content) > config.INGEST_MAX_CONTENT_LEN:
            content = content[:config.INGEST_MAX_CONTENT_LEN] + "\n...(truncated)"
        articles_text += f"\n### Paragraph {i+1}/{len(articles)}\nTitle: {art['title']}\nContent:\n{content}\n"

    system = "You are a knowledge base maintenance expert. Based on the research direction, source paragraphs, and existing directory indexes, determine which existing Wiki pages need to be viewed/updated."

    user = f"""## Research Direction
{purpose_content}

## Source Paragraphs to Process (total: {len(articles)})
{articles_text}

## Existing Directory Indexes (_index.md)
{index_text}

## Task

These Wikipedia paragraphs are about to be ingested into the knowledge base. Select which existing pages need to be viewed/updated.

### Output JSON format

{{
  "skip": false,
  "skip_articles": [],
  "pages_to_view": ["page_name_1", "page_name_2", ...],
  "reasoning": "Brief explanation (no more than 100 chars)"
}}

Notes:
- Only select pages **directly related** to the source paragraphs
- **Limit: at most 15 pages** total for all paragraphs
- Page names must exactly match those in the indexes
- The reasoning field MUST be concise, no more than 100 characters. Just one sentence summarizing why. Do NOT analyze each page individually.
"""

    return _apply_prompt_safety_valve(system, user)


# ─── Step 2 prompt (single): generate wiki pages ───

def build_ingest_prompt(source_title: str, source_time: str, source_content: str,
                        existing_pages_content: str = "", source_type: str = "wikipedia",
                        selected_pages: list[str] | None = None) -> tuple[str, str]:
    """Build English ingestion prompt for a single source paragraph."""

    today = datetime.now().strftime("%Y-%m-%d")
    dir_catalog = config.get_dir_catalog_text()
    purpose_content = _read_file_safe(config.get_purpose_file(), 3000)
    schema_content = _read_file_safe(config.SCHEMA_FILE, 5000)
    existing_page_names = _get_existing_page_names()

    page_types = config.get_page_types()
    page_types_desc = "\n".join(f"  - {name}: {info['description']}" for name, info in page_types.items())

    dir_info = config.get_all_dir_info()
    dirs_desc = "\n".join(f"  - wiki/{name}/: {info['description']}" for name, info in dir_info.items())

    _example_dirs = [n for n in page_types if n != "entities"]
    example_dir = _example_dirs[0] if _example_dirs else "concepts"

    if len(source_content) > config.INGEST_MAX_CONTENT_LEN:
        source_content = source_content[:config.INGEST_MAX_CONTENT_LEN] + "\n...(content truncated)"

    system = f"""You are a knowledge base maintenance expert. This knowledge base is designed for AI retrieval, not for human reading.

Core principles:
- High information density, prominent keywords, uniform format, easy for semantic matching
- Use structured fact lists, not prose paragraphs
- One key point per item, avoid long paragraphs
- Use full names for people, works — no pronouns ("he", "it", "they")
- Fill frontmatter `aliases` field with common alternative names, abbreviations, different spellings
- Use informative titles (e.g., "Biography and Career" not "A Life Story")

Your task: Analyze the source paragraph and directly generate/update Wiki page files."""

    # Inject error-book constraints (skipped when the error book is disabled).
    error_constraints = ""
    if _ENABLE_ERROR_BOOK:
        try:
            error_constraints = error_book.get_active_constraints()
        except Exception:
            pass

    user = f"""## Schema
{schema_content}

## Knowledge Base Directory Overview
{dir_catalog}

## Research Direction
{purpose_content}

## Current Page Types
{page_types_desc}

## Available Wiki Directories
{dirs_desc}

Note: Only write pages into directories listed above. If you need a new directory, propose it via ---DIR_CHANGES---.

## Existing Page Names (avoid creating duplicates, prefer updating existing pages)
{existing_page_names}

## Existing Page Content (when updating, merge new information with this content)
{existing_pages_content}

## Source Paragraph to Process
Title: {source_title}
Content:
{source_content}

{error_constraints}

## Task

1. **Analyze** the source paragraph: identify key facts, entities/concepts, which pages to create/update
2. **Generate** Wiki page files directly

Output format: `---FILE: path---` followed by complete file content.

**Deduplication rules**:
- If "Existing Page Names" contains the same entity under a different name/alias, update the existing page instead of creating a new one
- "Existing Page Names" are grouped by directory: `【directory】` with one page per line, format `main_name | alias1, alias2`
- When referencing existing pages, use `[[directory/main_name]]`
- Directories marked "(N pages, not selected)" are unlikely relevant — don't create new pages for them

**Files to output**:
1. **Source digest page** (wiki/sources/digests/): Structured summary of the source paragraph
   - Filename format: `title-keywords.md` (no `source-` prefix, the directory already indicates it's a source)
   - ⚠️ **All 4 sections must be output**:
   - **## Summary**: ≤200 word structured summary (required)
   - **## Key Facts**: Factual information from the source (required, ≥2 items)
   - **## Key Entities**: Entity names mentioned (required, ≥1 item)
   - **## Related Context**: Background info useful for multi-hop QA (required)

2. **Knowledge pages** (place in the most appropriate directory):
   - Each page's `type` field must match its directory name
   - ⚠️ **Must have one-sentence summary**: After `# Page Title`, add `> One-sentence summary` (blockquote)
   - New pages: create directly
   - Existing pages (content shown above): output the **complete updated content** merging old and new
   - ⚠️ **Only modify pages whose content was shown above**. For other pages in "Existing Page Names", only reference them via `[[...]]`
   - **Related Pages** links must include directory path: `[[directory/page_name]]`
   - ⚠️ Each Related Pages item MUST have a `[[...]]` link — writing only description without link is a format error. Only link to existing pages or pages newly created in this batch; if a concept is worth linking but has no corresponding page, **create that page**
   - ⚠️ **Every knowledge page MUST have at least 2 Related Pages links** — completely isolated pages are not allowed. If you cannot find relevant pages to link, create related pages for the key entities/concepts mentioned in the source
   - `aliases` field: common alternative names, abbreviations, different spellings
   - ⚠️ **Related Sources**: only `[[sources/digests/title-keywords]]` format links

Notes:
- ⚠️ **Do NOT output any _index.md files** (auto-maintained by code)
- **Do NOT output wiki/index.md** (auto-generated)
- Use English page names
- **Control output quantity**: max 3-5 knowledge pages per source paragraph
- Prefer merging related content into one page over creating many small pages

## Optional: Directory Change Proposals (conservative, usually not needed)

If directory structure has **obvious issues**, append after all files:

---DIR_CHANGES---
[{{"action": "split/merge/move_page", "from": "old_dir", "to": "new_dir", "description": "Description", "move_pages": ["page1"], "reason": "reason"}}]

## Output Example

---FILE: wiki/sources/digests/einstein-physics.md---
---
type: source
source_title: Albert Einstein
tags: [physics, relativity]
---
# Albert Einstein
> Source: Wikipedia paragraph
## Summary
Albert Einstein was a German-born theoretical physicist...
## Key Facts
- Born March 14, 1879 in Ulm, Germany
- Developed the theory of special relativity in 1905
## Key Entities
- Albert Einstein
- Theory of Relativity
## Related Context
- Einstein's work on the photoelectric effect earned him the Nobel Prize in 1921

---FILE: wiki/entities/Einstein.md---
---
type: entities
aliases: [Albert Einstein, A. Einstein]
tags: [physicist, Nobel laureate]
---
# Einstein
> German-born theoretical physicist, developer of the theory of relativity
## Key Facts
- Albert Einstein (1879-1955) was a German-born theoretical physicist
- Developed the theory of special relativity (1905) and general relativity (1915)
## Related Pages
- [[concepts/Relativity]] — Einstein's foundational theory
## Related Sources
- [[sources/digests/einstein-physics]] — Wikipedia paragraph about Einstein

**Important**: The above is just an example format. Place knowledge pages in the most appropriate directory based on content.
"""

    return _apply_prompt_safety_valve(system, user)


# ─── Step 2 prompt (batch): generate wiki pages ───

def build_ingest_prompt_batch(articles: list[dict], existing_pages_content: str = "",
                              selected_pages: list[str] | None = None) -> tuple[str, str]:
    """Build English ingestion prompt for multiple source paragraphs."""

    today = datetime.now().strftime("%Y-%m-%d")
    dir_catalog = config.get_dir_catalog_text()
    purpose_content = _read_file_safe(config.get_purpose_file(), 3000)
    schema_content = _read_file_safe(config.SCHEMA_FILE, 5000)
    existing_page_names = _get_existing_page_names()

    page_types = config.get_page_types()
    page_types_desc = "\n".join(f"  - {name}: {info['description']}" for name, info in page_types.items())

    dir_info = config.get_all_dir_info()
    dirs_desc = "\n".join(f"  - wiki/{name}/: {info['description']}" for name, info in dir_info.items())

    _example_dirs = [n for n in page_types if n != "entities"]
    example_dir = _example_dirs[0] if _example_dirs else "concepts"

    articles_text = ""
    for i, art in enumerate(articles):
        content = art["content"]
        if len(content) > config.INGEST_MAX_CONTENT_LEN:
            content = content[:config.INGEST_MAX_CONTENT_LEN] + "\n...(truncated)"
        title = art["title"]
        stem = _predict_article_stem(title)
        articles_text += f"""
### Paragraph {i+1}/{len(articles)}
Title: {title}
⚠️ Source archive stem (system-assigned): `{stem}`
   → When generating this paragraph's digest: **frontmatter must include `source_article: {stem}`**
Content:
{content}
"""

    system = f"""You are a knowledge base maintenance expert. This knowledge base is designed for AI retrieval, not for human reading.

Core principles:
- High information density, prominent keywords, uniform format, easy for semantic matching
- Use structured fact lists, not prose paragraphs
- One key point per item, avoid long paragraphs
- Use full names for people, works — no pronouns ("he", "it", "they")
- Fill frontmatter `aliases` field with common alternative names, abbreviations, different spellings
- Use informative titles (e.g., "Biography and Career" not "A Life Story")

Your task: Analyze multiple source paragraphs and generate/update all Wiki page files at once. Note that multiple paragraphs may share entities — merge updates into single pages."""

    user = f"""## Schema
{schema_content}

## Knowledge Base Directory Overview
{dir_catalog}

## Research Direction
{purpose_content}

## Current Page Types
{page_types_desc}

## Available Wiki Directories
{dirs_desc}

Note: Only write pages into directories listed above.

## Existing Page Names (avoid creating duplicates)
{existing_page_names}

## Existing Page Content (merge new information with this)
{existing_pages_content}

## Source Paragraphs to Process (total: {len(articles)})
{articles_text}

## Task

1. **Analyze** all source paragraphs: identify key facts, entities/concepts, pages to create/update
2. **Cross-reference**: If multiple paragraphs mention the same entity, merge into one page
3. **Generate** all Wiki page files

Output format: `---FILE: path---` followed by complete file content.

**Deduplication rules**:
- If "Existing Page Names" contains the same entity under a different name/alias, update existing page
- Multiple paragraphs mentioning the same entity → merge into one page
- If multiple paragraphs relate to the same knowledge page, list all digest links in "## Related Sources"

**Files to output**:
1. **Source digest pages** (wiki/sources/digests/): One per source paragraph
   - ⚠️ **Filename must match the stem given above** (i.e., `{{stem}}.md`)
   - ⚠️ **frontmatter must include `source_article: {{stem}}`**
   - ⚠️ **All 4 sections must be output**:
   - **## Summary**: ≤200 word summary (required)
   - **## Key Facts**: Factual information (required, ≥2 items)
   - **## Key Entities**: Entity names (required, ≥1)
   - **## Related Context**: Background for multi-hop QA (required)

2. **Knowledge pages** (place in most appropriate directory):
   - `type` must match directory name
   - ⚠️ Must have `> One-sentence summary` after title
   - New pages: create directly
   - Existing pages: output complete updated content
   - ⚠️ Only modify pages whose content was shown above
   - Links: `[[directory/page_name]]` format
   - ⚠️ **Related Sources**: only link digests from this batch

Notes:
- ⚠️ **Do NOT output any _index.md files**
- **Do NOT output wiki/index.md**
- Use English page names
- **Max 3-5 knowledge pages per source paragraph**
- Prefer merging over creating many small pages

## Optional: Directory Change Proposals

---DIR_CHANGES---
[{{"action": "split/merge/move_page", "from": "old_dir", "to": "new_dir", "description": "Description", "move_pages": ["page1"], "reason": "reason"}}]

Only output if truly needed. Most of the time, no changes are necessary.
"""

    return _apply_prompt_safety_valve(system, user)


def _predict_article_stem(title: str) -> str:
    """Predict the article stem for a source paragraph."""
    safe_title = re.sub(r'[<>:"/\\|?*\[\]]', '', title)[:60].strip()
    safe_title = safe_title.rstrip('.')
    stem = safe_title.replace(' ', '-').lower()
    stem = re.sub(r'-{2,}', '-', stem).strip('-')
    return stem


# ─── Validation and post-processing helpers ───

def _normalize_filename(filename: str) -> str:
    """Normalize filenames: unify whitespace and full/half-width characters."""
    result = []
    for ch in filename:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(ch)
    name = "".join(result)
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'\s*-\s*', '-', name)
    return name


def _sanitize_frontmatter(content: str) -> str:
    """Clean up frontmatter:
    - Convert full-width commas in tags/aliases to ASCII commas.
    - Remove the ``confidence`` field.
    - Convert multi-line list values into single-line array form.
    - Reorder fields as: type -> created -> updated -> aliases -> tags -> rest.
    """
    if not content.startswith("---"):
        return content
    result = config.split_frontmatter(content)
    if result is None:
        return content
    before, fm_text, body = result

    fm_lines = fm_text.strip().split("\n")

    merged_lines = []
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        stripped = line.strip()
        m = re.match(r'^(\w+):\s*$', stripped)
        if m and i + 1 < len(fm_lines) and fm_lines[i + 1].strip().startswith("- "):
            key = m.group(1)
            items = []
            i += 1
            while i < len(fm_lines) and fm_lines[i].strip().startswith("- "):
                item = fm_lines[i].strip()[2:].strip().strip("'\"")
                items.append(item)
                i += 1
            merged_lines.append(f"{key}: [{', '.join(items)}]")
            continue
        merged_lines.append(line)
        i += 1

    ordered_keys = ["type", "created", "updated", "aliases", "tags"]
    ordered = {}
    rest_lines = []
    for line in merged_lines:
        stripped = line.strip()
        if stripped.startswith("confidence:"):
            continue
        if stripped.startswith(("tags:", "aliases:", "sources:")):
            line = line.replace("，", ", ").replace("、", ", ")
            line = re.sub(r',\s*', ', ', line)
        m = re.match(r'^(\w+):', stripped)
        if m and m.group(1) in ordered_keys:
            ordered[m.group(1)] = stripped
        else:
            rest_lines.append(stripped)

    final_lines = []
    for key in ordered_keys:
        if key in ordered:
            final_lines.append(ordered[key])
    final_lines.extend(rest_lines)

    return f"---\n{chr(10).join(final_lines)}\n---{body}"


def _extract_type_from_content(content: str) -> str:
    """Extract the ``type`` field from frontmatter."""
    if not content.startswith("---"):
        return ""
    result = config.split_frontmatter(content)
    if result is None:
        return ""
    _, fm_text, _ = result
    m = re.search(r'^type:\s*(.+)$', fm_text, re.MULTILINE)
    if m:
        return m.group(1).strip().strip('"').strip("'")
    return ""


def _check_frontmatter_complete(content: str, rel_path: str) -> bool:
    """Check whether the frontmatter block is complete (not truncated)."""
    if not content.startswith("---"):
        return True
    result = config.split_frontmatter(content)
    if result is None:
        print(f"  ⚠️ frontmatter truncated: {rel_path} (missing closing ---)")
        return False
    _, _, body = result
    body = body.strip()
    if len(body) < 20 and not rel_path.endswith("_index.md"):
        print(f"  ⚠️ content possibly truncated: {rel_path} (body only {len(body)} chars)")
        return False
    return True


_DIGEST_REQUIRED_SECTIONS = ["Summary", "Key Facts", "Key Entities", "Related Context"]


def _check_digest_completeness(content: str, rel_path: str) -> str:
    """Validate digest pages; auto-insert placeholders for missing required sections."""
    result = config.split_frontmatter(content)
    if result is None:
        return content
    before, fm_text, body = result

    missing = []
    for section in _DIGEST_REQUIRED_SECTIONS:
        if f"## {section}" not in body:
            missing.append(section)

    if not missing:
        return content

    print(f"  ⚠️ digest missing required sections [{', '.join(missing)}]: {rel_path}, auto-filling placeholders")

    section_defaults = {
        "Summary": "(to be filled)",
        "Key Facts": "- (to be filled)",
        "Key Entities": "- (to be filled)",
        "Related Context": "- (to be filled)",
    }

    lines = body.split("\n")
    segments = []
    current_section = ""
    current_lines = []

    for line in lines:
        if line.strip().startswith("## ") and line.strip() != "## ":
            if current_lines:
                segments.append((current_section, current_lines))
            current_section = line.strip()[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        segments.append((current_section, current_lines))

    header_lines = []
    first_section_idx = 0
    for i, (name, _) in enumerate(segments):
        if name:
            first_section_idx = i
            break
        header_lines.extend(segments[i][1])
    else:
        first_section_idx = len(segments)
        for name, ls in segments:
            header_lines.extend(ls)

    existing_sections = {}
    for i in range(first_section_idx, len(segments)):
        name, ls = segments[i]
        existing_sections[name] = ls

    new_body_lines = list(header_lines)
    non_required = [name for name in existing_sections if name not in _DIGEST_REQUIRED_SECTIONS]

    for section in _DIGEST_REQUIRED_SECTIONS:
        if section in existing_sections:
            new_body_lines.extend(existing_sections[section])
        else:
            new_body_lines.append(f"\n## {section}")
            new_body_lines.append(section_defaults[section])

    for name in non_required:
        new_body_lines.extend(existing_sections[name])

    new_body = "\n".join(new_body_lines)
    return f"---\n{fm_text}\n---{new_body}"


def _inject_dates(content: str, file_exists: bool, existing_path=None) -> str:
    """Inject created/updated dates into the frontmatter."""
    if not content.startswith("---"):
        return content
    result = config.split_frontmatter(content)
    if result is None:
        return content
    before, fm_text, body = result

    today = datetime.now().strftime("%Y-%m-%d")

    is_source = False
    for line in fm_text.strip().split("\n"):
        if line.strip().startswith("type:") and "source" in line.split(":", 1)[1].strip():
            is_source = True
            break

    fm_lines = [line for line in fm_text.strip().split("\n")
                if not line.strip().startswith("created:") and not line.strip().startswith("updated:")]

    created = today
    if file_exists and existing_path:
        try:
            old_text = existing_path.read_text(encoding="utf-8")
            old_result = config.split_frontmatter(old_text)
            if old_result:
                _, old_fm, _ = old_result
                for line in old_fm.strip().split("\n"):
                    if line.strip().startswith("created:"):
                        created = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    new_lines = []
    inserted = False
    for line in fm_lines:
        new_lines.append(line)
        if line.strip().startswith("type:") and not inserted:
            new_lines.append(f"created: {created}")
            if not is_source:
                new_lines.append(f"updated: {today}")
            inserted = True

    if not inserted:
        new_lines.insert(0, f"created: {created}")
        if not is_source:
            new_lines.insert(1, f"updated: {today}")

    return f"---\n{chr(10).join(new_lines)}\n---{body}"


def _lcs_len(a: str, b: str) -> int:
    """Return the longest-common-subsequence length between two strings."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _fuzzy_match_article(title_part: str, candidates: list[str]) -> str | None:
    """Fuzzy-match the best article stem from a candidate list."""
    if len(candidates) == 1:
        return candidates[0]

    best_score = 0
    best_match = None
    for cand in candidates:
        cand_title = cand[11:] if len(cand) > 11 else cand
        lcs = _lcs_len(title_part.lower(), cand_title.lower())
        score = lcs / max(len(title_part), len(cand_title), 1)
        if score > best_score:
            best_score = score
            best_match = cand

    return best_match if best_score >= 0.2 else None


def _inject_article_link_to_digests(article_stems: list[str]):
    """Inject the original-article link into freshly ingested digests (source_article field + ## Original section)."""
    if not article_stems:
        return

    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return
    digests_dir = wiki_dir / "sources" / "digests"
    if not digests_dir.exists():
        return

    injected_fm = 0
    injected_link = 0
    for md in digests_dir.glob("*.md"):
        if md.name == "_index.md":
            continue
        text = md.read_text(encoding="utf-8")

        if "source_article:" in text[:500]:
            continue

        new_text = text
        digest_stem = md.stem
        best_art = None
        best_score = 0
        for art_stem in article_stems:
            lcs = _lcs_len(digest_stem.lower(), art_stem.lower())
            score = lcs / max(len(digest_stem), len(art_stem), 1)
            if score > best_score:
                best_score = score
                best_art = art_stem

        if best_art and best_score >= 0.3:
            if new_text.startswith("---"):
                fm_result = config.split_frontmatter(new_text)
                if fm_result:
                    _, fm_text, body = fm_result
                    fm_text = fm_text.rstrip() + f"\nsource_article: {best_art}"
                    new_text = f"---\n{fm_text}\n---{body}"
                    injected_fm += 1

            if "## Original" not in new_text:
                new_text = new_text.rstrip() + f"\n\n## Original\n- [[sources/articles/{best_art}]]\n"
                injected_link += 1

        if new_text != text:
            md.write_text(new_text, encoding="utf-8")

    if injected_fm or injected_link:
        parts = []
        if injected_fm:
            parts.append(f"injected {injected_fm} source_article fields")
        if injected_link:
            parts.append(f"added {injected_link} original links")
        print(f"  🔗 Article link injection: {', '.join(parts)}")


def _fix_digest_article_links():
    """Fix [[sources/articles/...]] links inside digest files."""
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return 0
    digests_dir = wiki_dir / "sources" / "digests"
    articles_dir = wiki_dir / "sources" / "articles"
    if not digests_dir.exists() or not articles_dir.exists():
        return 0

    all_article_stems = [md.stem for md in articles_dir.glob("*.md") if md.name != "_index.md"]

    fixed_count = 0
    for digest_md in digests_dir.glob("*.md"):
        if digest_md.name == "_index.md":
            continue
        text = digest_md.read_text(encoding="utf-8")
        new_text = text

        for match in re.finditer(r'\[\[sources/articles/([^\]]+)\]\]', text):
            link_name = match.group(1)
            real_path = articles_dir / f"{link_name}.md"
            if real_path.exists():
                continue
            # Fuzzy match.
            best = _fuzzy_match_article(link_name, all_article_stems)
            if best and best != link_name:
                old_link = f"[[sources/articles/{link_name}]]"
                new_link = f"[[sources/articles/{best}]]"
                new_text = new_text.replace(old_link, new_link)

        if '## Original' not in new_text:
            art_link = None
            if new_text.startswith("---"):
                fm_result = config.split_frontmatter(new_text)
                if fm_result:
                    _, fm_text, _ = fm_result
                    for line in fm_text.strip().split("\n"):
                        if line.strip().startswith("source_article:"):
                            art_link = line.split(":", 1)[1].strip().strip('"')
                            break
            if art_link:
                new_text = new_text.rstrip() + f"\n\n## Original\n- [[sources/articles/{art_link}]]\n"

        if new_text != text:
            digest_md.write_text(new_text, encoding="utf-8")
            fixed_count += 1

    if fixed_count:
        print(f"  🔗 Fixed/added {fixed_count} digest article links")
    return fixed_count


def _rebuild_sources_index():
    """Rebuild the sources/ indexes programmatically."""
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return
    sources_dir = wiki_dir / "sources"
    digests_dir = sources_dir / "digests"
    articles_dir = sources_dir / "articles"

    if not sources_dir.exists():
        return

    if digests_dir.exists():
        entries = []
        for md in sorted(digests_dir.glob("*.md")):
            if md.name == "_index.md":
                continue
            text = md.read_text(encoding="utf-8")
            title = md.stem
            tags_str = ""

            for line in text.split("\n"):
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    title = stripped[2:].strip()
                    break

            if text.startswith("---"):
                fm_result = config.split_frontmatter(text)
                if fm_result:
                    _, fm_text, _ = fm_result
                    for line in fm_text.strip().split("\n"):
                        if line.strip().startswith("tags:"):
                            tag_part = line.split(":", 1)[1].strip().strip("[]")
                            tag_list = [t.strip() for t in tag_part.replace("，", ",").split(",")][:3]
                            tags_str = " ".join(f"#{t}" for t in tag_list if t)

            entry = f"- [[{md.stem}]] — {title} {tags_str}"
            entries.append(entry)

        digest_lines = ["# Digest Index\n"]
        for entry in entries:
            digest_lines.append(entry)

        digests_idx = digests_dir / "_index.md"
        digests_idx.write_text("\n".join(digest_lines) + "\n", encoding="utf-8")

    if articles_dir.exists():
        art_entries = []
        for md in sorted(articles_dir.glob("*.md")):
            if md.name == "_index.md":
                continue
            text = md.read_text(encoding="utf-8")
            title = md.stem

            if text.startswith("---"):
                fm_result = config.split_frontmatter(text)
                if fm_result:
                    _, fm_text, _ = fm_result
                    for line in fm_text.strip().split("\n"):
                        if line.strip().startswith("source_title:"):
                            val = line.split(":", 1)[1].strip().strip('"')
                            if val:
                                title = val
                            break

            entry = f"- [[{md.stem}]] — {title}"
            art_entries.append(entry)

        art_lines = ["# Article Archive Index\n"]
        for entry in art_entries:
            art_lines.append(entry)

        articles_idx = articles_dir / "_index.md"
        articles_idx.write_text("\n".join(art_lines) + "\n", encoding="utf-8")

    digest_count = len(list(digests_dir.glob("*.md"))) - 1 if digests_dir.exists() else 0
    article_count = len(list(articles_dir.glob("*.md"))) - 1 if articles_dir.exists() and (articles_dir / "_index.md").exists() else len(list(articles_dir.glob("*.md"))) if articles_dir.exists() else 0

    sources_idx_content = f"""# Sources Index

## Directory Structure
- **digests/** ({digest_count} pages) — Structured summaries of source paragraphs
- **articles/** ({article_count} pages) — Original paragraph archives

See [digests/_index.md](digests/_index.md) for digest index.
See [articles/_index.md](articles/_index.md) for article archive index.
"""
    sources_idx = sources_dir / "_index.md"
    sources_idx.write_text(sources_idx_content, encoding="utf-8")


def _rebuild_global_index(update_overview: bool = False):
    """Regenerate the top-level index.md (knowledge overview + directory overview).

    Layout:
      1. Top: ``# title`` plus a knowledge-overview paragraph (LLM-generated cross-source summary).
      2. Bottom: directory overview (program-generated; one row per directory with description + page count).

    The LLM is invoked only when ``update_overview=True``; otherwise the existing overview text is kept verbatim.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return

    dir_catalog = config.get_dir_catalog_text()
    today_str = datetime.now().strftime("%Y-%m-%d")
    index_path = wiki_dir / "index.md"

    existing = ""
    if index_path.exists():
        existing = index_path.read_text(encoding="utf-8")

    if update_overview:
        page_count = _count_knowledge_pages()
        if page_count >= 3:
            overview = _generate_overview_text(existing, dir_catalog)
        else:
            overview = _extract_existing_overview(existing)
    else:
        overview = _extract_existing_overview(existing)

    parts = [f"# Wiki Directory Overview\n"]
    if overview:
        overview_quoted = overview.replace("\n", "\n> ")
        parts.append(f"\n> **Knowledge Overview** (updated {today_str})\n>\n> {overview_quoted}\n")
    parts.append(f"\n## Directory Catalog\n\n{dir_catalog}\n")

    index_path.write_text("\n".join(parts), encoding="utf-8")


def _extract_existing_overview(existing_index: str) -> str:
    """Extract the previous knowledge-overview text from index.md (no LLM call)."""
    if "> **Knowledge Overview**" not in existing_index:
        return ""
    m = re.search(r'> \*\*Knowledge Overview\*\*[^\n]*\n((?:>.*\n)*)', existing_index)
    if m:
        return m.group(1).replace("> ", "").replace(">", "").strip()
    return ""


def _generate_overview_text(existing_index: str, dir_catalog: str) -> str:
    """Ask the LLM to write a cross-source knowledge overview.

    The result is placed at the top of index.md and serves as the "zero-hop"
    context during retrieval.
    """
    purpose = _read_file_safe(config.get_purpose_file(), 2000)

    index_contents = _get_all_index_content()
    indexes_text = ""
    for path, content in index_contents.items():
        indexes_text += f"\n### {path}\n{content}\n"

    old_overview = _extract_existing_overview(existing_index)

    prompt = f"""You are a knowledge base maintenance expert. Generate a comprehensive overview for this knowledge base.

## Research Direction
{purpose}

## Current Directory Overview
{dir_catalog}

## Directory Indexes
{indexes_text}

## Previous Overview (if any)
{old_overview if old_overview else "(none)"}

## Task
Generate a 200-400 word comprehensive overview as a SINGLE PARAGRAPH (no line breaks, no bullet points, no numbered lists, no separate sections). The paragraph should:
1. Summarize the core knowledge domains and coverage
2. Naturally weave in the most important entity/concept keywords (for semantic matching)
3. Point out the main threads and topic connections
4. If there is a previous overview, update it rather than rewriting from scratch

CRITICAL: Output ONLY ONE continuous paragraph with NO line breaks. Do not split into multiple paragraphs. Do not use any markdown formatting or titles."""

    try:
        result = call_llm("You are a knowledge base maintenance expert.", prompt,
                          max_tokens=1024,
                          model=config.LLM_FAST_MODEL,
                          temperature=0.3)
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r"^```\w*\n?", "", result)
            result = re.sub(r"\n?```$", "", result)
        return result.strip()
    except Exception as e:
        print(f"  ⚠️ Overview generation failed: {e}")
        return old_overview



_PERIODIC_EVERY = 30
_CONTENT_FIX_EVERY = 60
_CONSOLIDATE_EVERY = 180


def _count_knowledge_pages() -> int:
    """Count current knowledge pages (excluding ``sources``)."""
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return 0
    count = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if dir_path.exists():
            count += len([f for f in dir_path.glob("*.md") if f.name != "_index.md"])
    return count


def quick_lint_bench() -> dict:
    """Lightweight lint pass (pure code checks; no WikiGraph dependency).

    Checks performed:
      1. Broken wikilinks (links pointing at non-existent pages).
      2. Index consistency (_index.md references that no longer exist).
      3. ``type`` field vs directory consistency.
      4. Digest section completeness.
      5. Duplicate page detection.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir or not wiki_dir.exists():
        return {}

    issues = {}

    all_pages: dict[str, Path] = {}  # page_name → path
    all_pages_with_dir: dict[str, Path] = {}  # "dir/page_name" → path
    for dir_name, dir_path in config.get_page_dirs().items():
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            all_pages[md.stem] = md
            all_pages_with_dir[f"{dir_name}/{md.stem}"] = md

    broken_links = []
    for page_name, page_path in all_pages.items():
        try:
            text = page_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in re.finditer(r'\[\[([^\]]+)\]\]', text):
            link = m.group(1)
            if "|" in link:
                link = link.split("|")[0].strip()
            if link in all_pages or link in all_pages_with_dir:
                continue
            link_name = link.rsplit("/", 1)[-1] if "/" in link else link
            if link_name not in all_pages:
                broken_links.append({"from": page_name, "to": link})
    if broken_links:
        issues["broken_links"] = broken_links[:20]

    index_issues = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        idx_path = dir_path / "_index.md"
        if not idx_path.exists():
            continue
        try:
            idx_text = idx_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in re.finditer(r'\[\[([^\]]+)\]\]', idx_text):
            link = m.group(1)
            if link not in all_pages and link not in all_pages_with_dir:
                link_name = link.rsplit("/", 1)[-1] if "/" in link else link
                if link_name not in all_pages:
                    index_issues.append(f"{dir_name}/_index.md references non-existent page: [[{link}]]")
    if index_issues:
        issues["index_inconsistencies"] = index_issues[:10]

    type_issues = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            page_type = _extract_type_from_content(text)
            if page_type and page_type != dir_name:
                type_issues.append(f"{dir_name}/{md.stem}: type='{page_type}' but in {dir_name}/ dir")
    if type_issues:
        issues["type_path_mismatch"] = type_issues[:10]

    digest_issues = []
    digests_dir = wiki_dir / "sources" / "digests"
    if digests_dir.exists():
        for md in sorted(digests_dir.glob("*.md")):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            missing_secs = [s for s in _DIGEST_REQUIRED_SECTIONS if f"## {s}" not in text]
            if missing_secs:
                digest_issues.append(f"sources/digests/{md.name}: missing [{', '.join(missing_secs)}]")
    if digest_issues:
        issues["digest_incomplete"] = digest_issues

    name_locations: dict[str, list[str]] = {}
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            name_locations.setdefault(md.stem, []).append(f"{dir_name}/{md.stem}")
    duplicates = [{"title": name, "locations": locs} for name, locs in name_locations.items() if len(locs) > 1]
    if duplicates:
        issues["duplicates"] = duplicates[:10]

    missing_summary = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            result = config.split_frontmatter(text)
            if result is None:
                continue
            _, _, body = result
            lines = body.strip().split("\n")
            found_title = False
            has_blockquote = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    found_title = True
                    continue
                if found_title:
                    if stripped == "":
                        continue
                    if stripped.startswith("> "):
                        has_blockquote = True
                    break
            if found_title and not has_blockquote:
                missing_summary.append(f"{dir_name}/{md.stem}")
    if missing_summary:
        issues["missing_summary"] = missing_summary

    _KNOWLEDGE_REQUIRED = ["Key Facts", "Related Pages"]
    missing_sections = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            result = config.split_frontmatter(text)
            body = result[2] if result else text
            missing = [s for s in _KNOWLEDGE_REQUIRED if f"## {s}" not in body]
            if missing:
                missing_sections.append(f"{dir_name}/{md.stem}: missing [{', '.join(missing)}]")
    if missing_sections:
        issues["missing_sections"] = missing_sections

    _ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\ufeff\u00ad]')
    zero_width_issues = []
    if wiki_dir.exists():
        for md in sorted(wiki_dir.rglob("*.md")):
            if _ZERO_WIDTH_RE.search(md.name):
                zero_width_issues.append(f"{md.relative_to(wiki_dir)}: filename contains zero-width chars")
            try:
                head = md.read_text(encoding="utf-8")[:500]
                if _ZERO_WIDTH_RE.search(head):
                    zero_width_issues.append(f"{md.relative_to(wiki_dir)}: content contains zero-width chars")
            except OSError:
                pass
    if zero_width_issues:
        issues["zero_width_chars"] = zero_width_issues

    hollow_pages = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            result = config.split_frontmatter(text)
            body = result[2] if result else text
            body_lines = [l for l in body.strip().split("\n")
                          if l.strip() and not l.strip().startswith("#")]
            if len(body_lines) < 3:
                hollow_pages.append(f"{dir_name}/{md.stem}: only {len(body_lines)} effective lines")
    if hollow_pages:
        issues["hollow_pages"] = hollow_pages

    invalid_files = []
    digests_dir2 = wiki_dir / "sources" / "digests"
    if digests_dir2.exists():
        for md in digests_dir2.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                size = md.stat().st_size
            except OSError:
                continue
            if size < 100:
                invalid_files.append(f"sources/digests/{md.name}: too small ({size} bytes)")
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                size = md.stat().st_size
            except OSError:
                continue
            if size < 100:
                invalid_files.append(f"{dir_name}/{md.name}: too small ({size} bytes)")
    if invalid_files:
        issues["invalid_files"] = invalid_files

    related_page_issues = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "## Related Pages" not in text:
                continue
            in_section = False
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped == "## Related Pages":
                    in_section = True
                    continue
                if in_section and stripped.startswith("## "):
                    break
                if in_section and stripped.startswith("- ") and not re.search(r'\[\[', stripped):
                    related_page_issues.append(f"{dir_name}/{md.stem}: Related Pages item missing link — `{stripped[:60]}`")
                if in_section and (stripped.startswith("—") or stripped.startswith("–")) and not re.search(r'\[\[', stripped):
                    related_page_issues.append(f"{dir_name}/{md.stem}: Related Pages item missing link — `{stripped[:60]}`")
    if related_page_issues:
        issues["related_page_format"] = related_page_issues[:20]

    related_source_issues = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "## Related Sources" not in text:
                continue
            in_section = False
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped == "## Related Sources":
                    in_section = True
                    continue
                if in_section and stripped.startswith("## "):
                    break
                if in_section and stripped and not stripped.startswith("#"):
                    all_links = re.findall(r'\[\[([^\]]+)\]\]', stripped)
                    non_digest = [l for l in all_links if not l.startswith("sources/digests/")]
                    if non_digest:
                        related_source_issues.append(f"{dir_name}/{md.stem}: Related Sources has non-digest link — `{stripped[:80]}`")
                    elif not all_links and stripped.startswith("- "):
                        related_source_issues.append(f"{dir_name}/{md.stem}: Related Sources item missing link — `{stripped[:80]}`")
    if related_source_issues:
        issues["related_source_format"] = related_source_issues[:20]

    completeness_issues = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            result = config.split_frontmatter(text)
            if not result:
                continue
            _, fm_text, _ = result
            has_type = bool(re.search(r'^type:', fm_text, re.MULTILINE))
            has_tags = bool(re.search(r'^tags:', fm_text, re.MULTILINE))
            missing_fields = []
            if not has_type:
                missing_fields.append("type")
            if not has_tags:
                missing_fields.append("tags")
            if missing_fields:
                completeness_issues.append(f"{dir_name}/{md.stem}: missing {', '.join(missing_fields)}")
    if completeness_issues:
        issues["completeness"] = completeness_issues[:20]

    md_suffix_issues = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            result = config.split_frontmatter(text)
            if not result:
                continue
            _, fm_text, _ = result
            for line in fm_text.split("\n"):
                if line.strip().startswith("source_article:") or line.strip().startswith("- "):
                    val = line.split(":", 1)[-1].strip().strip('"').strip("'") if ":" in line else line.strip().lstrip("- ").strip('"').strip("'")
                    if val.endswith(".md"):
                        md_suffix_issues.append(f"{dir_name}/{md.stem}: value '{val}' has .md suffix")
                        break
    if md_suffix_issues:
        issues["md_suffix"] = md_suffix_issues[:10]

    # Append to the error book (skipped when the error book is disabled).
    if _ENABLE_ERROR_BOOK and issues:
        try:
            error_book.record_lint_issues(issues)
        except Exception as e:
            print(f"  ⚠️ Failed to record to error book: {e}")

    return issues


def auto_fix_bench():
    """Full auto-fix pass (pure code, no LLM).

    Fixes applied:
      1. Broken wikilinks: drop links to non-existent pages.
      2. Index inconsistency: remove non-existent entries from _index.md.
      3. Type/path mismatch: move files into the correct directory.
      4. Rebuild every _index.md.
      5. Digest placeholder content checks.
      6. Missing blockquote summary: convert list-item form to blockquote.
      7. Strip non-digest links from "Related Sources" sections.
      8. Frontmatter shape fixes (e.g. merging split ``type`` fields).
      9. Completeness: fill in missing ``type``/``tags`` fields.
      10. Drop invalid files (e.g. empty/too-small ones).
      11. Related Pages format fixes (drop list items without links).
      12. Orphan digest repair (re-link to a knowledge page).
      13. Drop free-text lines from "Related Sources".
      14. Zero-width character fixes (filenames and content).
      15. ``.md`` suffix fixes (strip ``.md`` from frontmatter values).
      16. Related Sources format fixes (drop non-digest link lines).
      17. Append a log entry to ``lint_ledger.jsonl``.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir or not wiki_dir.exists():
        return {}

    results = {}

    all_pages: set[str] = set()
    all_pages_with_dir: set[str] = set()
    for dir_name, dir_path in config.get_page_dirs().items():
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            all_pages.add(md.stem)
            all_pages_with_dir.add(f"{dir_name}/{md.stem}")

    broken_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            new_text = text
            for m in re.finditer(r'\[\[([^\]]+)\]\]', text):
                link = m.group(1)
                if "|" in link:
                    link = link.split("|")[0].strip()
                link_name = link.rsplit("/", 1)[-1] if "/" in link else link
                if link not in all_pages and link not in all_pages_with_dir and link_name not in all_pages:
                    pass  # do not delete in-body links, only record
            if "## Related Pages" in new_text:
                lines = new_text.split("\n")
                new_lines = []
                in_related = False
                for line in lines:
                    if line.strip().startswith("## Related Pages"):
                        in_related = True
                        new_lines.append(line)
                        continue
                    if in_related and line.strip().startswith("## "):
                        in_related = False
                    if in_related and "[[" in line:
                        link_match = re.search(r'\[\[([^\]]+)\]\]', line)
                        if link_match:
                            link = link_match.group(1)
                            if "|" in link:
                                link = link.split("|")[0].strip()
                            link_name = link.rsplit("/", 1)[-1] if "/" in link else link
                            if link not in all_pages and link not in all_pages_with_dir and link_name not in all_pages:
                                broken_fixed += 1
                                continue  # skip broken-link line
                    new_lines.append(line)
                new_text = "\n".join(new_lines)
            if new_text != text:
                md.write_text(new_text, encoding="utf-8")
    results["broken_links"] = broken_fixed

    index_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        idx_path = dir_path / "_index.md"
        if not idx_path.exists():
            continue
        try:
            idx_text = idx_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = idx_text.split("\n")
        new_lines = []
        for line in lines:
            link_match = re.search(r'\[\[([^\]]+)\]\]', line)
            if link_match:
                link = link_match.group(1)
                link_name = link.rsplit("/", 1)[-1] if "/" in link else link
                if link_name not in all_pages:
                    index_fixed += 1
                    continue  # drop the non-existent entry
            new_lines.append(line)
        if index_fixed > 0:
            idx_path.write_text("\n".join(new_lines), encoding="utf-8")
    results["index_inconsistencies"] = index_fixed

    type_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in list(dir_path.glob("*.md")):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            page_type = _extract_type_from_content(text)
            if page_type and page_type != dir_name:
                target_dir = wiki_dir / page_type
                if target_dir.exists():
                    target_path = target_dir / md.name
                    if not target_path.exists():
                        import shutil
                        shutil.move(str(md), str(target_path))
                        type_fixed += 1
                        print(f"  🔧 Moved {dir_name}/{md.name} → {page_type}/{md.name}")
    results["type_path_mismatch"] = type_fixed

    idx_rebuilt = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        pages = [f for f in dir_path.glob("*.md") if f.name != "_index.md"]
        if not pages:
            continue
        idx_path = dir_path / "_index.md"
        existing_entries = set()
        if idx_path.exists():
            idx_text = idx_path.read_text(encoding="utf-8")
            existing_entries = set(re.findall(r'\[\[([^\]]+)\]\]', idx_text))
        missing = [p for p in pages if p.stem not in existing_entries]
        if missing:
            for p in missing:
                content = p.read_text(encoding="utf-8")
                _append_to_index(dir_name, p.stem, content)
                idx_rebuilt += 1
    results["knowledge_index"] = idx_rebuilt

    digest_placeholder_fixed = 0
    digests_dir = wiki_dir / "sources" / "digests"
    if digests_dir and digests_dir.exists():
        for md in sorted(digests_dir.glob("*.md")):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            new_text = text
            for section in _DIGEST_REQUIRED_SECTIONS:
                pattern = rf'(## {re.escape(section)}\n)\(to be filled\)\n'
                if re.search(pattern, new_text):
                    pass
            if new_text != text:
                md.write_text(new_text, encoding="utf-8")
                digest_placeholder_fixed += 1
    results["digest_placeholder"] = digest_placeholder_fixed

    summary_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            result = config.split_frontmatter(text)
            if result is None:
                continue
            before, fm_text, body = result
            lines = body.strip().split("\n")
            found_title = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    found_title = True
                    continue
                if found_title:
                    if stripped == "":
                        continue
                    if stripped.startswith("- ") and not stripped.startswith("- [["):
                        summary_text = stripped[2:]
                        lines[i] = f"> {summary_text}"
                        new_body = "\n".join(lines)
                        new_text = f"---\n{fm_text}\n---\n{new_body}"
                        md.write_text(new_text, encoding="utf-8")
                        summary_fixed += 1
                    break
    results["summary_format"] = summary_fixed

    related_source_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "## Related Sources" not in text:
                continue
            lines = text.split("\n")
            new_lines = []
            in_related_src = False
            changed = False
            for line in lines:
                stripped = line.strip()
                if stripped == "## Related Sources":
                    in_related_src = True
                    new_lines.append(line)
                    continue
                if in_related_src and stripped.startswith("## "):
                    in_related_src = False
                if in_related_src and "[[" in stripped:
                    if "sources/digests/" not in stripped and stripped.startswith("- [["):
                        changed = True
                        related_source_fixed += 1
                        continue  # drop non-digest link
                new_lines.append(line)
            if changed:
                md.write_text("\n".join(new_lines), encoding="utf-8")
    results["related_source_format"] = related_source_fixed

    fm_format_fixed = 0
    digests_dir_fm = wiki_dir / "sources" / "digests"
    if digests_dir_fm.exists():
        for md in digests_dir_fm.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.startswith("---"):
                continue
            new_text = text
            new_text = re.sub(
                r'^(type:\s*)source_date:\s*(\d{4}-\d{2}-\d{2})\s*$',
                r'type: source\nsource_date: \2',
                new_text,
                flags=re.MULTILINE
            )
            if new_text != text:
                md.write_text(new_text, encoding="utf-8")
                fm_format_fixed += 1
    results["frontmatter_format"] = fm_format_fixed

    completeness_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            if "sources/articles" in str(md):
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.startswith("---"):
                continue
            fm_result = config.split_frontmatter(text)
            if fm_result is None:
                continue
            _, fm_text, body = fm_result
            needs_fix = False
            new_fm_lines = fm_text.strip().split("\n")
            has_type = any(l.strip().startswith("type:") for l in new_fm_lines)
            has_tags = any(l.strip().startswith("tags:") for l in new_fm_lines)
            if not has_type:
                expected_type = "source" if dir_name.startswith("sources") else dir_name
                new_fm_lines.insert(0, f"type: {expected_type}")
                needs_fix = True
            if not has_tags:
                tag = "source" if dir_name.startswith("sources") else dir_name
                new_fm_lines.append(f"tags: [{tag}]")
                needs_fix = True
            if needs_fix:
                new_fm = "\n".join(new_fm_lines)
                new_text = f"---\n{new_fm}\n---{body}"
                md.write_text(new_text, encoding="utf-8")
                completeness_fixed += 1
    results["completeness"] = completeness_fixed

    invalid_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if not dir_path.exists():
            continue
        for md in list(dir_path.glob("*.md")):
            if md.name == "_index.md":
                continue
            if md.stat().st_size < 100:
                try:
                    md.unlink()
                    invalid_fixed += 1
                    print(f"  🗑️ Deleted invalid file: {dir_name}/{md.name} ({md.stat().st_size if md.exists() else 0}B)")
                except OSError:
                    pass
    results["invalid_files"] = invalid_fixed

    rp_format_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "## Related Pages" not in text:
                continue
            lines = text.split("\n")
            new_lines = []
            in_rp = False
            changed = False
            for line in lines:
                stripped = line.strip()
                if stripped == "## Related Pages":
                    in_rp = True
                    new_lines.append(line)
                    continue
                if in_rp and stripped.startswith("## "):
                    in_rp = False
                if in_rp and stripped.startswith("- ") and not re.search(r'\[\[', stripped):
                    changed = True
                    rp_format_fixed += 1
                    continue
                if in_rp and (stripped.startswith("—") or stripped.startswith("–")) and not re.search(r'\[\[', stripped):
                    changed = True
                    rp_format_fixed += 1
                    continue
                new_lines.append(line)
            if changed:
                new_text = "\n".join(new_lines)
                new_text = re.sub(r'\n{3,}', '\n\n', new_text)
                md.write_text(new_text, encoding="utf-8")
    results["related_page_format"] = rp_format_fixed


    rs_text_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "## Related Sources" not in text:
                continue
            lines = text.split("\n")
            new_lines = []
            in_rs = False
            changed = False
            for line in lines:
                stripped = line.strip()
                if stripped == "## Related Sources":
                    in_rs = True
                    new_lines.append(line)
                    continue
                if in_rs and stripped.startswith("## "):
                    in_rs = False
                if in_rs and stripped and not stripped.startswith("#") and "[[" not in stripped:
                    changed = True
                    rs_text_fixed += 1
                    continue
                new_lines.append(line)
            if changed:
                new_text = "\n".join(new_lines)
                new_text = re.sub(r'\n{3,}', '\n\n', new_text)
                md.write_text(new_text, encoding="utf-8")
    results["related_source_text"] = rs_text_fixed

    _ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\ufeff\u00ad]')
    zw_fixed = 0
    if wiki_dir.exists():
        for md in sorted(wiki_dir.rglob("*.md")):
            if _ZERO_WIDTH_RE.search(md.name):
                new_name = _ZERO_WIDTH_RE.sub('', md.name)
                new_path = md.parent / new_name
                if not new_path.exists():
                    md.rename(new_path)
                    zw_fixed += 1
                    md = new_path
                else:
                    md.unlink()
                    zw_fixed += 1
                    continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            new_text = _ZERO_WIDTH_RE.sub('', text)
            if new_text != text:
                md.write_text(new_text, encoding="utf-8")
                zw_fixed += 1
    results["zero_width_chars"] = zw_fixed

    md_suffix_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.startswith("---"):
                continue
            new_text = text
            # Normalise `source_article: xxx.md` to `source_article: xxx`.
            new_text = re.sub(
                r'^(source_article:\s*["\']?)(.+?)(\.md)(["\']?\s*)$',
                r'\1\2\4',
                new_text,
                flags=re.MULTILINE
            )
            new_text = re.sub(
                r'^(\s*-\s*["\']?)(.+?)(\.md)(["\']?\s*)$',
                lambda m: m.group(0) if '/' not in m.group(2) and not m.group(2).startswith('source') else f"{m.group(1)}{m.group(2)}{m.group(4)}",
                new_text,
                flags=re.MULTILINE
            )
            if new_text != text:
                md.write_text(new_text, encoding="utf-8")
                md_suffix_fixed += 1
    results["md_suffix"] = md_suffix_fixed

    rs_format_fixed = 0
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "## Related Sources" not in text:
                continue
            lines = text.split("\n")
            new_lines = []
            in_rs = False
            changed = False
            for line in lines:
                stripped = line.strip()
                if stripped == "## Related Sources":
                    in_rs = True
                    new_lines.append(line)
                    continue
                if in_rs and stripped.startswith("## "):
                    in_rs = False
                if in_rs and "[[" in stripped:
                    all_links = re.findall(r'\[\[([^\]]+)\]\]', stripped)
                    non_digest = [l for l in all_links if not l.startswith("sources/digests/")]
                    if non_digest and not any(l.startswith("sources/digests/") for l in all_links):
                        changed = True
                        rs_format_fixed += 1
                        continue
                new_lines.append(line)
            if changed:
                new_text = "\n".join(new_lines)
                new_text = re.sub(r'\n{3,}', '\n\n', new_text)
                md.write_text(new_text, encoding="utf-8")
    results["related_source_format"] = rs_format_fixed

    try:
        ledger_type_map = {
            "broken_links": "broken_link",
            "index_inconsistencies": "index_error",
            "type_path_mismatch": "type_path_mismatch",
            "knowledge_index": "knowledge_index",
            "digest_placeholder": "digest_incomplete",
            "summary_format": "missing_summary",
            "related_source_format": "related_source_format",
            "frontmatter_format": "incomplete",
            "completeness": "incomplete",
            "invalid_files": "invalid_file",
            "related_page_format": "related_page_format",
            "related_source_text": "related_source_format",
            "zero_width_chars": "zero_width_chars",
            "md_suffix": "md_suffix",
        }
        for key, count in results.items():
            if count > 0:
                error_book.append_ledger(
                    issue_type=ledger_type_map.get(key, key),
                    auto_fixed=True,
                    fix_method=f"auto_fix_{key}",
                    count=count,
                )
    except Exception:
        pass

    return results



def llm_fix_incomplete_digests() -> int:
    """Use the LLM to complete incomplete digest pages.

    Scan ``sources/digests/`` for pages that miss required sections or contain
    only placeholder text, then generate the missing content from the original
    article via the LLM.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return 0

    digests_dir = wiki_dir / "sources" / "digests"
    articles_dir = wiki_dir / "sources" / "articles"
    if not digests_dir.exists():
        return 0

    incomplete = []
    for md in sorted(digests_dir.glob("*.md")):
        if md.name == "_index.md":
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        result = config.split_frontmatter(text)
        if result is None:
            continue
        _, fm_text, body = result

        missing = []
        for section in _DIGEST_REQUIRED_SECTIONS:
            if f"## {section}" not in body:
                missing.append(section)
            else:
                pattern = rf'## {re.escape(section)}\s*\n(.*?)(?=\n## |\Z)'
                m = re.search(pattern, body, re.DOTALL)
                if m:
                    content = m.group(1).strip()
                    if not content or content in ("(to be filled)", "- (to be filled)"):
                        missing.append(section)

        if not missing:
            continue

        article_path = None
        # 1) frontmatter.source_article
        for line in fm_text.split("\n"):
            if line.strip().startswith("source_article:"):
                src_stem = line.split(":", 1)[1].strip().strip('"').strip("'")
                if src_stem and articles_dir.exists():
                    candidate = articles_dir / f"{src_stem}.md"
                    if candidate.exists():
                        article_path = candidate
                break
        if article_path is None and articles_dir.exists():
            candidate = articles_dir / md.name
            if candidate.exists():
                article_path = candidate
        if article_path is None and articles_dir.exists() and len(md.name) >= 10 and md.name[4] == '-':
            date_prefix = md.name[:10]
            matches = list(articles_dir.glob(f"{date_prefix}-*.md"))
            if matches:
                article_path = matches[0]

        incomplete.append((md, missing, article_path))

    if not incomplete:
        return 0

    print(f"  📄 Found {len(incomplete)} incomplete digests, starting LLM fix...")

    fixed = 0
    fixed_names = []
    for md, missing, article_path in incomplete:
        article_text = ""
        if article_path and article_path.exists():
            article_text = article_path.read_text(encoding="utf-8")
            r = config.split_frontmatter(article_text)
            if r:
                article_text = r[2]
        else:
            if set(missing) - {"Key Entities"}:
                continue

        digest_text = md.read_text(encoding="utf-8")

        sys_prompt = "You are a knowledge base maintenance expert. Complete the missing sections of the digest page based on the original article. Only output the sections that need to be added, do not repeat existing content."
        article_block = f"## Original Article\n{article_text}" if article_text else "## Original Article\n(not available, use existing digest content to extract information)"
        user_prompt = f"""## Current Digest Page
{digest_text}

## Missing/Incomplete Sections
{', '.join(missing)}

{article_block}

Please complete the missing sections. Format requirements:
- ## Summary: ≤200 word structured summary
- ## Key Facts: Factual information, each item starts with -
- ## Key Entities: Entity names mentioned, each item starts with - (no [[]] links)
- ## Related Context: Background context and connections

Only output sections that need to be added (starting with ##), do not output existing complete sections."""

        try:
            result = call_llm(sys_prompt, user_prompt, max_tokens=2048,
                            model=config.LLM_PREMIUM_MODEL, temperature=0.2)
            if not result or not result.strip():
                continue

            r = config.split_frontmatter(digest_text)
            if r is None:
                continue
            before, fm, body = r

            llm_sections = {}
            current_sec = None
            current_lines = []
            for line in result.strip().split("\n"):
                if line.strip().startswith("## "):
                    if current_sec:
                        llm_sections[current_sec] = "\n".join(current_lines)
                    current_sec = line.strip()[3:].strip()
                    current_lines = [line]
                else:
                    current_lines.append(line)
            if current_sec:
                llm_sections[current_sec] = "\n".join(current_lines)

            for section in missing:
                if section not in llm_sections:
                    continue
                placeholder_pattern = rf'## {re.escape(section)}\s*\n\(?to be filled\)?\s*'
                if re.search(placeholder_pattern, body):
                    body = re.sub(placeholder_pattern, llm_sections[section] + "\n", body)
                elif f"## {section}" not in body:
                    body = body.rstrip() + "\n\n" + llm_sections[section] + "\n"

            new_text = f"---\n{fm}\n---\n{body}"
            md.write_text(new_text, encoding="utf-8")
            fixed += 1
            fixed_names.append(f"sources/digests/{md.name}")
            print(f"    ✅ {md.name} — completed [{', '.join(missing)}]")
        except Exception as e:
            print(f"    ⚠️ {md.name} — LLM fix failed: {e}")

    if fixed_names:
        try:
            error_book.mark_samples_fixed("digest_incomplete", fixed_names)
        except Exception:
            pass

    return fixed


def llm_fix_missing_summary() -> int:
    """Use the LLM to add a one-sentence blockquote summary to knowledge pages.

    Rule: a knowledge page must have a ``> ...`` blockquote summary right after
    its ``# Title`` line. The pure-code pass already handles "list item -> blockquote"
    conversion; this pass handles pages that are missing the summary entirely.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return 0

    missing_pages = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            result = config.split_frontmatter(text)
            if result is None:
                continue
            _, fm_text, body = result
            lines = body.strip().split("\n")
            found_title = False
            has_blockquote = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    found_title = True
                    continue
                if found_title:
                    if stripped == "":
                        continue
                    if stripped.startswith("> "):
                        has_blockquote = True
                    break
            if found_title and not has_blockquote:
                missing_pages.append((md, dir_name))

    if not missing_pages:
        return 0

    print(f"  📝 Found {len(missing_pages)} knowledge pages missing summary, starting LLM fix...")

    fixed = 0
    for md, dir_name in missing_pages:
        text = md.read_text(encoding="utf-8")
        result = config.split_frontmatter(text)
        if result is None:
            continue
        before, fm, body = result

        sys_prompt = "You are a knowledge base expert. Generate a one-sentence summary in blockquote format for the given knowledge page."
        user_prompt = f"""## Knowledge Page Content
{body.strip()}

Generate ONE line of blockquote summary that describes the core identity of this entity/concept.
Format: > One-sentence summary
Example: > Baroque-era German composer, father of modern Western music"""

        try:
            llm_result = call_llm(sys_prompt, user_prompt, max_tokens=256,
                                  model=config.LLM_FAST_MODEL, temperature=0.2)
            summary_line = llm_result.strip().split("\n")[0]
            if not summary_line.startswith("> "):
                summary_line = f"> {summary_line.lstrip('> ')}"

            body_lines = body.split("\n")
            for i, line in enumerate(body_lines):
                if line.strip().startswith("# ") and not line.strip().startswith("## "):
                    j = i + 1
                    while j < len(body_lines) and body_lines[j].strip() == "":
                        j += 1
                    body_lines.insert(j, summary_line)
                    break

            body = "\n".join(body_lines)
            new_text = f"---\n{fm}\n---\n{body}"
            md.write_text(new_text, encoding="utf-8")
            fixed += 1
        except Exception as e:
            print(f"    ⚠️ {dir_name}/{md.stem} — LLM summary failed: {e}")

    return fixed


def llm_fix_missing_sections() -> int:
    """Use the LLM to fill in required sections (Key Facts / Related Pages) on knowledge pages."""
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return 0

    _KNOWLEDGE_REQUIRED_SECTIONS = ["Key Facts", "Related Pages"]

    all_page_names = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for f in dir_path.glob("*.md"):
            if f.name != "_index.md":
                all_page_names.append(f"{dir_name}/{f.stem}")
    all_page_names.sort()

    digests_dir = wiki_dir / "sources" / "digests"
    digest_info = {}
    if digests_dir and digests_dir.exists():
        for md in digests_dir.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            desc = md.stem
            r = config.split_frontmatter(text)
            body = r[2] if r else text
            for line in body.split("\n"):
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    desc = stripped[2:].strip()
                    break
                if stripped.startswith("> "):
                    desc = stripped[2:].strip()
                    break
            digest_info[md.stem] = desc

    missing_pages = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            r = config.split_frontmatter(text)
            body = r[2] if r else text
            missing = [s for s in _KNOWLEDGE_REQUIRED_SECTIONS if f"## {s}" not in body]
            if missing:
                missing_pages.append((md, dir_name, missing))

    if not missing_pages:
        return 0

    print(f"  📑 Found {len(missing_pages)} knowledge pages missing sections, starting LLM fix...")

    page_list_text = ", ".join(all_page_names[:200])
    digest_list_text = "\n".join(
        f"- [[sources/digests/{stem}]] — {desc}"
        for stem, desc in sorted(digest_info.items())
    )[:3000]

    fixed = 0
    for md, dir_name, missing in missing_pages:
        text = md.read_text(encoding="utf-8")
        r = config.split_frontmatter(text)
        if r is None:
            continue
        before, fm, body = r

        context_parts = []
        if "Related Pages" in missing:
            context_parts.append(f"Available pages for Related Pages: {page_list_text}")
        if "Related Sources" in missing and digest_info:
            context_parts.append(f"Available digests for Related Sources:\n{digest_list_text}")
        context_hint = "\n".join(context_parts)

        sys_prompt = "You are a knowledge base maintenance expert. Complete the missing required sections for the knowledge page. Only output sections that need to be added."
        user_prompt = f"""## Current Knowledge Page Content
{body.strip()}

## Missing Sections
{', '.join(missing)}

{context_hint}

Please complete the missing sections. Format:
- ## Key Facts: At least 2 facts, each starting with -
- ## Related Pages: Related knowledge page links, format `- [[type/name]] — brief description`
- ## Related Sources: Related digest links, format `- [[sources/digests/filename]] — brief description`

Only output sections that need to be added (starting with ##)."""

        try:
            llm_result = call_llm(sys_prompt, user_prompt, max_tokens=1024,
                                  model=config.LLM_FAST_MODEL, temperature=0.2)
            if not llm_result or not llm_result.strip():
                continue

            llm_sections = {}
            current_sec = None
            current_lines = []
            for line in llm_result.strip().split("\n"):
                if line.strip().startswith("## "):
                    if current_sec:
                        llm_sections[current_sec] = "\n".join(current_lines)
                    current_sec = line.strip()[3:].strip()
                    current_lines = [line]
                else:
                    current_lines.append(line)
            if current_sec:
                llm_sections[current_sec] = "\n".join(current_lines)

            for section in missing:
                if section in llm_sections:
                    body = body.rstrip() + "\n\n" + llm_sections[section] + "\n"

            new_text = f"---\n{fm}\n---\n{body}"
            md.write_text(new_text, encoding="utf-8")
            fixed += 1
        except Exception as e:
            print(f"    ⚠️ {dir_name}/{md.stem} — LLM section fix failed: {e}")

    return fixed


def llm_fix_empty_related_pages(batch_size: int = 10, max_pages: int = 0) -> int:
    """Repair knowledge pages with empty "Related Pages" using code pre-matching + LLM selection.

    Pipeline:
      1. Code pre-match: for each knowledge page with an empty Related Pages
         section, gather candidates using:
           - Other entities co-occurring in the same digest.
           - Pages in the same directory with overlapping tags.
           - Pages already linked via in-body [[...]] (reverse-link signal).
      2. Batch LLM selection: feed the candidate list together with the page
         summary to the LLM and let it pick the 3-5 most relevant items.

    Args:
        batch_size: number of pages to process per LLM call (default 10).
        max_pages:  cap on the number of pages to repair (0 means no cap).

    Returns:
        Number of pages repaired.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return 0


    all_pages = {}
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            r = config.split_frontmatter(text)
            if r is None:
                continue
            _, fm, body = r
            tags = set()
            tags_match = re.search(r'^tags:\s*\[([^\]]*)\]', fm, re.MULTILINE)
            if tags_match:
                for t in tags_match.group(1).split(","):
                    t = t.strip().strip('"').strip("'")
                    if t:
                        tags.add(t.lower())
            else:
                # YAML list format
                in_tags = False
                for line in fm.split("\n"):
                    if line.strip().startswith("tags:"):
                        in_tags = True
                        continue
                    if in_tags:
                        if line.strip().startswith("- "):
                            tags.add(line.strip()[2:].strip().strip('"').strip("'").lower())
                        elif line.strip() and not line.startswith(" "):
                            break
            aliases = set()
            alias_match = re.search(r'^aliases:\s*\[([^\]]*)\]', fm, re.MULTILINE)
            if alias_match:
                for a in alias_match.group(1).split(","):
                    a = a.strip().strip('"').strip("'")
                    if a:
                        aliases.add(a)
            body_snippet = body.strip()[:200]

            all_pages[md.stem] = {
                "path": md,
                "dir_name": dir_name,
                "tags": tags,
                "aliases": aliases,
                "body_snippet": body_snippet,
            }

    if not all_pages:
        return 0

    # digest_entities: {digest_stem: set of page_stems mentioned}
    digest_entities: dict[str, set] = {}
    digests_dir = wiki_dir / "sources" / "digests"
    if digests_dir and digests_dir.exists():
        name_to_stem: dict[str, str] = {}
        for stem, info in all_pages.items():
            name_to_stem[stem.lower()] = stem
            name_to_stem[stem.replace("-", " ").lower()] = stem
            for alias in info["aliases"]:
                name_to_stem[alias.lower()] = stem

        for digest_md in digests_dir.glob("*.md"):
            if digest_md.name == "_index.md":
                continue
            try:
                dtext = digest_md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            entities_in_digest = set()
            in_entities = False
            for line in dtext.split("\n"):
                if line.strip() == "## Key Entities":
                    in_entities = True
                    continue
                if in_entities and line.strip().startswith("## "):
                    break
                if in_entities and line.strip().startswith("- "):
                    entity_name = line.strip()[2:].strip()
                    key = entity_name.lower()
                    if key in name_to_stem:
                        entities_in_digest.add(name_to_stem[key])
            for m in re.finditer(r'\[\[([^\]]+)\]\]', dtext):
                link = m.group(1)
                if "|" in link:
                    link = link.split("|")[0].strip()
                link_stem = link.rsplit("/", 1)[-1] if "/" in link else link
                if link_stem in all_pages:
                    entities_in_digest.add(link_stem)

            if entities_in_digest:
                digest_entities[digest_md.stem] = entities_in_digest

    backlinks: dict[str, set] = {}
    for stem, info in all_pages.items():
        try:
            text = info["path"].read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in re.finditer(r'\[\[([^\]]+)\]\]', text):
            link = m.group(1)
            if "|" in link:
                link = link.split("|")[0].strip()
            link_stem = link.rsplit("/", 1)[-1] if "/" in link else link
            if link_stem in all_pages and link_stem != stem:
                backlinks.setdefault(link_stem, set()).add(stem)

    empty_rp_pages = []
    for stem, info in all_pages.items():
        try:
            text = info["path"].read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "## Related Pages" not in text:
            empty_rp_pages.append(stem)
        else:
            in_rp = False
            has_link = False
            for line in text.split("\n"):
                if line.strip() == "## Related Pages":
                    in_rp = True
                    continue
                if in_rp and line.strip().startswith("## "):
                    break
                if in_rp and "[[" in line:
                    has_link = True
                    break
            if in_rp and not has_link:
                empty_rp_pages.append(stem)

    if not empty_rp_pages:
        print("  ✅ No empty Related Pages found.")
        return 0

    if max_pages > 0:
        empty_rp_pages = empty_rp_pages[:max_pages]

    print(f"  📑 Found {len(empty_rp_pages)} pages with empty Related Pages, generating candidates...")

    # candidates: {page_stem: list of candidate_stems (scored)}
    page_candidates: dict[str, list] = {}

    for stem in empty_rp_pages:
        info = all_pages[stem]
        candidates_score: dict[str, float] = {}

        for d_stem, d_entities in digest_entities.items():
            if stem in d_entities:
                for co_entity in d_entities:
                    if co_entity != stem:
                        candidates_score[co_entity] = candidates_score.get(co_entity, 0) + 3.0

        my_tags = info["tags"]
        if my_tags:
            for other_stem, other_info in all_pages.items():
                if other_stem == stem:
                    continue
                overlap = len(my_tags & other_info["tags"])
                if overlap >= 1:
                    weight = 1.5 if other_info["dir_name"] == info["dir_name"] else 0.8
                    candidates_score[other_stem] = candidates_score.get(other_stem, 0) + overlap * weight

        if stem in backlinks:
            for bl_stem in backlinks[stem]:
                candidates_score[bl_stem] = candidates_score.get(bl_stem, 0) + 2.0

        try:
            text = info["path"].read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = ""
        for m in re.finditer(r'\[\[([^\]]+)\]\]', text):
            link = m.group(1)
            if "|" in link:
                link = link.split("|")[0].strip()
            link_stem = link.rsplit("/", 1)[-1] if "/" in link else link
            if link_stem in all_pages and link_stem != stem:
                candidates_score[link_stem] = candidates_score.get(link_stem, 0) + 1.5

        if not candidates_score:
            same_dir_pages = [
                s for s, inf in all_pages.items()
                if inf["dir_name"] == info["dir_name"] and s != stem
            ]
            for sd_stem in same_dir_pages[:10]:
                candidates_score[sd_stem] = 0.5

        sorted_candidates = sorted(candidates_score.items(), key=lambda x: -x[1])[:15]
        page_candidates[stem] = [c[0] for c in sorted_candidates]

    print(f"  🤖 Starting LLM selection in batches of {batch_size}...")

    fixed = 0
    batches = [empty_rp_pages[i:i + batch_size] for i in range(0, len(empty_rp_pages), batch_size)]

    for batch_idx, batch in enumerate(batches):
        pages_info_parts = []
        for stem in batch:
            info = all_pages[stem]
            candidates = page_candidates.get(stem, [])
            if not candidates:
                continue
            cand_descs = []
            for c_stem in candidates:
                c_info = all_pages.get(c_stem)
                if c_info:
                    cand_descs.append(f"    - {c_info['dir_name']}/{c_stem}")
            if not cand_descs:
                continue
            pages_info_parts.append(
                f"### Page: {info['dir_name']}/{stem}\n"
                f"  Summary: {info['body_snippet']}\n"
                f"  Candidates:\n" + "\n".join(cand_descs)
            )

        if not pages_info_parts:
            continue

        sys_prompt = (
            "You are a knowledge base curator. For each knowledge page below, "
            "select 3-5 most relevant related pages from the candidate list. "
            "Only select pages that have a meaningful semantic relationship "
            "(e.g., same topic, same event, same person, cause-effect, part-whole). "
            "Output in the exact format specified."
        )

        user_prompt = (
            "For each page below, select 3-5 related pages from its candidate list.\n\n"
            "Output format (one page per block, separated by blank lines):\n"
            "PAGE: <dir_name/page_stem>\n"
            "- [[<dir_name/selected_stem>]] — <one-sentence reason>\n"
            "- [[<dir_name/selected_stem>]] — <one-sentence reason>\n\n"
            "Pages to process:\n\n" + "\n\n".join(pages_info_parts)
        )

        try:
            llm_result = call_llm(sys_prompt, user_prompt, max_tokens=2048,
                                  model=config.LLM_FAST_MODEL, temperature=0.2)
            if not llm_result or not llm_result.strip():
                continue
        except Exception as e:
            print(f"    ⚠️ Batch {batch_idx + 1}/{len(batches)} LLM call failed: {e}")
            continue

        current_page = None
        current_links = []
        for line in llm_result.strip().split("\n"):
            line_s = line.strip()
            if line_s.startswith("PAGE:"):
                if current_page and current_links:
                    _write_related_pages(current_page, current_links, all_pages)
                    fixed += 1
                page_ref = line_s[5:].strip()
                page_stem = page_ref.rsplit("/", 1)[-1] if "/" in page_ref else page_ref
                current_page = page_stem if page_stem in all_pages else None
                current_links = []
            elif line_s.startswith("- [[") and current_page:
                current_links.append(line_s)

        if current_page and current_links:
            _write_related_pages(current_page, current_links, all_pages)
            fixed += 1

        if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
            print(f"    ✅ Batch {batch_idx + 1}/{len(batches)} done, total fixed: {fixed}")

    print(f"  🎉 Fixed {fixed}/{len(empty_rp_pages)} empty Related Pages.")
    return fixed


def _write_related_pages(page_stem: str, link_lines: list, all_pages: dict):
    """Write the LLM-selected Related Pages links back into the knowledge page."""
    info = all_pages.get(page_stem)
    if not info:
        return

    page_path = info["path"]
    try:
        text = page_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    r = config.split_frontmatter(text)
    if r is None:
        return
    before, fm, body = r

    valid_lines = []
    for line in link_lines:
        m = re.search(r'\[\[([^\]]+)\]\]', line)
        if m:
            link = m.group(1)
            if "|" in link:
                link = link.split("|")[0].strip()
            link_stem = link.rsplit("/", 1)[-1] if "/" in link else link
            if link_stem in all_pages:
                valid_lines.append(line)

    if not valid_lines:
        return

    rp_content = "\n".join(valid_lines)

    if "## Related Pages" in body:
        new_body_lines = []
        in_rp = False
        inserted = False
        for line in body.split("\n"):
            if line.strip() == "## Related Pages":
                in_rp = True
                new_body_lines.append(line)
                new_body_lines.append("")
                for vl in valid_lines:
                    new_body_lines.append(vl)
                inserted = True
                continue
            if in_rp:
                if line.strip().startswith("## "):
                    in_rp = False
                    new_body_lines.append("")
                    new_body_lines.append(line)
                continue
            new_body_lines.append(line)
        body = "\n".join(new_body_lines)
    else:
        if "## Related Sources" in body:
            body = body.replace(
                "## Related Sources",
                f"## Related Pages\n\n{rp_content}\n\n## Related Sources"
            )
        else:
            body = body.rstrip() + f"\n\n## Related Pages\n\n{rp_content}\n"

    new_text = f"---\n{fm}\n---\n{body}"
    page_path.write_text(new_text, encoding="utf-8")


def _inject_related_sources_for_page(page_path: Path, page_name: str):
    """Reverse-match digests for a single knowledge page and inject Related Sources links.

    Scan every digest under ``sources/digests/``; whenever a digest mentions the
    page name (or any of its aliases), inject a link to that digest into the
    knowledge page's ``## Related Sources`` section.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return

    digests_dir = wiki_dir / "sources" / "digests"
    if not digests_dir.exists():
        return

    try:
        kp_text = page_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    search_names = {page_name, page_name.replace("-", " ")}
    alias_match = re.search(r'^aliases:\s*\[([^\]]*)\]', kp_text, re.MULTILINE)
    if alias_match:
        for alias in alias_match.group(1).split(","):
            alias = alias.strip().strip('"').strip("'")
            if alias:
                search_names.add(alias)

    matched_digests = []
    for digest_md in digests_dir.glob("*.md"):
        if digest_md.name == "_index.md":
            continue
        try:
            digest_text = digest_md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        digest_lower = digest_text.lower()
        for name in search_names:
            if name.lower() in digest_lower:
                matched_digests.append(digest_md.stem)
                break

    if not matched_digests:
        return

    modified = False
    for stem in matched_digests:
        digest_link = f"[[sources/digests/{stem}]]"
        if digest_link in kp_text:
            continue  # already present, skip

        if "## Related Sources" in kp_text:
            lines = kp_text.split("\n")
            insert_idx = None
            in_rs = False
            for i, line in enumerate(lines):
                if line.strip() == "## Related Sources":
                    in_rs = True
                    continue
                if in_rs and line.strip().startswith("## "):
                    insert_idx = i
                    break
            if insert_idx is None:
                kp_text = kp_text.rstrip() + f"\n- {digest_link}\n"
            else:
                lines.insert(insert_idx, f"- {digest_link}")
                if insert_idx + 1 < len(lines) and lines[insert_idx + 1].strip().startswith("## "):
                    lines.insert(insert_idx + 1, "")
                kp_text = "\n".join(lines)
        else:
            kp_text = kp_text.rstrip() + f"\n\n## Related Sources\n- {digest_link}\n"
        modified = True

    if modified:
        page_path.write_text(kp_text, encoding="utf-8")
        print(f"    📎 Injected Related Sources for {page_name}")


def llm_fix_broken_links() -> int:
    """Read broken-link records from the error book and have the LLM create the missing knowledge pages in batches.

    Only active ``broken_link`` entries with ``fixed=False`` samples are processed.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return 0

    try:
        from bench_error_book import get_unfixed_samples, load_error_book, _sample_name
        errors = load_error_book()
        broken_errors = [e for e in errors
                         if e.get("category") == "broken_link" and e.get("status") != "closed"]
        if not broken_errors:
            return 0
        unfixed = []
        for e in broken_errors:
            unfixed.extend(get_unfixed_samples(e))
    except Exception:
        return 0

    if not unfixed:
        return 0

    missing_targets = set()
    for name in unfixed:
        if " → " in name:
            target = name.split(" → ")[-1].strip("[]")
        else:
            target = name.strip("[]")
        if target.startswith("sources/"):
            continue
        missing_targets.add(target)

    if not missing_targets:
        return 0

    all_pages = set()
    for dir_name, dir_path in config.get_page_dirs().items():
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name != "_index.md":
                all_pages.add(md.stem)
                all_pages.add(f"{dir_name}/{md.stem}")

    still_missing = []
    already_exist = []
    for target in missing_targets:
        clean = target.rsplit("/", 1)[-1] if "/" in target else target
        if target in all_pages or clean in all_pages:
            already_exist.append(target)
        else:
            still_missing.append(target)

    if already_exist:
        try:
            error_book.mark_samples_fixed("broken_link", already_exist)
        except Exception:
            pass

    if not still_missing:
        return 0

    batch = still_missing[:10]
    print(f"  🔗 Creating {len(batch)} missing pages for broken links...")

    ref_context = {}
    for target in batch:
        clean = target.rsplit("/", 1)[-1] if "/" in target else target
        refs = []
        for dir_name, dir_path in config.get_page_dirs().items():
            if not dir_path.exists():
                continue
            for md in dir_path.glob("*.md"):
                if md.name == "_index.md":
                    continue
                try:
                    text = md.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if f"[[{target}]]" in text or f"[[{clean}]]" in text:
                    lines = text.split("\n")
                    for i, line in enumerate(lines):
                        if f"[[{target}]]" in line or f"[[{clean}]]" in line:
                            ctx_lines = lines[max(0, i-1):i+2]
                            refs.append(f"From {dir_name}/{md.stem}:\n" + "\n".join(ctx_lines))
                            break
                    if len(refs) >= 3:
                        break
            if len(refs) >= 3:
                break
        ref_context[target] = refs

    page_types = config.get_page_types()
    dir_list = "\n".join(f"- {name}: {info.get('description', '')}" for name, info in page_types.items())

    fixed = 0
    fixed_names = []
    for target in batch:
        clean = target.rsplit("/", 1)[-1] if "/" in target else target
        target_dir = None
        if "/" in target:
            dir_prefix = target.split("/")[0]
            if dir_prefix in page_types:
                target_dir = dir_prefix

        refs_text = "\n\n".join(ref_context.get(target, [])) or "(no reference context available)"

        sys_prompt = "You are a knowledge base maintenance expert. Create a new knowledge page for a missing entity/concept that is referenced by other pages."
        user_prompt = f"""## Missing Page Name
{clean}

## Reference Context (how other pages reference this entity)
{refs_text}

## Available Directories
{dir_list}

{"## Suggested Directory: " + target_dir if target_dir else "Please choose the most appropriate directory from the list above."}

Please generate a complete knowledge page. Requirements:
- ## Related Pages MUST have at least 2 links in format `[[directory/page_name]]` — link to pages from the reference context above
- ## Related Sources can be empty if no specific digest is known

Output format:

---DIR: directory_name---
(just the directory name, e.g., "entities" or "events")

---CONTENT---
(complete page content in markdown, including frontmatter with type/tags/aliases, # Title, > summary, ## Key Facts, ## Related Pages, ## Related Sources)"""

        try:
            llm_result = call_llm(sys_prompt, user_prompt, max_tokens=2048,
                                  model=config.LLM_PREMIUM_MODEL, temperature=0.2)
            if not llm_result or not llm_result.strip():
                continue

            chosen_dir = target_dir
            content = llm_result.strip()

            dir_match = re.search(r'---DIR:\s*(\w+)\s*---', content)
            if dir_match:
                chosen_dir = dir_match.group(1)
                content = content[dir_match.end():].strip()

            content_match = re.search(r'---CONTENT---\s*\n(.*)', content, re.DOTALL)
            if content_match:
                content = content_match.group(1).strip()

            if not chosen_dir or chosen_dir not in page_types:
                type_match = re.search(r'type:\s*(\w+)', content)
                if type_match and type_match.group(1) in page_types:
                    chosen_dir = type_match.group(1)
                else:
                    chosen_dir = list(page_types.keys())[0] if page_types else "entities"

            target_path = wiki_dir / chosen_dir / f"{clean}.md"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                target_path.write_text(content, encoding="utf-8")
                fixed += 1
                fixed_names.append(target)
                print(f"    ✅ Created {chosen_dir}/{clean}")
                _append_to_index(chosen_dir, clean, content)
                _inject_related_sources_for_page(target_path, clean)
        except Exception as e:
            print(f"    ⚠️ {clean} — LLM page creation failed: {e}")

    if fixed_names:
        try:
            error_book.mark_samples_fixed("broken_link", fixed_names)
        except Exception:
            pass

    return fixed


def llm_verify_source_grounding(sample_size: int = 30) -> int:
    """Layer 2: spot-check whether Key Facts on knowledge pages are supported by sources (unsupported-fact detection and repair).

    Randomly sample knowledge pages and verify, fact by fact, whether each Key
    Fact can be located in the source digests it references. Unsupported facts
    are marked and removed.

    Args:
      sample_size  number of knowledge pages to sample per run (default 30).

    Returns:
      Number of unsupported facts removed.
    """
    import random

    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return 0

    knowledge_pages = []
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            knowledge_pages.append((md, dir_name))

    if not knowledge_pages:
        return 0

    sample = random.sample(knowledge_pages, min(sample_size, len(knowledge_pages)))

    digests_dir = wiki_dir / "sources" / "digests"
    removed_count = 0

    for md, dir_name in sample:
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if "## Key Facts" not in text:
            continue

        lines = text.split("\n")
        facts_start = -1
        facts_end = len(lines)
        for i, line in enumerate(lines):
            if line.strip() == "## Key Facts":
                facts_start = i + 1
            elif facts_start >= 0 and line.strip().startswith("## "):
                facts_end = i
                break

        if facts_start < 0:
            continue

        fact_lines = [l for l in lines[facts_start:facts_end] if l.strip().startswith("- ")]
        if not fact_lines:
            continue

        source_content = ""
        if "## Related Sources" in text:
            in_sources = False
            for line in lines:
                if line.strip() == "## Related Sources":
                    in_sources = True
                    continue
                if in_sources and line.strip().startswith("## "):
                    break
                if in_sources:
                    # Extract [[sources/digests/...]] links.
                    for m in re.finditer(r'\[\[sources/digests/([^\]]+)\]\]', line):
                        digest_name = m.group(1)
                        digest_path = digests_dir / f"{digest_name}.md"
                        if digest_path.exists():
                            try:
                                source_content += digest_path.read_text(encoding="utf-8") + "\n\n"
                            except (OSError, UnicodeDecodeError):
                                pass

        if not source_content:
            continue

        facts_text = "\n".join(fact_lines)
        sys_prompt = "You are a fact-checking expert. Verify whether each Key Fact is supported by the provided source content."
        user_prompt = f"""## Source Digest Content
{source_content[:3000]}

## Key Facts to Verify
{facts_text}

## Task
For each fact, determine if it is SUPPORTED or UNSUPPORTED by the source content above.
A fact is UNSUPPORTED if:
- It contains claims not mentioned in the source
- It adds details (dates, numbers, relations) not present in the source
- It makes generalizations beyond what the source states

Output JSON format:
{{
  "results": [
    {{"fact": "the fact text", "supported": true/false, "reason": "brief reason"}}
  ]
}}"""

        try:
            result = call_llm_json(sys_prompt, user_prompt,
                                   model=config.LLM_FAST_MODEL, temperature=0.1)
            results = result.get("results", [])

            unsupported_facts = [r for r in results if not r.get("supported", True)]
            if not unsupported_facts:
                continue

            unsupported_texts = set()
            for r in unsupported_facts:
                fact_text = r.get("fact", "")
                if fact_text:
                    unsupported_texts.add(fact_text.strip().lstrip("- ").strip())

            new_lines = []
            removed_in_page = 0
            in_facts = False
            for i, line in enumerate(lines):
                if line.strip() == "## Key Facts":
                    in_facts = True
                    new_lines.append(line)
                    continue
                if in_facts and line.strip().startswith("## "):
                    in_facts = False

                if in_facts and line.strip().startswith("- "):
                    fact_content = line.strip().lstrip("- ").strip()
                    is_unsupported = False
                    for ut in unsupported_texts:
                        if ut in fact_content or fact_content in ut:
                            is_unsupported = True
                            break
                    if is_unsupported:
                        removed_in_page += 1
                        continue  # skip this line (drop it)

                new_lines.append(line)

            if removed_in_page > 0:
                md.write_text("\n".join(new_lines), encoding="utf-8")
                removed_count += removed_in_page
                error_book.append_ledger(
                    issue_type="unsupported_facts",
                    file=f"{dir_name}/{md.stem}",
                    auto_fixed=False,
                    fix_method="llm_verify_source_grounding",
                    note=f"Removed {removed_in_page} unsupported facts",
                    count=removed_in_page,
                )

        except Exception as e:
            continue

    if removed_count > 0:
        print(f"    ✂️ Removed {removed_count} unsupported facts from {sample_size} sampled pages")

    return removed_count


def llm_detect_contradictions(sample_size: int = 10) -> int:
    """Layer 2: sample-detect cross-page contradictions.

    Randomly pick a knowledge page, read each page it references via Related
    Pages, and have the LLM check whether the two contain contradictory
    information (dates, attributes, relations). When a contradiction is found,
    the source digest is treated as ground truth and the wrong side is fixed.

    Args:
      sample_size  number of page pairs to sample per run (default 10).

    Returns:
      Number of contradictions repaired.
    """
    import random

    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return 0

    all_pages_map: dict[str, Path] = {}  # page_name → path
    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            all_pages_map[md.stem] = md
            all_pages_map[f"{dir_name}/{md.stem}"] = md

    if len(all_pages_map) < 5:
        return 0

    page_pairs = []
    candidates = list(set(all_pages_map.values()))
    random.shuffle(candidates)

    for md in candidates[:sample_size * 3]:  # over-sample to ensure enough page pairs
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if "## Related Pages" not in text:
            continue

        in_related = False
        related_links = []
        for line in text.split("\n"):
            if line.strip() == "## Related Pages":
                in_related = True
                continue
            if in_related and line.strip().startswith("## "):
                break
            if in_related:
                for m in re.finditer(r'\[\[([^\]]+)\]\]', line):
                    link = m.group(1)
                    if link in all_pages_map:
                        related_links.append(link)

        if related_links:
            target_link = random.choice(related_links)
            target_path = all_pages_map[target_link]
            if target_path != md:
                page_pairs.append((md, target_path))

        if len(page_pairs) >= sample_size:
            break

    if not page_pairs:
        return 0

    fixed_count = 0

    for page_a, page_b in page_pairs:
        try:
            text_a = page_a.read_text(encoding="utf-8")[:2000]
            text_b = page_b.read_text(encoding="utf-8")[:2000]
        except (OSError, UnicodeDecodeError):
            continue

        sys_prompt = "You are a knowledge consistency expert. Check whether two related Wiki pages contain contradictory information."
        user_prompt = f"""## Page A: {page_a.stem}
{text_a}

## Page B: {page_b.stem}
{text_b}

## Task
Check if these two pages contain any contradictions in:
- Dates (birth, death, events)
- Numerical facts (population, scores, counts)
- Relationships (parent/child, teacher/student, member/group)
- Attributes (nationality, occupation, location)

Output JSON:
{{
  "has_contradiction": true/false,
  "contradictions": [
    {{
      "page": "which page is likely wrong (A or B)",
      "claim": "the contradictory claim",
      "correct_info": "what it should be based on the other page",
      "field": "which Key Fact line to fix"
    }}
  ]
}}

If no contradictions found, return {{"has_contradiction": false, "contradictions": []}}"""

        try:
            result = call_llm_json(sys_prompt, user_prompt,
                                   model=config.LLM_FAST_MODEL, temperature=0.1)

            if not result.get("has_contradiction", False):
                continue

            contradictions = result.get("contradictions", [])
            for c in contradictions:
                wrong_page = c.get("page", "")
                claim = c.get("claim", "")
                correct_info = c.get("correct_info", "")

                if not claim or not correct_info:
                    continue

                target_md = page_a if wrong_page == "A" else page_b
                try:
                    target_text = target_md.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

                if claim in target_text:
                    new_text = target_text.replace(claim, correct_info, 1)
                    if new_text != target_text:
                        target_md.write_text(new_text, encoding="utf-8")
                        fixed_count += 1
                        dir_name = target_md.parent.name
                        error_book.append_ledger(
                            issue_type="cross_page_contradiction",
                            file=f"{dir_name}/{target_md.stem}",
                            auto_fixed=False,
                            fix_method="llm_detect_contradictions",
                            note=f"Fixed contradiction: '{claim}' → '{correct_info}'",
                            count=1,
                        )

        except Exception:
            continue

    if fixed_count > 0:
        print(f"    🔄 Fixed {fixed_count} cross-page contradictions from {len(page_pairs)} page pairs")

    return fixed_count


def llm_fix_structural() -> dict:
    """Run structural LLM repairs and return per-step counts.

    Structural fixes: complete digests, generate summaries, fill in sections,
    create missing pages, populate Related Pages.
    Frequency: triggered every ``_PERIODIC_EVERY`` (30) articles.
    """
    results = {}

    print(f"\n  🤖 LLM Fix (Structural): Incomplete digests...")
    results["digest_complete"] = llm_fix_incomplete_digests()

    print(f"  🤖 LLM Fix (Structural): Missing summaries...")
    results["summary_added"] = llm_fix_missing_summary()

    print(f"  🤖 LLM Fix (Structural): Missing sections...")
    results["sections_added"] = llm_fix_missing_sections()

    print(f"  🤖 LLM Fix (Structural): Broken links (create pages)...")
    results["pages_created"] = llm_fix_broken_links()

    print(f"  🤖 LLM Fix (Structural): Empty Related Pages...")
    results["related_pages_fixed"] = llm_fix_empty_related_pages()

    total = sum(results.values())
    if total > 0:
        parts = [f"{k}={v}" for k, v in results.items() if v > 0]
        print(f"  🤖 Structural LLM fixed {total} issues ({', '.join(parts)})")
    else:
        print(f"  🤖 No structural LLM fixes needed")

    return results


def llm_fix_content() -> dict:
    """Run content-level LLM repairs and return per-step counts.

    Content-level fixes: validate source support for Key Facts (unsupported
    facts) and detect cross-page contradictions.
    Frequency: triggered every ``_CONTENT_FIX_EVERY`` (60) articles. These are
    expensive and change slowly, so they need not fire as often as structural
    repairs.
    """
    results = {}

    print(f"\n  🤖 LLM Fix (Content): Source grounding verification (Unsupported Facts)...")
    results["unsupported_facts_removed"] = llm_verify_source_grounding()

    print(f"  🤖 LLM Fix (Content): Cross-page contradiction detection...")
    results["contradictions_fixed"] = llm_detect_contradictions()

    total = sum(results.values())
    if total > 0:
        parts = [f"{k}={v}" for k, v in results.items() if v > 0]
        print(f"  🤖 Content LLM fixed {total} issues ({', '.join(parts)})")
    else:
        print(f"  🤖 No content LLM fixes needed")

    return results


def llm_fix_all() -> dict:
    """Run every LLM repair (structural + content-level) and return per-step counts.

    Used by ``finalize_wiki`` and other scenarios that require a full pass.
    """
    results = {}
    results.update(llm_fix_structural())
    results.update(llm_fix_content())
    return results


def merge_duplicate_pages():
    """Scan each directory's _index.md and have the LLM detect mergeable duplicate pages.

    Simplified: detection + LLM merge only; no error-book dependency.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return

    page_types = config.get_page_types()
    page_dirs = config.get_page_dirs()

    print("  🔍 Checking for duplicate pages...")

    dir_indices = {}
    for dir_name in page_types:
        dir_path = page_dirs.get(dir_name)
        if not dir_path:
            continue
        idx_path = dir_path / "_index.md"
        if idx_path.exists():
            try:
                content = idx_path.read_text(encoding="utf-8")
                if content.strip():
                    dir_indices[dir_name] = content
            except Exception:
                pass

    if not dir_indices:
        print("    ⏭️ No _index.md to check")
        return

    index_text_parts = []
    for dir_name, content in dir_indices.items():
        index_text_parts.append(f"### {dir_name}/_index.md\n{content}")
    all_index_text = "\n\n".join(index_text_parts)

    detect_sys = """You are a knowledge base deduplication expert. Check the directory indexes (_index.md) and find pages that may be duplicates — i.e., the same entity/concept created under different names.

Criteria:
1. Same person, different names: e.g., "Einstein" and "Albert-Einstein"
2. Same work, different names: e.g., "Relativity" and "Theory-of-Relativity"
3. Translation/spelling variants
4. Note: aliases in parentheses are NOT separate pages

Only flag cases you are very confident about."""

    detect_user = f"""## Directory Index Content
{all_index_text}

## Task
Find pages that are likely the same entity but with different names.

Output strictly in JSON format:
```json
[
  {{"keep": "more_standard_page_name", "merge": "duplicate_page_name", "dir": "directory_name", "reason": "why they are the same"}},
  ...
]
```

If no duplicates found, output: `[]`

Rules:
- keep and merge must be actual page names from [[...]] in the indexes
- Only page names, no directory prefix, no .md suffix
- keep should be the more complete/standard name"""

    try:
        detect_result = call_llm(detect_sys, detect_user,
                                 max_tokens=2000,
                                 model=config.LLM_FAST_MODEL,
                                 temperature=0.1)
    except Exception as e:
        print(f"    ⚠️ Duplicate detection LLM failed: {e}")
        return

    merge_groups = []
    if detect_result and detect_result.strip():
        try:
            merge_groups = json.loads(detect_result.strip())
        except json.JSONDecodeError:
            m = re.search(r'```json\s*(.*?)\s*```', detect_result, re.DOTALL)
            if m:
                try:
                    merge_groups = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            if not merge_groups:
                start = detect_result.find('[')
                end = detect_result.rfind(']')
                if start != -1 and end != -1:
                    try:
                        merge_groups = json.loads(detect_result[start:end + 1])
                    except json.JSONDecodeError:
                        pass

    if not merge_groups:
        print("    ✅ No duplicate pages found")
        return

    print(f"    🔄 Found {len(merge_groups)} potential duplicate groups")

    valid_groups = []
    for group in merge_groups:
        keep_name = group.get("keep", "")
        merge_name = group.get("merge", "")
        dir_name = group.get("dir", "")
        if not keep_name or not merge_name or not dir_name or keep_name == merge_name:
            continue
        dir_path = page_dirs.get(dir_name)
        if not dir_path or not dir_path.exists():
            found = False
            for d_name, d_path in page_dirs.items():
                if not d_path.exists():
                    continue
                if (d_path / f"{keep_name}.md").exists() or (d_path / f"{merge_name}.md").exists():
                    group["dir"] = d_name
                    found = True
                    break
            if not found:
                continue
            dir_path = page_dirs.get(group["dir"])
        keep_path = dir_path / f"{keep_name}.md"
        merge_path = dir_path / f"{merge_name}.md"
        if keep_path.exists() and merge_path.exists():
            valid_groups.append(group)

    if not valid_groups:
        print("    ⚠️ No valid merge groups (files may not exist)")
        return

    merged_count = 0
    for group in valid_groups:
        keep_name = group["keep"]
        merge_name = group["merge"]
        dir_name = group["dir"]
        reason = group.get("reason", "")

        dir_path = page_dirs.get(dir_name)
        keep_path = dir_path / f"{keep_name}.md"
        merge_path = dir_path / f"{merge_name}.md"

        if not keep_path.exists() or not merge_path.exists():
            continue

        try:
            keep_content = keep_path.read_text(encoding="utf-8")
            merge_content = merge_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"    ⚠️ Failed to read pages: {e}")
            continue

        merge_sys = """You are a knowledge base maintenance expert. Merge two duplicate knowledge pages into one.

Merge principles:
- Keep all non-duplicate information
- aliases field must include the merged page's name
- Merge and deduplicate tags
- Merge and deduplicate content, keep structured fact list format
- Merge and deduplicate Related Pages and Related Sources"""

        merge_user = f"""## Keep page: {keep_name}
```
{keep_content}
```

## Merge page: {merge_name}
```
{merge_content}
```

## Merge reason
{reason}

Output the complete merged page content (including frontmatter and body).
Notes:
1. aliases must include "{merge_name}"
2. Title: `# {keep_name}`
3. Output page content directly, no code block wrappers"""

        try:
            merged_result = call_llm(merge_sys, merge_user,
                                     max_tokens=3000,
                                     model=config.LLM_FAST_MODEL,
                                     temperature=0.2)
            if not merged_result.strip():
                continue

            merged_text = re.sub(r'^```\w*\n?', '', merged_result.strip())
            merged_text = re.sub(r'\n?```\s*$', '', merged_text)
            merged_text = _sanitize_frontmatter(merged_text)

            keep_path.write_text(merged_text, encoding="utf-8")
            print(f"    📝 Merged → {dir_name}/{keep_name}")

            merge_full = f"{dir_name}/{merge_name}"
            keep_full = f"{dir_name}/{keep_name}"
            replaced_files = 0
            for d_name, d_path in page_dirs.items():
                if not d_path.exists():
                    continue
                for md_file in d_path.glob("*.md"):
                    try:
                        text = md_file.read_text(encoding="utf-8")
                        original = text
                        text = text.replace(f"[[{merge_full}]]", f"[[{keep_full}]]")
                        text = text.replace(f"[[{merge_name}]]", f"[[{keep_name}]]")
                        if text != original:
                            md_file.write_text(text, encoding="utf-8")
                            replaced_files += 1
                    except Exception:
                        pass
            if replaced_files:
                print(f"    ✏️ Updated links in {replaced_files} files")

            idx_path = dir_path / "_index.md"
            if idx_path.exists():
                try:
                    idx_text = idx_path.read_text(encoding="utf-8")
                    lines = idx_text.split("\n")
                    new_lines = [l for l in lines if f"[[{merge_name}]]" not in l]
                    if len(new_lines) != len(lines):
                        idx_path.write_text("\n".join(new_lines), encoding="utf-8")
                        print(f"    ✏️ Removed [[{merge_name}]] from {dir_name}/_index.md")
                except Exception:
                    pass

            try:
                merge_path.unlink()
                print(f"    🗑️ Deleted duplicate: {dir_name}/{merge_name}")
            except Exception as e:
                print(f"    ⚠️ Failed to delete: {e}")

            merged_count += 1

        except Exception as e:
            print(f"    ⚠️ Merge {keep_name} + {merge_name} failed: {e}")

    if merged_count:
        print(f"    ✅ Merged {merged_count} duplicate page groups")


def detect_alias_overlaps():
    """Detect page pairs in the same directory whose aliases overlap, then repair them automatically.

    Simplified: no WikiGraph dependency; works by direct filesystem scanning.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return

    print("  🔍 Checking for alias overlaps...")

    import yaml as _yaml
    dir_pages: dict[str, list[tuple[str, set, Path]]] = {}  # dir → [(name, aliases_set, path)]

    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        if not dir_path.exists():
            continue
        pages = []
        for md in dir_path.glob("*.md"):
            if md.name == "_index.md":
                continue
            try:
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            aliases = set()
            fm_result = config.split_frontmatter(text)
            if fm_result:
                _, fm_text, _ = fm_result
                am = re.search(r'aliases:\s*\[(.+?)\]', fm_text)
                if am:
                    for a in am.group(1).split(","):
                        a = a.strip().strip("'\"")
                        if a:
                            aliases.add(a)
            aliases.add(md.stem)  # include the primary name as well
            pages.append((md.stem, aliases, md))
        if pages:
            dir_pages[dir_name] = pages

    overlaps = []
    for dir_name, pages in dir_pages.items():
        for i in range(len(pages)):
            for j in range(i + 1, len(pages)):
                common = pages[i][1] & pages[j][1]
                if common:
                    overlaps.append((dir_name, pages[i], pages[j], common))

    if not overlaps:
        print("    ✅ No alias overlaps found")
        return

    print(f"    ⚠️ Found {len(overlaps)} alias overlap pairs")

    fixed_count = 0
    for dir_name, (name_a, aliases_a, path_a), (name_b, aliases_b, path_b), common in overlaps:
        for alias in common:
            if alias == name_a:
                target_path = path_b
            elif alias == name_b:
                target_path = path_a
            else:
                len_a = path_a.stat().st_size if path_a.exists() else 0
                len_b = path_b.stat().st_size if path_b.exists() else 0
                target_path = path_b if len_a >= len_b else path_a

            try:
                text = target_path.read_text(encoding="utf-8")
                fm_result = config.split_frontmatter(text)
                if not fm_result:
                    continue
                _, fm_text, body = fm_result
                am = re.search(r'aliases:\s*\[(.+?)\]', fm_text)
                if not am:
                    continue
                old_aliases = [a.strip().strip("'\"")
                               for a in am.group(1).split(",")]
                new_aliases = [a for a in old_aliases if a != alias]
                if len(new_aliases) == len(old_aliases):
                    continue
                new_aliases_str = ", ".join(new_aliases)
                new_fm = fm_text[:am.start(1)] + new_aliases_str + fm_text[am.end(1):]
                new_text = f"---\n{new_fm}\n---{body}"
                target_path.write_text(new_text, encoding="utf-8")
                fixed_count += 1
                print(f"    🔧 Removed alias '{alias}' from {target_path.parent.name}/{target_path.stem}")
            except Exception:
                pass

    if fixed_count:
        print(f"    ✅ Fixed {fixed_count} alias overlaps")


def generate_overview():
    """Have the LLM produce a knowledge overview placed at the top of index.md.

    Called once after ingestion completes; serves as the "zero-hop" context
    during retrieval.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return

    page_count = _count_knowledge_pages()
    if page_count < 3:
        print("  ⏭️ Too few pages for overview generation")
        return

    purpose = _read_file_safe(config.get_purpose_file(), 2000)
    dir_catalog = config.get_dir_catalog_text()

    index_contents = _get_all_index_content()
    indexes_text = ""
    for path, content in index_contents.items():
        indexes_text += f"\n### {path}\n{content}\n"

    prompt = f"""You are a knowledge base maintenance expert. Generate a comprehensive overview for this knowledge base.

## Research Direction
{purpose}

## Current Directory Overview
{dir_catalog}

## Directory Indexes
{indexes_text}

## Task
Generate a 200-400 word comprehensive overview:
1. Summarize the core knowledge domains and coverage
2. List the most important entity/concept keywords (densely packed for semantic matching)
3. Point out the main threads and topic connections
4. This overview will be used as "zero-hop" context during retrieval

Output only the overview text, no title, no markdown formatting."""

    try:
        result = call_llm("You are a knowledge base maintenance expert.", prompt,
                          max_tokens=1024,
                          model=config.LLM_FAST_MODEL,
                          temperature=0.3)
        result = result.strip()
        if result.startswith("```"):
            result = re.sub(r"^```\w*\n?", "", result)
            result = re.sub(r"\n?```$", "", result)
        overview = result.strip()
    except Exception as e:
        print(f"  ⚠️ Overview generation failed: {e}")
        overview = ""

    if not overview:
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    content = f"# Wiki Directory Overview\n\n"
    content += f"> **Knowledge Overview** (updated {today_str})\n>\n"
    for line in overview.split("\n"):
        content += f"> {line}\n"
    content += f"\n## Directory Catalog\n\n{dir_catalog}\n"

    index_path = wiki_dir / "index.md"
    index_path.write_text(content, encoding="utf-8")
    print(f"  ✅ Knowledge overview generated ({len(overview)} chars)")



def _parse_index_sections(text: str) -> list[tuple[str, list[str]]]:
    """Parse _index.md into [(section_header_line, [item_line, ...]), ...].
    Non-`## ` content (e.g. the file-top `# title` and `> description`) is
    stored under the key ``"__preamble__"``.
    """
    sections: list[tuple[str, list[str]]] = []
    current_header = "__preamble__"
    current_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.rstrip()
        if stripped.startswith("## "):
            sections.append((current_header, current_lines))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_header, current_lines))
    return sections


def _assemble_sections(sections: list[tuple[str, list[str]]]) -> str:
    """Reassemble parsed sections back into text."""
    parts: list[str] = []
    for header, lines in sections:
        if header == "__preamble__":
            parts.append("\n".join(lines))
        else:
            body = "\n".join(lines).rstrip()
            if body:
                parts.append(f"{header}\n{body}")
            else:
                parts.append(header)
    return "\n".join(parts).rstrip() + "\n"


def _extract_entry_name(entry_line: str) -> str:
    """Extract the page name from a ``- [[page-name]] ...`` line."""
    m = re.match(r'\s*-\s*\[\[([^\]|#]+)', entry_line)
    if not m:
        return ""
    name = m.group(1).strip()
    return name.rsplit("/", 1)[-1]


def relocate_pending_entries(dry_run: bool = False, batch_size: int = 40,
                             force_final: bool = False) -> dict:
    """Move entries from each directory's "## Unsorted" section into existing sections (or propose new sections).

    Algorithm:
      1. For every knowledge-page directory (i.e. not ``sources``), gather the
         set of existing ``## ...`` section names and every entry under
         ``## Unsorted``.
      2. Ask the LLM to decide, per entry, whether to move it into an existing
         section, propose a new section, or keep it in "Unsorted".
      3. Rewrite ``_index.md`` according to the LLM's decisions: entries move
         to the end of the chosen section; new sections are inserted right
         before ``## Unsorted``; undecided entries stay in ``## Unsorted``.

    Args:
        force_final: final-pass fallback. With the default ``False``, a directory
            whose total entries are ``<=3`` *and* which has no existing sections is
            skipped (to save LLM calls) and left in "Unsorted" — reasonable during
            periodic maintenance (classify later once more entries accrue). But if
            every pass skips them, such entries stay stuck in "Unsorted" forever.
            The ``finalize_wiki`` step should call this once with
            ``force_final=True`` so these few entries are still classified.
    """
    summary = {"dirs_processed": 0, "moved": 0, "new_sections": 0, "left_pending": 0, "details": []}

    for dir_name, dir_path in config.get_page_dirs().items():
        if dir_name.startswith("sources"):
            continue
        idx_path = dir_path / "_index.md"
        if not idx_path.exists():
            continue

        text = idx_path.read_text(encoding="utf-8")
        sections = _parse_index_sections(text)

        pending_idx = None
        existing_sections: list[str] = []  # excluding "Unsorted" and preamble
        for i, (header, _) in enumerate(sections):
            if header == "__preamble__":
                continue
            name = header[3:].strip()
            if name == "Unsorted":
                pending_idx = i
            else:
                existing_sections.append(name)

        if pending_idx is None:
            continue

        pending_lines = sections[pending_idx][1]
        entries: list[str] = [l for l in pending_lines if l.lstrip().startswith("- [[")]
        other_lines: list[str] = [l for l in pending_lines if not l.lstrip().startswith("- [[")]

        if not entries:
            continue

        summary["dirs_processed"] += 1
        print(f"\n  📂 {dir_name}/: 「## Unsorted」{len(entries)} entries, {len(existing_sections)} existing sections")

        existing_entry_count = sum(
            sum(1 for l in lines if l.lstrip().startswith("- [["))
            for header, lines in sections
            if header not in ("__preamble__",) and header[3:].strip() != "Unsorted"
        )
        total_entries = existing_entry_count + len(entries)

        if total_entries <= 10:
            suggested_range = "2-3"
        elif total_entries <= 25:
            suggested_range = "3-5"
        elif total_entries <= 50:
            suggested_range = "4-6"
        elif total_entries <= 100:
            suggested_range = "5-8"
        elif total_entries <= 200:
            suggested_range = "6-10"
        else:
            suggested_range = "8-12"

        suggested_min = int(suggested_range.split("-")[0])
        suggested_max = int(suggested_range.split("-")[-1])

        if total_entries <= 3 and not force_final:
            if existing_sections:
                target_sec = existing_sections[0]
                print(f"    ⏭️  Too few entries ({total_entries}), moving to「{target_sec}」, skip LLM")
                new_sections_list = []
                for header, sec_lines in sections:
                    if header == "__preamble__":
                        continue
                    sec_name = header[3:].strip()
                    if sec_name == "Unsorted":
                        continue
                    if sec_name == target_sec:
                        new_sections_list.append((header, list(sec_lines) + entries))
                    else:
                        new_sections_list.append((header, sec_lines))
                preamble_lines = next((lines for h, lines in sections if h == "__preamble__"), [])
                new_content = "\n".join(preamble_lines)
                if new_content and not new_content.endswith("\n"):
                    new_content += "\n"
                for header, sec_lines in new_sections_list:
                    new_content += f"\n{header}\n" + "\n".join(sec_lines) + "\n"
                if not dry_run:
                    idx_path.write_text(new_content, encoding="utf-8")
                summary["moved"] += len(entries)
            else:
                print(f"    ⏭️  Too few entries ({total_entries}) and no existing sections, skip LLM")
            continue

        decisions: dict[str, dict] = {}  # name → {"section": str, "is_new": bool}
        for start in range(0, len(entries), batch_size):
            batch = entries[start:start + batch_size]
            prompt_entries = "\n".join(batch)

            if existing_sections:
                cur_count = len(existing_sections)

                if cur_count >= suggested_max:
                    section_stance = "at_limit"
                elif cur_count < suggested_min:
                    section_stance = "under"
                else:
                    section_stance = "ok"

                if section_stance == "at_limit":
                    new_section_rule = (
                        f"Currently {cur_count} sections, at the suggested limit ({suggested_range}). "
                        "**Do NOT create any new sections**. All entries must go into an existing section. "
                        "Match loosely — if an entry is even tangentially related to an existing section, put it there."
                    )
                elif section_stance == "under":
                    new_section_rule = (
                        f"Currently only {cur_count} sections, suggested range is {suggested_range}. "
                        "If entries have clearly different themes, create new sections to reach the suggested range. "
                        "But if entries are highly concentrated, putting them in existing sections is fine."
                    )
                else:
                    remaining = suggested_max - cur_count
                    new_section_rule = (
                        f"Currently {cur_count} sections, within suggested range ({suggested_range}). "
                        f"Can create up to {remaining} more sections. "
                        "**Prefer existing sections** — only create new ones if entries are clearly different from all existing sections."
                    )

                sections_text = "\n".join(f"- {s}" for s in existing_sections)
                sys_prompt = (
                    "You are a knowledge base editor. Decide which section each 'Unsorted' entry belongs to. "
                    "Section names should be short English topic labels (2-4 words). "
                    "Do NOT leave any entry in 'Unsorted' — every entry must go into a section."
                )
                user_prompt = f"""Directory: {dir_name}/ — stores {dir_name}-related knowledge pages

## Existing sections (currently {cur_count}, suggested range {suggested_range})
{sections_text}

## Entries to relocate (format: - [[Name]] — summary #tag)
{prompt_entries}

## Output JSON
{{
  "decisions": [
    {{"name": "page name (from [[]])", "section": "section name", "is_new": false}},
    // section = existing section name → is_new=false
    // section = new section name you suggest → is_new=true
    // Every entry must go into a section, do NOT output "Unsorted"
  ]
}}

Rules:
- Output JSON only, no extra explanation
- Section names: short English topic labels (2-4 words), consistent style with existing sections
- {new_section_rule}
- Do NOT use the directory name "{dir_name}" as a section name — sections must be finer-grained sub-topics
- Do NOT use "Other", "Uncategorized", "Miscellaneous" as section names
- Decide based on entry tags and summary"""
            else:
                sys_prompt = (
                    "You are a knowledge base editor. Create a section structure for these entries. "
                    "Section names should be short English topic labels (2-4 words). "
                    f"Create {suggested_min}-{suggested_max} sections total. "
                    "Do NOT leave any entry in 'Unsorted'."
                )
                user_prompt = f"""Directory: {dir_name}/ — stores {dir_name}-related knowledge pages

## Entries to classify (need to create sections from scratch, {len(entries)} entries)
{prompt_entries}

## Output JSON
{{
  "decisions": [
    {{"name": "page name (from [[]])", "section": "section name", "is_new": true}}
  ]
}}

Rules:
- Output JSON only, no extra explanation
- Section names: short English topic labels (2-4 words), consistent style
- Must create at least 2 different sections
- Total sections strictly between {suggested_min}-{suggested_max}
- Do NOT use "{dir_name}" as a section name
- Do NOT use "Other", "Uncategorized", "Miscellaneous" as section names
- Decide based on entry tags and summary"""

            try:
                result = call_llm_json(sys_prompt, user_prompt, model=config.LLM_FAST_MODEL)
            except Exception as e:
                print(f"    ⚠️ LLM relocate failed (skip batch): {e}")
                continue

            for dec in result.get("decisions", []) or []:
                name = (dec.get("name") or "").strip()
                section = (dec.get("section") or "").strip()
                is_new = bool(dec.get("is_new"))
                if not name or not section:
                    continue
                decisions[name] = {"section": section, "is_new": is_new}

        move_to: dict[str, list[str]] = {}
        new_sections_requested: dict[str, list[str]] = {}
        left_behind: list[str] = []

        for entry in entries:
            name = _extract_entry_name(entry)
            dec = decisions.get(name)
            if not dec:
                left_behind.append(entry)
                continue
            if dec["section"] == "Unsorted":
                left_behind.append(entry)
                continue
            if dec["section"] in ("Other", "Uncategorized", "Miscellaneous", dir_name):
                left_behind.append(entry)
                continue
            target = dec["section"]
            if dec["is_new"] and target not in existing_sections:
                new_sections_requested.setdefault(target, []).append(entry)
            else:
                if target in existing_sections:
                    move_to.setdefault(target, []).append(entry)
                else:
                    new_sections_requested.setdefault(target, []).append(entry)

        final_new_sections: dict[str, list[str]] = {}
        for sec, items in new_sections_requested.items():
            final_new_sections[sec] = items

        moved_count = sum(len(v) for v in move_to.values()) + sum(len(v) for v in final_new_sections.values())
        summary["moved"] += moved_count
        summary["new_sections"] += len(final_new_sections)
        summary["left_pending"] += len(left_behind)
        detail = {
            "dir": dir_name,
            "moved_to_existing": {k: len(v) for k, v in move_to.items()},
            "new_sections": {k: len(v) for k, v in final_new_sections.items()},
            "left_pending": len(left_behind),
        }
        summary["details"].append(detail)
        print(
            f"    → Moved to existing: {sum(len(v) for v in move_to.values())}, "
            f"new sections: {len(final_new_sections)} ({sum(len(v) for v in final_new_sections.values())} entries)"
        )
        if move_to:
            distrib = ", ".join(f"{k}(+{len(v)})" for k, v in sorted(move_to.items(), key=lambda x: -len(x[1])))
            print(f"       Existing: {distrib}")
        if final_new_sections:
            new_s = ", ".join(f"{k}({len(v)})" for k, v in sorted(final_new_sections.items(), key=lambda x: -len(x[1])))
            print(f"       New: {new_s}")

        if dry_run or moved_count == 0:
            continue

        for i, (header, lines) in enumerate(sections):
            if not header.startswith("## "):
                continue
            sec_name = header[3:].strip()
            if sec_name in move_to:
                new_lines = list(lines)
                while new_lines and not new_lines[-1].strip():
                    new_lines.pop()
                new_lines.extend(move_to[sec_name])
                new_lines.append("")
                sections[i] = (header, new_lines)

        insert_before = pending_idx
        for sec, items in final_new_sections.items():
            block = items + [""]
            sections.insert(insert_before, (f"## {sec}", block))
            insert_before += 1
            pending_idx += 1

        if left_behind:
            new_pending_body = other_lines + left_behind
            sections[pending_idx] = ("## Unsorted", new_pending_body)
        else:
            sections.pop(pending_idx)

        new_text = _assemble_sections(sections)
        idx_path.write_text(new_text, encoding="utf-8")
        print(f"    ✅ Updated {idx_path.relative_to(config.WIKI_DIR)}")

    return summary


def consolidate_wiki_bench(total_ingested: int = 0) -> dict:
    """Have the LLM audit the directory structure and propose splits/merges/moves, then execute them.

    Adapted for the benchmark setting:
      - No WikiGraph dependency; works by direct filesystem scanning.
      - Split threshold raised to 100 pages (the benchmark corpora are larger).
      - The LLM is only invoked when a directory exceeds the threshold.

    Returns ``dict`` with ``status`` in {"skipped", "no_changes", "executed"}.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir or not wiki_dir.exists():
        return {"status": "skipped", "reason": "wiki_dir not set"}

    page_types = config.get_page_types()
    dir_stats = {}
    max_count = 0

    for dir_name in page_types:
        dir_path = wiki_dir / dir_name
        if not dir_path.exists():
            dir_stats[dir_name] = {"count": 0, "pages": []}
            continue

        pages = [f for f in dir_path.glob("*.md") if f.name != "_index.md"]
        count = len(pages)
        max_count = max(max_count, count)

        page_names = sorted([f.stem for f in pages])[:30]
        dir_stats[dir_name] = {
            "count": count,
            "pages": page_names,
            "description": page_types[dir_name].get("description", ""),
        }

    if max_count < 100:
        print(f"  📂 Directory audit skipped (max dir size: {max_count} pages, threshold: 100)")
        return {"status": "skipped", "reason": f"max dir {max_count} < 100 threshold"}

    print(f"  📂 Directory structure audit (max dir: {max_count} pages)...")

    dir_text = ""
    for dir_name, stats in sorted(dir_stats.items()):
        dir_text += f"\n### {dir_name}/ ({stats['count']} pages) — {stats.get('description', '')}\n"
        for pname in stats["pages"]:
            dir_text += f"  - {pname}\n"
        if stats["count"] > 30:
            dir_text += f"  - ... and {stats['count'] - 30} more pages\n"

    purpose = ""
    purpose_file = config.get_purpose_file()
    if purpose_file and purpose_file.exists():
        try:
            purpose = purpose_file.read_text(encoding="utf-8")[:2000]
        except (OSError, UnicodeDecodeError):
            pass

    total_pages = sum(s["count"] for s in dir_stats.values())

    prompt = f"""You are a knowledge base architect. Review the directory structure below and determine whether it **truly** needs optimization.

**Key Principle: Be conservative. Prefer no changes.**
Directory structure stability is critical — frequent changes cause index confusion. Only adjust when there are clear structural problems.

## Research Focus
{purpose}

## Current Directory Structure
{dir_text}

## Current Progress
- Total source documents ingested: {total_ingested}
- Total knowledge pages: {total_pages}

## Criteria (only consider changes if these conditions are met)

1. **Split**: A directory has >100 pages AND clearly contains 2+ unrelated sub-topics
2. **Merge**: After ingesting >200 articles, a directory still has only 1-2 pages AND those pages highly overlap with another directory's topic
3. **Migrate**: A page is placed in a clearly wrong directory (completely inconsistent with directory description)
4. **No changes**: If the structure is generally reasonable, each directory has 3+ pages, and there are no obvious misclassifications, **return an empty list**

## Output JSON Format
{{
  "changes": [
    {{
      "action": "split",
      "from": "source_directory",
      "to": "new_directory",
      "description": "english_type_name — one sentence describing the directory's content scope",
      "move_pages": ["page_name1", "page_name2"],
      "reason": "reason for split"
    }},
    {{
      "action": "merge",
      "from": "directory_to_merge",
      "to": "target_directory",
      "reason": "reason for merge"
    }},
    {{
      "action": "move_page",
      "from": "source_directory",
      "to": "target_directory",
      "description": "english_type_name — one sentence describing the directory's content scope",
      "move_pages": ["page_name1"],
      "reason": "reason for migration"
    }}
  ],
  "reasoning": "Overall judgment reason (max 100 chars)"
}}

Notes:
- **In most cases you should return an empty list** — directory structures usually don't need frequent adjustments
- Directory names must be a single lowercase English word
- split only moves the pages listed in move_pages, not all pages
- merge will delete the 'from' directory (all pages moved to 'to'), use with caution
- For split, new directory description format: "english_name — one sentence description" """

    try:
        result = call_llm_json(
            "You are a knowledge base architect. Review the directory structure and output optimization suggestions.",
            prompt,
            model=config.LLM_FAST_MODEL,
        )
        changes = result.get("changes", [])
        reasoning = result.get("reasoning", "")
        if reasoning:
            print(f"  💡 LLM audit: {reasoning[:200]}")

        if not changes:
            return {"status": "no_changes", "reasoning": reasoning}

        _apply_consolidate_changes(changes)
        return {"status": "executed", "changes": changes, "reasoning": reasoning}

    except Exception as e:
        print(f"  ⚠️ Directory audit failed: {e}")
        return {"status": "error", "error": str(e)}


def _apply_consolidate_changes(changes: list[dict]):
    """Execute directory-structure changes (split/merge/move)."""
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return

    for change in changes:
        action = change.get("action", "")
        from_dir = change.get("from", "")
        to_dir = change.get("to", "")
        move_pages = change.get("move_pages", [])
        desc = change.get("description", "")

        if action == "split" and from_dir and to_dir and move_pages:
            config.apply_dir_changes([change])
            new_dir_path = wiki_dir / to_dir
            new_dir_path.mkdir(parents=True, exist_ok=True)

            from_path = wiki_dir / from_dir
            moved = 0
            for page_name in move_pages:
                candidates = list(from_path.glob(f"{page_name}.md"))
                if not candidates:
                    candidates = [f for f in from_path.glob("*.md")
                                  if f.stem.lower() == page_name.lower()]
                for src_file in candidates:
                    dst_file = new_dir_path / src_file.name
                    if not dst_file.exists():
                        import shutil
                        shutil.move(str(src_file), str(dst_file))
                        try:
                            text = dst_file.read_text(encoding="utf-8")
                            text = re.sub(
                                r'^(type:\s*).*$',
                                f'\\1{to_dir}',
                                text,
                                count=1,
                                flags=re.MULTILINE,
                            )
                            dst_file.write_text(text, encoding="utf-8")
                        except (OSError, UnicodeDecodeError):
                            pass
                        moved += 1

            print(f"  📂 Split: {from_dir}/ → {to_dir}/ ({moved} pages moved)")

        elif action == "merge" and from_dir and to_dir:
            from_path = wiki_dir / from_dir
            to_path = wiki_dir / to_dir
            if not from_path.exists() or not to_path.exists():
                continue

            moved = 0
            for src_file in from_path.glob("*.md"):
                if src_file.name == "_index.md":
                    continue
                dst_file = to_path / src_file.name
                if not dst_file.exists():
                    import shutil
                    shutil.move(str(src_file), str(dst_file))
                    try:
                        text = dst_file.read_text(encoding="utf-8")
                        text = re.sub(
                            r'^(type:\s*).*$',
                            f'\\1{to_dir}',
                            text,
                            count=1,
                            flags=re.MULTILINE,
                        )
                        dst_file.write_text(text, encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        pass
                    moved += 1

            print(f"  📂 Merge: {from_dir}/ → {to_dir}/ ({moved} pages merged)")

        elif action == "move_page" and from_dir and to_dir and move_pages:
            config.apply_dir_changes([change])
            to_path = wiki_dir / to_dir
            to_path.mkdir(parents=True, exist_ok=True)
            from_path = wiki_dir / from_dir

            moved = 0
            for page_name in move_pages:
                candidates = list(from_path.glob(f"{page_name}.md"))
                if not candidates:
                    candidates = [f for f in from_path.glob("*.md")
                                  if f.stem.lower() == page_name.lower()]
                for src_file in candidates:
                    dst_file = to_path / src_file.name
                    if not dst_file.exists():
                        import shutil
                        shutil.move(str(src_file), str(dst_file))
                        try:
                            text = dst_file.read_text(encoding="utf-8")
                            text = re.sub(
                                r'^(type:\s*).*$',
                                f'\\1{to_dir}',
                                text,
                                count=1,
                                flags=re.MULTILINE,
                            )
                            dst_file.write_text(text, encoding="utf-8")
                        except (OSError, UnicodeDecodeError):
                            pass
                        moved += 1

            print(f"  📂 Move: {moved} pages from {from_dir}/ → {to_dir}/")

    _update_wiki_references(changes)


def _update_wiki_references(changes: list[dict]):
    """Rewrite ``[[old_dir/page]]`` references across the wiki to ``[[new_dir/page]]``.

    After a page is moved/split/merged, every reference to it elsewhere in the
    wiki must be updated accordingly.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir or not wiki_dir.exists():
        return

    ref_map: dict[str, str] = {}
    for change in changes:
        action = change.get("action", "")
        from_dir = change.get("from", "")
        to_dir = change.get("to", "")
        move_pages = change.get("move_pages", [])

        if action in ("split", "move_page") and from_dir and to_dir and move_pages:
            for page_name in move_pages:
                ref_map[f"{from_dir}/{page_name}"] = f"{to_dir}/{page_name}"
        elif action == "merge" and from_dir and to_dir:
            to_path = wiki_dir / to_dir
            if to_path.exists():
                for md_file in to_path.glob("*.md"):
                    if md_file.name == "_index.md":
                        continue
                    ref_map[f"{from_dir}/{md_file.stem}"] = f"{to_dir}/{md_file.stem}"

    if not ref_map:
        return

    updated_files = 0
    for md_file in sorted(wiki_dir.rglob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        new_text = text
        for old_ref, new_ref in ref_map.items():
            new_text = new_text.replace(f"[[{old_ref}]]", f"[[{new_ref}]]")

        if new_text != text:
            try:
                md_file.write_text(new_text, encoding="utf-8")
                updated_files += 1
            except OSError:
                pass

    if updated_files > 0:
        print(f"  🔗 Updated references in {updated_files} files ({len(ref_map)} page refs remapped)")


def periodic_maintenance(articles_since_last: int, total_ingested: int = 0) -> bool:
    """Periodic LLM maintenance (triggered every N articles).

    Per-article code checks and fixes already run after each ingest. This pass
    only performs:
      1. Structural LLM repairs   (every ``_PERIODIC_EVERY``=30 articles).
      1.5 Content-level LLM repairs (every ``_CONTENT_FIX_EVERY``=60 articles;
          less frequent than structural repairs).
      2. Place unsorted entries  (LLM decides the right section).
      3. Merge duplicate pages.
      4. Alias-conflict detection.
      5. Directory-structure audit (every ``_CONSOLIDATE_EVERY`` articles).
      6. Index rebuild.

    Returns ``True`` if maintenance ran; the caller should then reset the counter.
    """
    if articles_since_last < _PERIODIC_EVERY:
        return False

    print(f"\n{'─'*50}")
    print(f"  🔄 Periodic LLM maintenance (every {_PERIODIC_EVERY} articles)")
    print(f"{'─'*50}")

    # 1. Structural LLM repair (skipped when the error book is disabled).
    if _ENABLE_ERROR_BOOK:
        try:
            llm_results = llm_fix_structural()
        except Exception as e:
            print(f"  ⚠️ Structural LLM fix failed: {e}")
    else:
        print(f"  ⏭️ LLM fix skipped (Error Book disabled)")

    if _ENABLE_ERROR_BOOK and total_ingested > 0 and (total_ingested % _CONTENT_FIX_EVERY) < _PERIODIC_EVERY:
        try:
            print(f"  🔬 Content-level LLM fix (every {_CONTENT_FIX_EVERY} articles)...")
            content_results = llm_fix_content()
        except Exception as e:
            print(f"  ⚠️ Content LLM fix failed: {e}")

    try:
        relocate_result = relocate_pending_entries(dry_run=False)
        if relocate_result["moved"] > 0:
            print(f"  📑 Relocated {relocate_result['moved']} entries from Unsorted")
    except Exception as e:
        print(f"  ⚠️ Relocate pending entries failed: {e}")

    try:
        merge_duplicate_pages()
    except Exception as e:
        print(f"  ⚠️ Merge duplicates failed: {e}")

    try:
        detect_alias_overlaps()
    except Exception as e:
        print(f"  ⚠️ Alias detection failed: {e}")

    current_pages = _count_knowledge_pages()
    if current_pages >= _CONSOLIDATE_EVERY and (current_pages % _CONSOLIDATE_EVERY) < _PERIODIC_EVERY:
        try:
            consolidate_result = consolidate_wiki_bench(total_ingested=total_ingested)
            status = consolidate_result.get("status", "")
            if status == "executed":
                changes = consolidate_result.get("changes", [])
                print(f"  📂 Directory restructured: {len(changes)} changes applied")
        except Exception as e:
            print(f"  ⚠️ Directory consolidation failed: {e}")

    _rebuild_sources_index()
    _rebuild_global_index(update_overview=True)

    print(f"  ✅ Periodic maintenance complete")
    return True


def finalize_wiki():
    """Post-ingestion finalization.

    Runs three lint <-> repair rounds:
      Round 1: lint -> code_fix -> llm_fix -> merge duplicates -> alias check.
      Round 2: lint -> code_fix -> llm_fix (fix issues created by Round 1).
      Round 3: lint -> code_fix -> llm_fix (fix issues created by Round 2; converges).
    Final steps: rebuild indexes -> generate overview -> print the error book.
    """
    print(f"\n{'='*50}")
    print(f"  🏁 Finalization (3-round code+LLM fix loop)")
    print(f"{'='*50}")

    for round_num in range(1, 4):
        print(f"\n  {'─'*40}")
        print(f"  📋 Round {round_num}: Code fix...")
        print(f"  {'─'*40}")

        issues = quick_lint_bench()
        if issues:
            total = sum(len(v) if isinstance(v, list) else 1 for v in issues.values())
            print(f"    Found {total} issues")
            for key, items in issues.items():
                count = len(items) if isinstance(items, list) else 1
                print(f"      {key}: {count}")
            fixes = auto_fix_bench()
            total_fixed = sum(fixes.values())
            if total_fixed > 0:
                parts = [f"{k}={v}" for k, v in fixes.items() if v > 0]
                print(f"    🔧 Code fixed {total_fixed} ({', '.join(parts)})")
        else:
            print(f"    ✅ No issues")

        print(f"\n  📋 Round {round_num}: LLM fix...")
        if _ENABLE_ERROR_BOOK:
            try:
                llm_results = llm_fix_all()
            except Exception as e:
                print(f"    ⚠️ LLM fix failed: {e}")
        else:
            print(f"    ⏭️ LLM fix skipped (Error Book disabled)")

        if round_num == 1:
            print(f"\n  📋 Merge duplicates...")
            try:
                merge_duplicate_pages()
            except Exception as e:
                print(f"    ⚠️ Merge failed: {e}")

            print(f"\n  📋 Alias overlap detection...")
            try:
                detect_alias_overlaps()
            except Exception as e:
                print(f"    ⚠️ Alias detection failed: {e}")

    print(f"\n  📋 Directory structure audit...")
    try:
        page_count_before = _count_knowledge_pages()
        consolidate_result = consolidate_wiki_bench(total_ingested=page_count_before)
        status = consolidate_result.get("status", "")
        if status == "executed":
            changes = consolidate_result.get("changes", [])
            print(f"    📂 Restructured: {len(changes)} changes applied")
        elif status == "no_changes":
            print(f"    ✅ Directory structure is reasonable")
        else:
            reason = consolidate_result.get("reason", status)
            print(f"    ℹ️ {reason}")
    except Exception as e:
        print(f"    ⚠️ Directory audit failed: {e}")

    # Final forced relocation of leftover "## Unsorted" entries (including
    # directories with <=3 entries and no existing sections, which periodic
    # maintenance skips; force_final=True classifies them so they do not stay
    # stuck in Unsorted forever).
    print(f"\n  📋 Final Unsorted relocation (force)...")
    try:
        final_relocate = relocate_pending_entries(dry_run=False, force_final=True)
        if final_relocate["moved"] > 0:
            print(f"    📑 Force-relocated {final_relocate['moved']} entries from Unsorted")
    except Exception as e:
        print(f"    ⚠️ Final relocation failed: {e}")

    print(f"\n  📋 Rebuild indexes...")
    _rebuild_sources_index()
    _rebuild_global_index(update_overview=True)

    print(f"\n  📋 Final lint check...")
    final_issues = quick_lint_bench()
    if final_issues:
        total = sum(len(v) if isinstance(v, list) else 1 for v in final_issues.values())
        print(f"    ⚠️ {total} remaining issues:")
        for key, items in final_issues.items():
            count = len(items) if isinstance(items, list) else 1
            print(f"      {key}: {count}")
    else:
        print(f"    ✅ Zero issues 🎉")

    # Print the error-book summary (skipped when the error book is disabled).
    if _ENABLE_ERROR_BOOK:
        print(f"\n  📋 Error book summary...")
        try:
            error_book.print_error_book()
        except Exception as e:
            print(f"    ⚠️ Error book print failed: {e}")
    else:
        print(f"\n  📋 Error book disabled (ablation mode)")

    page_count = _count_knowledge_pages()
    print(f"\n  🏁 Finalization complete | {page_count} knowledge pages")



def parse_file_outputs(text: str) -> dict[str, str]:
    """Parse ``---FILE: path---`` blocks (logic reused from the ingest engine)."""
    dir_changes_idx = text.find("---DIR_CHANGES---")
    if dir_changes_idx != -1:
        text = text[:dir_changes_idx]

    pattern = r'---FILE:\s*(.+?)---\s*\n'
    parts = re.split(pattern, text)

    files = {}
    i = 1
    while i < len(parts) - 1:
        path = parts[i].strip()
        content = parts[i + 1].strip()
        if path and content:
            if not path.startswith("wiki/"):
                path = "wiki/" + path
            files[path] = content
        i += 2

    return files


def parse_dir_changes(text: str) -> list[dict]:
    """Parse ``---DIR_CHANGES---`` blocks."""
    idx = text.find("---DIR_CHANGES---")
    if idx == -1:
        return []
    changes_text = text[idx + len("---DIR_CHANGES---"):].strip()
    try:
        return json.loads(changes_text)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', changes_text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []



def write_wiki_files(file_outputs: dict[str, str], selected_pages: list[str] = None,
                     batch_digest_stems: list[str] | None = None):
    """Write LLM-generated files into the wiki directory.

    Includes automatic validation:
      1. Filename normalization: full-width -> half-width, whitespace, hyphens.
      2. Type/path correction: when frontmatter ``type`` disagrees with the
         current directory, ``type`` wins (the file is moved).
      3. Cross-directory dedup: when a same-named file already exists in
         another directory, merge into the existing one.
      4. _index.md upkeep: append newly written knowledge pages to the
         appropriate directory's "Unsorted" section.
      5. Frontmatter truncation detection: skip incomplete files.
      6. Sources path correction: wiki/sources/xxx.md -> wiki/sources/digests/xxx.md.
      7. Auto-correct easily confused directory names.
      8. Unseen-page protection: existing pages whose content was not shown to
         the LLM may not be overwritten.
      9. Wikilink normalization.
      10. Frontmatter cleanup.
      11. created/updated date injection.
      12. Digest completeness validation.
      13. Related Sources injection: ensure each knowledge page links to the
          digests of the current batch.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        print("  ❌ WIKI_DIR not set")
        return 0

    known_dirs = set(config.get_page_types().keys()) | set(config.FIXED_DIRS.keys())

    _fixed_dir_basenames = set()
    for fd in config.FIXED_DIRS:
        base = fd.split("/")[0]
        _fixed_dir_basenames.add(base)
    _confusable_map: dict[str, str] = {}
    for base in _fixed_dir_basenames:
        if base.endswith("s"):
            _confusable_map[base[:-1]] = base
            _confusable_map[base[:-2]] = base
            _confusable_map[base + "s"] = base
        else:
            _confusable_map[base + "s"] = base
            _confusable_map[base + "es"] = base

    def _is_sources_path(path: str) -> bool:
        return path.startswith("wiki/sources/")

    normalized_outputs = {}
    _filename_norm_map: dict[str, str] = {}
    for rel_path, content in file_outputs.items():
        if rel_path.startswith("wiki/"):
            if re.match(r'^wiki/sources/[^/]+\.md$', rel_path) and not rel_path.endswith("_index.md"):
                old_path = rel_path
                filename = rel_path.split("/")[-1]
                rel_path = f"wiki/sources/digests/{filename}"
                print(f"  🔧 sources path fix: {old_path} → {rel_path}")

            parts_check = rel_path.split("/")
            if len(parts_check) >= 3 and not rel_path.endswith("_index.md"):
                dir_name = parts_check[1]
                if dir_name in _confusable_map and dir_name not in known_dirs:
                    correct_dir = _confusable_map[dir_name]
                    old_path = rel_path
                    if correct_dir == "sources":
                        filename = parts_check[-1]
                        rel_path = f"wiki/sources/digests/{filename}"
                    else:
                        parts_check[1] = correct_dir
                        rel_path = "/".join(parts_check)
                    print(f"  🔧 confusable dir fix: {old_path} → {rel_path}")

            parts = rel_path.split("/")
            if not rel_path.endswith("_index.md"):
                old_name = parts[-1]
                new_name = _normalize_filename(old_name)
                if new_name != old_name:
                    parts[-1] = new_name
                    new_path = "/".join(parts)
                    print(f"  🔤 filename normalized: {old_name} → {new_name}")
                    old_link = rel_path[len("wiki/"):].removesuffix(".md")
                    new_link = new_path[len("wiki/"):].removesuffix(".md")
                    _filename_norm_map[old_link] = new_link
                    rel_path = new_path
        normalized_outputs[rel_path] = content
    file_outputs = normalized_outputs

    if _filename_norm_map:
        synced_outputs = {}
        for rel_path, content in file_outputs.items():
            for old_link, new_link in _filename_norm_map.items():
                if f"[[{old_link}]]" in content:
                    content = content.replace(f"[[{old_link}]]", f"[[{new_link}]]")
            synced_outputs[rel_path] = content
        file_outputs = synced_outputs
        print(f"  🔗 Synced {len(_filename_norm_map)} wikilinks after filename normalization")

    def _normalize_wikilink_targets(content: str) -> str:
        def _norm_link(m):
            link = m.group(1)
            if "/" in link:
                prefix, name = link.rsplit("/", 1)
                normed = _normalize_filename(name).rstrip('.').rstrip(',')
                new_link = f"{prefix}/{normed}"
            else:
                new_link = _normalize_filename(link)
            return f"[[{new_link}]]" if new_link != link else m.group(0)
        return re.sub(r'\[\[([^\]]+)\]\]', _norm_link, content)

    wikilink_norm_count = 0
    normed_outputs = {}
    for rel_path, content in file_outputs.items():
        new_content = _normalize_wikilink_targets(content)
        if new_content != content:
            wikilink_norm_count += 1
        normed_outputs[rel_path] = new_content
    file_outputs = normed_outputs
    if wikilink_norm_count:
        print(f"  🔗 Global wikilink normalization: {wikilink_norm_count} files")

    corrected_paths = {}
    for rel_path, content in file_outputs.items():
        if not rel_path.startswith("wiki/") or _is_sources_path(rel_path) or rel_path.endswith("_index.md"):
            continue
        path_parts = rel_path.split("/")
        if len(path_parts) < 3:
            continue
        path_dir = path_parts[1]
        page_type = _extract_type_from_content(content)
        if page_type and page_type != path_dir and page_type in known_dirs:
            corrected_paths[rel_path] = f"wiki/{page_type}/{path_parts[-1]}"

    _selected_names: set[str] | None = None
    if selected_pages is not None:
        _selected_names = set()
        for sp in selected_pages:
            name = sp.strip("[]")  # strip the [[...]] double brackets
            name = name.rsplit("/", 1)[-1] if "/" in name else name
            name = name.removesuffix(".md")
            _selected_names.add(name)

    _new_knowledge_pages: list[tuple[str, str]] = []

    written = 0
    sorted_items = sorted(file_outputs.items(), key=lambda kv: kv[0].endswith("_index.md"))
    for rel_path, content in sorted_items:
        if rel_path.endswith("log.md") or rel_path == "wiki/index.md":
            continue
        if rel_path in ("wiki/sources/_index.md", "wiki/sources/digests/_index.md", "wiki/sources/articles/_index.md"):
            print(f"  ⏭️ Skipped {rel_path} (auto-generated by code)")
            continue
        if ".." in rel_path:
            print(f"  ⚠️ Skipped unsafe path: {rel_path}")
            continue

        if not _check_frontmatter_complete(content, rel_path):
            print(f"  ⏭️ Skipped {rel_path} (incomplete content)")
            continue

        if (_selected_names is not None
                and rel_path.startswith("wiki/")
                and not rel_path.endswith("_index.md")
                and not _is_sources_path(rel_path)):
            page_name = rel_path.split("/")[-1].removesuffix(".md")
            actual_path = wiki_dir / rel_path[len("wiki/"):]
            if actual_path.exists() and page_name not in _selected_names:
                print(f"  🛡️ Skipped {rel_path} (existing page not selected in Step1, would lose data)")
                continue

        if rel_path.startswith("wiki/sources/digests/") and not rel_path.endswith("_index.md"):
            content = _check_digest_completeness(content, rel_path)

        if rel_path.startswith("wiki/") and not _is_sources_path(rel_path) and rel_path.endswith("_index.md"):
            print(f"  ⏭️ Skipped {rel_path} (_index.md auto-maintained by code)")
            continue

        _type_corrected = False
        if rel_path in corrected_paths:
            old_path = rel_path
            rel_path = corrected_paths[old_path]
            _type_corrected = True
            print(f"  🔧 Type path fix: {old_path} → {rel_path}")

        if rel_path.startswith("wiki/") and not rel_path.endswith("_index.md") and not _is_sources_path(rel_path):
            path_parts = rel_path.split("/")
            if len(path_parts) == 3:
                filename = path_parts[2]
                target_dir_path = wiki_dir / path_parts[1]
                if target_dir_path.exists():
                    norm_current = filename.lower().replace(' ', '-').replace('_', '-')
                    for existing_file in target_dir_path.iterdir():
                        if existing_file.name == filename or not existing_file.name.endswith('.md'):
                            continue
                        if existing_file.name == '_index.md':
                            continue
                        norm_existing = existing_file.name.lower().replace(' ', '-').replace('_', '-')
                        if norm_existing == norm_current:
                            old_rel = rel_path
                            rel_path = f"wiki/{path_parts[1]}/{existing_file.name}"
                            print(f"  🔀 Same-dir dedup: {filename} → {existing_file.name} (normalized match)")
                            break

        if rel_path.startswith("wiki/") and not rel_path.endswith("_index.md") and not _is_sources_path(rel_path):
            path_parts = rel_path.split("/")
            if len(path_parts) == 3:
                filename = path_parts[2]
                target_dir = path_parts[1]
                if wiki_dir.exists():
                    for d in wiki_dir.iterdir():
                        if d.is_dir() and d.name != target_dir and d.name != "sources":
                            existing = d / filename
                            if existing.exists():
                                if _type_corrected:
                                    try:
                                        existing.unlink()
                                        print(f"  🔀 Type fix move: deleted old {d.name}/{filename}, writing to {target_dir}/")
                                    except OSError as e:
                                        print(f"  ⚠️ Failed to delete old file {d.name}/{filename}: {e}")
                                else:
                                    print(f"  🔀 Cross-dir dedup: {filename} exists in {d.name}/, writing there")
                                    rel_path = f"wiki/{d.name}/{filename}"
                                break

        if rel_path.startswith("wiki/") and not _is_sources_path(rel_path) and not rel_path.endswith("_index.md"):
            final_parts = rel_path.split("/")
            if len(final_parts) >= 3:
                actual_dir = final_parts[1]
                _new_knowledge_pages.append((rel_path, actual_dir))

        if rel_path.startswith("wiki/"):
            actual_rel = rel_path[len("wiki/"):]
            full_path = wiki_dir / actual_rel
        else:
            full_path = Path(rel_path)

        full_path.parent.mkdir(parents=True, exist_ok=True)

        if not rel_path.endswith("_index.md"):
            content = _inject_dates(content, full_path.exists(), existing_path=full_path)

        content = _sanitize_frontmatter(content)

        full_path.write_text(content, encoding="utf-8")
        written += 1
        print(f"  📄 Written {rel_path} ({len(content)} chars)")

    if _new_knowledge_pages:
        registered_dirs = set(config.get_page_types().keys())
        for _, actual_dir in _new_knowledge_pages:
            if actual_dir not in registered_dirs and actual_dir not in ("sources", "syntheses"):
                if actual_dir in _confusable_map:
                    print(f"  ⚠️ Refused to register confusable dir '{actual_dir}' (similar to '{_confusable_map[actual_dir]}')")
                    continue
                config.register_page_type(actual_dir, f"Auto-registered — {actual_dir}", auto_created=True)
                registered_dirs.add(actual_dir)

    _rebuild_sources_index()
    _rebuild_global_index()

    if _new_knowledge_pages:
        for rel_path, actual_dir in _new_knowledge_pages:
            page_name = rel_path.split("/")[-1].removesuffix(".md")
            page_content = file_outputs.get(rel_path, "")
            if not page_content:
                for orig_path, orig_content in file_outputs.items():
                    if orig_path.split("/")[-1].removesuffix(".md") == page_name:
                        page_content = orig_content
                        break
            _append_to_index(actual_dir, page_name, page_content)

    if batch_digest_stems and _new_knowledge_pages:
        inject_count = 0
        for rel_path, actual_dir in _new_knowledge_pages:
            full_path = wiki_dir / rel_path[len("wiki/"):]
            if not full_path.exists():
                continue
            try:
                kp_text = full_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            modified = False
            for stem in batch_digest_stems:
                digest_link = f"[[sources/digests/{stem}]]"
                if digest_link in kp_text:
                    continue  # already present, skip
                if "## Related Sources" in kp_text:
                    lines = kp_text.split("\n")
                    insert_idx = None
                    in_rs = False
                    for i, line in enumerate(lines):
                        if line.strip() == "## Related Sources":
                            in_rs = True
                            continue
                        if in_rs:
                            if line.strip().startswith("## "):
                                insert_idx = i
                                break
                    if insert_idx is None:
                        kp_text = kp_text.rstrip() + f"\n- {digest_link}\n"
                    else:
                        lines.insert(insert_idx, f"- {digest_link}")
                        if insert_idx + 1 < len(lines) and lines[insert_idx + 1].strip().startswith("## "):
                            lines.insert(insert_idx + 1, "")
                        kp_text = "\n".join(lines)
                else:
                    kp_text = kp_text.rstrip() + f"\n\n## Related Sources\n- {digest_link}\n"
                modified = True

            if modified:
                full_path.write_text(kp_text, encoding="utf-8")
                inject_count += 1

        if inject_count:
            print(f"  🔗 Auto-injected Related Sources links into {inject_count} knowledge pages")

    return written


def _append_to_index(dir_name: str, page_name: str, content: str):
    """Append a new page to the "Unsorted" section of _index.md.

    Includes cross-directory validation: only append pages that actually live
    in the target directory.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return

    page_path = wiki_dir / dir_name / f"{page_name}.md"
    if not page_path.exists():
        print(f"  ⚠️ Skip cross-dir ref: {page_name} not found in {dir_name}/")
        return

    idx_path = wiki_dir / dir_name / "_index.md"

    aliases = ""
    summary = ""
    tags = ""
    result = config.split_frontmatter(content)
    if result:
        _, fm, body = result
        am = re.search(r'aliases:\s*\[(.+?)\]', fm)
        if am:
            aliases = am.group(1).strip()
        tm = re.search(r'tags:\s*\[(.+?)\]', fm)
        if tm:
            tag_list = [t.strip().strip("'\"") for t in tm.group(1).split(",")]
            tags = " ".join(f"#{t}" for t in tag_list if t)
    else:
        body = content

    for line in body.strip().split("\n"):
        if line.startswith("> ") and not line.startswith("> Source"):
            summary = line[2:].strip()
            break

    alias_part = f" ({aliases})" if aliases else ""
    summary_part = f" — {summary}" if summary else ""
    tags_part = f" {tags}" if tags else ""
    entry = f"- [[{page_name}]]{alias_part}{summary_part}{tags_part}"

    if not idx_path.exists():
        pt = config.get_page_types()
        type_info = pt.get(dir_name, {})
        description = type_info.get("description", dir_name)
        idx_path.write_text(f"# {dir_name}\n> {description}\n\n## Unsorted\n{entry}\n", encoding="utf-8")
    else:
        idx_text = idx_path.read_text(encoding="utf-8")
        if f"[[{page_name}]]" in idx_text:
            return
        if "## Unsorted" in idx_text:
            idx_text = idx_text.replace("## Unsorted", f"## Unsorted\n{entry}")
        else:
            idx_text = idx_text.rstrip() + f"\n\n## Unsorted\n{entry}"
        idx_path.write_text(idx_text, encoding="utf-8")


# ─── Dedup cache ───

def load_cache() -> dict:
    cache_file = config.CACHE_FILE
    if cache_file and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache: dict):
    cache_file = config.CACHE_FILE
    if cache_file:
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")



def ingest_single(article_path: Path, cache: dict, force: bool = False) -> bool:
    """Ingest a single article."""
    try:
        text = article_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"  ❌ Failed to read {article_path}: {e}")
        return False

    sha = hashlib.sha256(text.encode()).hexdigest()
    if not force and sha in cache:
        return False

    result = config.split_frontmatter(text)
    if result:
        _, fm, body = result
        title_match = re.search(r'title:\s*["\']?(.+?)["\']?\s*$', fm, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else article_path.stem
    else:
        title = article_path.stem
        body = text

    content = body.strip()
    # All bench articles may contain answers; never auto-skip short ones.
    # if len(content) < config.INGEST_MIN_CONTENT_LEN:
    #     print(f"  ⏭️ Skipped (too short): {article_path.name}")
    #     cache[sha] = {"file": article_path.name, "status": "skipped_short"}
    #     return False

    print(f"  🔍 Ingesting [{title[:50]}] ...")

    t1 = time.time()
    sel_sys, sel_user = build_select_pages_prompt(title, content)
    try:
        sel_result = call_llm_json(sel_sys, sel_user, model=config.LLM_STEP1_MODEL, temperature=0.1)
    except Exception as e:
        print(f"  ❌ Step1 LLM failed: {e}")
        return False

    # All bench articles may contain answers; LLM-driven skip is disabled.
    # if sel_result and sel_result.get("skip"):
    #     t_step1 = time.time() - t1
    #     print(f"  🚫 Skipped ({sel_result.get('skip_reason', 'no reason')}) ({t_step1:.1f}s)")
    #     cache[sha] = {"file": article_path.name, "status": "skipped", "reason": sel_result.get("skip_reason")}
    #     save_cache(cache)
    #     return False

    selected_pages = sel_result.get("pages_to_view", []) if sel_result else []
    MAX_SELECTED_PAGES = 15
    if len(selected_pages) > MAX_SELECTED_PAGES:
        print(f"  ⚠️ LLM selected {len(selected_pages)} pages, truncating to {MAX_SELECTED_PAGES}")
        selected_pages = selected_pages[:MAX_SELECTED_PAGES]
    t_step1 = time.time() - t1
    print(f"  🔍 Selected {len(selected_pages)} pages: {selected_pages[:5]}{'...' if len(selected_pages) > 5 else ''} ({t_step1:.1f}s)")

    existing_content = _read_selected_pages(selected_pages)

    # Brief pause between Step 1 and Step 2 to ease back-pressure.
    time.sleep(5)

    t2 = time.time()
    gen_sys, gen_user = build_ingest_prompt(
        title, "", content,
        existing_pages_content=existing_content,
        selected_pages=selected_pages,
    )
    try:
        response = call_llm(gen_sys, gen_user,
                            model=config.LLM_STEP2_MODEL,
                            temperature=config.LLM_STEP2_TEMPERATURE)
    except Exception as e:
        print(f"  ❌ Step2 LLM failed: {e}")
        return False

    t_step2 = time.time() - t2

    if not response:
        print(f"  ❌ Empty response from LLM")
        return False

    file_outputs = parse_file_outputs(response)
    dir_changes = parse_dir_changes(response)

    if not file_outputs:
        print(f"  ⚠️ No files in LLM output")
        cache[sha] = {"file": article_path.name, "status": "no_output"}
        save_cache(cache)
        return False

    article_stem = _save_article_original(title, content, article_path)

    batch_digest_stems = [_predict_article_stem(title)]

    written = write_wiki_files(file_outputs, selected_pages, batch_digest_stems=batch_digest_stems)

    if article_stem:
        _inject_article_link_to_digests([article_stem])

    _fix_digest_article_links()

    if dir_changes:
        config.apply_dir_changes(dir_changes)

    cache[sha] = {"file": article_path.name, "status": "ingested", "pages": list(file_outputs.keys())}
    save_cache(cache)

    print(f"  ✅ Done: wrote {written} files (select {t_step1:.1f}s + generate {t_step2:.1f}s)")
    return True


def _read_selected_pages(selected_pages: list[str], max_total_chars: int = 100000) -> str:
    """Load the full text of selected pages, with a total-size safety valve.

    Args:
        selected_pages: list of page names selected by the LLM.
        max_total_chars: cap on total characters across all loaded pages;
            once exceeded, only filenames (not content) are returned.
    """
    wiki_dir = config.WIKI_DIR
    if not wiki_dir or not selected_pages:
        return "(no existing pages selected)"

    contents = []
    total_chars = 0
    for page_name in selected_pages:
        page_name = page_name.strip("[]")  # strip the [[...]] double brackets
        found = False
        for dir_name, dir_path in config.get_page_dirs().items():
            if dir_name.startswith("sources"):
                continue
            page_path = dir_path / f"{page_name}.md"
            if page_path.exists():
                try:
                    text = page_path.read_text(encoding="utf-8")
                    entry = f"### {dir_name}/{page_name}.md\n{text}"
                    total_chars += len(entry)
                    if total_chars > max_total_chars:
                        contents.append(f"### {dir_name}/{page_name}.md\n(content truncated due to length limit)")
                    else:
                        contents.append(entry)
                    found = True
                    break
                except (OSError, UnicodeDecodeError):
                    continue
        if not found:
            if "/" in page_name:
                page_path = wiki_dir / f"{page_name}.md"
                if page_path.exists():
                    try:
                        text = page_path.read_text(encoding="utf-8")
                        entry = f"### {page_name}.md\n{text}"
                        total_chars += len(entry)
                        if total_chars > max_total_chars:
                            contents.append(f"### {page_name}.md\n(content truncated due to length limit)")
                        else:
                            contents.append(entry)
                    except (OSError, UnicodeDecodeError):
                        pass

    if total_chars > max_total_chars:
        print(f"  ⚠️ Selected pages content {total_chars:,} chars exceeds limit {max_total_chars:,}, some pages truncated")

    return "\n\n".join(contents) if contents else "(no existing pages found)"


def _save_article_original(title: str, content: str, source_path: Path) -> str | None:
    """Save the original article into ``sources/articles/`` and return its stem."""
    wiki_dir = config.WIKI_DIR
    if not wiki_dir:
        return None

    articles_dir = wiki_dir / "sources" / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    stem = _predict_article_stem(title)
    filepath = articles_dir / f"{stem}.md"

    if not filepath.exists():
        escaped_title = title.replace('"', '\\"')
        md = f"""---
type: source
source_title: "{escaped_title}"
---

# {title}

{content}
"""
        lines = md.split('\n')
        h1_indices = [i for i, l in enumerate(lines) if l.startswith('# ')]
        if len(h1_indices) >= 2 and lines[h1_indices[0]].strip() == lines[h1_indices[1]].strip():
            remove_idx = h1_indices[1]
            lines.pop(remove_idx)
            if remove_idx < len(lines) and lines[remove_idx].strip() == '':
                lines.pop(remove_idx)
            md = '\n'.join(lines)

        filepath.write_text(md, encoding="utf-8")

    return stem



def _ingest_batch_one(batch_paths: list[Path], cache: dict, force: bool = False) -> dict:
    """Process a batch of articles: combine Step 1 (page selection) and Step 2 (generation).

    Combines multiple articles into a single LLM call to reduce the number of
    API requests.

    Returns:
        {"success": int, "failed": int}
    """
    articles = []  # list of dict: {title, content, sha, path}
    for p in batch_paths:
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"  ❌ Failed to read {p}: {e}")
            continue

        sha = hashlib.sha256(text.encode()).hexdigest()

        result = config.split_frontmatter(text)
        if result:
            _, fm, body = result
            title_match = re.search(r'title:\s*["\']?(.+?)["\']?\s*$', fm, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else p.stem
        else:
            title = p.stem
            body = text

        content = body.strip()
        articles.append({
            "title": title,
            "content": content,
            "sha": sha,
            "path": p,
        })

    if not articles:
        return {"success": 0, "failed": len(batch_paths)}

    titles_str = ", ".join(a["title"][:30] for a in articles)
    print(f"  📦 Batch mode: {len(articles)} articles [{titles_str}]")

    t1 = time.time()
    try:
        sel_result = call_llm_json(
            *build_select_pages_batch_prompt(articles),
            model=config.LLM_STEP1_MODEL, temperature=0.1
        )
    except Exception as e:
        print(f"  ❌ Step1 (batch select) failed: {e}")
        return {"success": 0, "failed": len(articles)}

    selected_pages = sel_result.get("pages_to_view", []) if sel_result else []
    MAX_SELECTED_PAGES = 15
    if len(selected_pages) > MAX_SELECTED_PAGES:
        print(f"  ⚠️ LLM selected {len(selected_pages)} pages, truncating to {MAX_SELECTED_PAGES}")
        selected_pages = selected_pages[:MAX_SELECTED_PAGES]
    t_step1 = time.time() - t1
    print(f"  🔍 Selected {len(selected_pages)} pages: {selected_pages[:5]}{'...' if len(selected_pages) > 5 else ''} ({t_step1:.1f}s)")

    existing_content = _read_selected_pages(selected_pages)

    # Brief pause between Step 1 and Step 2 to ease back-pressure.
    time.sleep(5)

    t2 = time.time()
    gen_sys, gen_user = build_ingest_prompt_batch(
        articles,
        existing_pages_content=existing_content,
        selected_pages=selected_pages,
    )
    try:
        response = call_llm(gen_sys, gen_user,
                            model=config.LLM_STEP2_MODEL,
                            temperature=config.LLM_STEP2_TEMPERATURE)
    except Exception as e:
        print(f"  ❌ Step2 (batch generate) failed: {e}")
        return {"success": 0, "failed": len(articles)}

    t_step2 = time.time() - t2

    if not response:
        print(f"  ❌ Empty response from LLM")
        return {"success": 0, "failed": len(articles)}

    file_outputs = parse_file_outputs(response)
    dir_changes = parse_dir_changes(response)

    if not file_outputs:
        print(f"  ⚠️ No files in LLM output")
        for art in articles:
            cache[art["sha"]] = {"file": art["path"].name, "status": "no_output"}
        save_cache(cache)
        return {"success": 0, "failed": len(articles)}

    batch_digest_stems = []
    for art in articles:
        stem = _save_article_original(art["title"], art["content"], art["path"])
        if stem:
            batch_digest_stems.append(stem)

    written = write_wiki_files(file_outputs, selected_pages, batch_digest_stems=None)

    if batch_digest_stems:
        _inject_article_link_to_digests(batch_digest_stems)

    _fix_digest_article_links()

    if dir_changes:
        config.apply_dir_changes(dir_changes)

    for art in articles:
        cache[art["sha"]] = {"file": art["path"].name, "status": "ingested", "pages": list(file_outputs.keys())}
    save_cache(cache)

    print(f"  ✅ Batch done: wrote {written} files for {len(articles)} articles (select {t_step1:.1f}s + generate {t_step2:.1f}s)")
    return {"success": len(articles), "failed": 0}


def ingest_batch(article_paths: list[Path], batch_size: int = 5,
                 limit: int = None, force: bool = False):
    """Ingest articles in batches (with periodic maintenance and final repair)."""
    cache = load_cache()

    to_process = []
    skipped = 0
    for p in article_paths:
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        sha = hashlib.sha256(text.encode()).hexdigest()
        if not force and sha in cache:
            skipped += 1
            continue
        to_process.append(p)

    if not to_process:
        print(f"All articles already ingested ({skipped} skipped)")
        return

    total = len(to_process)
    print(f"📋 Batch ingestion | Total: {total} | Batch size: {batch_size} | Skipped: {skipped}")

    success = 0
    failed = 0
    articles_since_last_periodic = 0
    t_total = time.time()
    processed_count = 0  # processed-article counter (used by the progress bar)

    def _format_time(seconds: float) -> str:
        """Format a duration in seconds as HH:MM:SS or MM:SS."""
        seconds = int(seconds)
        if seconds >= 3600:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            return f"{h:d}h{m:02d}m{s:02d}s"
        elif seconds >= 60:
            m = seconds // 60
            s = seconds % 60
            return f"{m:d}m{s:02d}s"
        else:
            return f"{seconds:d}s"

    def _print_progress_bar(current: int, total_items: int, elapsed: float,
                            bar_width: int = 30):
        """Print a progress bar with percentage, elapsed time and ETA."""
        percent = current / total_items if total_items > 0 else 0
        filled = int(bar_width * percent)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed_str = _format_time(elapsed)
        if current > 0:
            eta = elapsed * (total_items - current) / current
            eta_str = _format_time(eta)
        else:
            eta_str = "..."

        avg_per_item = elapsed / current if current > 0 else 0

        line = (f"\r  ⏳ [{bar}] {current}/{total_items} "
                f"({percent*100:.1f}%) | "
                f"elapsed: {elapsed_str} | ETA: {eta_str} | "
                f"avg: {avg_per_item:.1f}s/article")
        print(line, end="", flush=True)

    for i in range(0, total, batch_size):
        batch = to_process[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"\n{'='*50}")
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} articles)")
        print(f"{'='*50}")

        if batch_size > 1:
            # Batch mode: combine multiple paragraphs into a single LLM call.
            result = _ingest_batch_one(batch, cache, force=force)
            batch_success = result["success"]
            batch_failed = result["failed"]
            success += batch_success
            failed += batch_failed
            articles_since_last_periodic += batch_success
            processed_count += len(batch)
            _print_progress_bar(processed_count, total, time.time() - t_total)
        else:
            for article_path in batch:
                ok = ingest_single(article_path, cache, force=force)
                if ok:
                    success += 1
                    articles_since_last_periodic += 1
                else:
                    failed += 1

                processed_count += 1
                _print_progress_bar(processed_count, total, time.time() - t_total)

        issues = quick_lint_bench()
        if issues:
            fixes = auto_fix_bench()
            total_fixed = sum(fixes.values())
            if total_fixed > 0:
                parts = [f"{k}={v}" for k, v in fixes.items() if v > 0]
                print(f"\n  🔧 Quick fix: {', '.join(parts)}")

        if periodic_maintenance(articles_since_last_periodic, total_ingested=success):
            articles_since_last_periodic = 0

    print()

    finalize_wiki()

    t_elapsed = time.time() - t_total
    print(f"\n{'='*50}")
    print(f"✅ Batch ingestion complete: {success} success / {failed} failed / {total} total")
    print(f"  total: {_format_time(t_elapsed)} | avg: {t_elapsed/max(processed_count,1):.1f}s/article")



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Wiki ingestion")
    parser.add_argument("--dataset", "-d", required=True,
                        choices=["hotpotqa", "musique", "2wikimhqa"],
                        help="Dataset to ingest")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit number of articles to ingest")
    parser.add_argument("--batch-size", "-b", type=int, default=3,
                        help="Batch size: articles per LLM call (default: 3)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force re-ingest already processed articles")
    args = parser.parse_args()

    config.set_dataset(args.dataset)
    config.ensure_wiki_dirs()

    raw_dir = BENCH_DIR / "raw" / args.dataset / "articles"
    if not raw_dir.exists():
        print(f"❌ Raw articles directory not found: {raw_dir}")
        print(f"   Please run: python preprocess_bench.py --dataset {args.dataset}")
        sys.exit(1)

    article_paths = sorted(raw_dir.glob("*.md"))
    if not article_paths:
        print(f"❌ No articles found in {raw_dir}")
        sys.exit(1)

    print(f"📦 Dataset: {args.dataset}")
    print(f"📂 Articles: {len(article_paths)} files in {raw_dir}")
    print(f"📁 Wiki output: {config.WIKI_DIR}")

    if args.limit:
        article_paths = article_paths[:args.limit]
        print(f"🔢 Limited to {args.limit} articles")

    ingest_batch(article_paths, batch_size=args.batch_size,
                 limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
