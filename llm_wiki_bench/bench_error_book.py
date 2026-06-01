#!/usr/bin/env python3
"""Error book — closed loop for wiki-quality improvement.

Design:
  detect issue -> record entry -> inject constraint into the prompt ->
  next generation avoids it -> verify disappearance -> close.

Storage:
- ``error_book.yaml`` (sits next to the wiki) — co-edited by code and humans.
- ``lint_ledger.jsonl`` — append-only audit log of every fix attempt.

Each entry tracks: phenomenon, root cause, generation constraint, check
method, and a per-sample fix status.

Two orthogonal states:
1. Entry status (``open``/``closed``) — controls whether the constraint is
   injected into the LLM prompt.
2. Fix status (``fixed: true/false`` on each sample) — controls whether the
   periodic LLM repair pass should attempt to fix it.
"""

import json
import re
import yaml
from datetime import datetime
from pathlib import Path

import bench_config as config


# ─── Core IO ───

def _get_error_book_path() -> Path:
    """Path to the error-book YAML file."""
    wiki_dir = config.WIKI_DIR
    if wiki_dir:
        return wiki_dir / "error_book.yaml"
    return Path("error_book.yaml")


def load_error_book() -> list[dict]:
    """Load the error book."""
    path = _get_error_book_path()
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("errors", [])
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_error_book(errors: list[dict]):
    """Save the error book."""
    path = _get_error_book_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.dump(
        {"errors": errors},
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    path.write_text(content, encoding="utf-8")


# ─── Constraint injection ───

def _get_samples(e: dict) -> list:
    """Read the sample list, supporting both legacy and current schemas.

    Current schema uses ``active_samples`` (only live broken links); the legacy
    schema used ``samples``.
    """
    return e.get("active_samples", e.get("samples", []))


def _get_count(e: dict) -> int:
    """Read the issue count, supporting both schemas.

    Current schema uses ``still_active``; the legacy schema used ``count``.
    """
    if "still_active" in e:
        return e["still_active"]
    return e.get("count", 0)


def _set_samples(e: dict, samples: list):
    """Write the sample list, picking the field present on the entry."""
    if "active_samples" in e:
        e["active_samples"] = samples
    else:
        e["samples"] = samples


def _set_count(e: dict, count: int):
    """Write the issue count, picking the field present on the entry."""
    if "still_active" in e:
        e["still_active"] = count
    else:
        e["count"] = count


def get_active_constraints() -> str:
    """Return the formatted constraint text for every open error-book entry.

    The returned string is ready to be appended to the system / user prompt.
    """
    errors = load_error_book()
    if not errors:
        return ""

    active = [e for e in errors if e.get("status") != "closed"]
    if not active:
        return ""

    lines = []
    for e in active:
        constraint = e.get("constraint", "")
        eid = e.get("id", "?")
        category = e.get("category", "")

        if e.get("brief") or category == "broken_link":
            # Short mode: emit only the running counts.
            count = _get_count(e)
            samples = _get_samples(e)
            sample_names = [_sample_name(s) for s in samples[:3]]
            sample_str = ", ".join(sample_names) if sample_names else ""
            if count > 0:
                line = f"- [{eid}] {count} broken links still active"
                if sample_str:
                    line += f" (e.g., {sample_str})"
                lines.append(line)
        elif constraint:
            lines.append(f"- [{eid}] {constraint}")

    if not lines:
        return ""

    return "## Known Issues (must avoid)\n\n" + "\n".join(lines) + "\n"


# ─── Issue logging ───

# Mapping from lint-issue category to error-book template.
_ISSUE_TEMPLATES = {
    "broken_links": {
        "category": "broken_link",
        "description": "Broken links: wikilinks pointing to non-existent pages (auto-deleted, pending LLM fix to create missing pages)",
        "constraint": "Avoid creating links to non-existent pages. If referencing a new entity, create the page simultaneously",
        "brief": True,
        "needs_llm_fix": True,
    },
    "index_inconsistencies": {
        "category": "index_error",
        "description": "Index inconsistency: _index.md references non-existent pages",
        "constraint": "When creating new knowledge pages, always add an index entry in the corresponding _index.md",
        "needs_llm_fix": True,
    },
    "duplicates": {
        "category": "duplicate",
        "description": "Duplicate pages: same entity created under different names",
        "constraint": "Before creating a new page, check existing page names. Same entity must update existing page, not create new one",
    },
    "digest_incomplete": {
        "category": "digest_incomplete",
        "description": "Incomplete digest: missing required sections (Summary/Key Facts/Key Entities/Related Context)",
        "constraint": "Every source digest page (sources/digests/) must contain all 4 required sections: ## Summary, ## Key Facts, ## Key Entities, ## Related Context. None can be omitted",
        "needs_llm_fix": True,
    },
    "type_path_mismatch": {
        "category": "type_path_mismatch",
        "description": "Type-path mismatch: frontmatter type field doesn't match directory name",
        "constraint": "Each page's frontmatter `type` field must match its directory name exactly",
    },
    "missing_source_article": {
        "category": "missing_source_article",
        "description": "Digest missing source_article: cannot precisely locate original paragraph",
        "constraint": "When generating sources/digests/ pages, frontmatter must include source_article field with the corresponding article filename (without .md suffix)",
        "needs_llm_fix": True,
    },
    "unseen_page_overwrite": {
        "category": "unseen_page_overwrite",
        "description": "LLM attempted to overwrite unseen page: Step1 didn't select this page but Step2 still output an update (blocked by code)",
        "constraint": "Only modify pages whose full content was shown in 'Existing Page Content'. For pages you only see in the index, reference them via [[...]] only — do NOT output updates",
    },
    "missing_summary": {
        "category": "missing_summary",
        "description": "Knowledge page missing one-sentence summary: no > blockquote after # title",
        "constraint": "Every knowledge page must have a > one-sentence summary (blockquote) immediately after the # title line",
        "needs_llm_fix": True,
    },
    "missing_sections": {
        "category": "missing_sections",
        "description": "Knowledge page missing required sections: Key Facts / Related Pages / Related Sources",
        "constraint": "Every knowledge page must include ## Key Facts (≥2 facts), ## Related Pages, and ## Related Sources sections",
        "needs_llm_fix": True,
    },
}


def record_lint_issues(issues: dict, batch_article_stems: list[str] | None = None):
    """Record lint findings into the error book.

    New issues (keyed by ``category``) are appended; pre-existing entries get
    their ``last_seen`` timestamp and sample list updated.

    Args:
      issues: lint result dict.
      batch_article_stems: stems of articles in the current batch; attached as
        ``context.batch_articles`` on broken-link samples so the periodic LLM
        repair pass can pull the original text.
    """
    errors = load_error_book()
    today = datetime.now().strftime("%Y-%m-%d")
    changed = False

    for issue_key, items in issues.items():
        if issue_key not in _ISSUE_TEMPLATES:
            continue

        template = _ISSUE_TEMPLATES[issue_key]
        category = template["category"]

        # Check whether this category already has an entry.
        existing = None
        for e in errors:
            if e.get("category") == category:
                existing = e
                break

        item_count = len(items) if isinstance(items, list) else 1

        # Collect concrete samples (up to 5).
        samples = []
        if isinstance(items, list):
            for item in items[:5]:
                if isinstance(item, dict):
                    if "from" in item and "to" in item:
                        sample = {"name": f"{item['from']} → [[{item['to']}]]", "fixed": False}
                    elif "title" in item:
                        sample = {"name": item["title"], "fixed": False}
                    else:
                        continue
                elif isinstance(item, str):
                    sample = {"name": item[:100], "fixed": False}
                else:
                    continue
                # broken_link entries carry the batch's article stems so the periodic LLM
            # repair pass can read the original text.
                if category == "broken_link" and batch_article_stems:
                    sample["context"] = {
                        "batch_articles": list(batch_article_stems),
                        "recorded_at": today,
                    }
                samples.append(sample)

        if existing:
            existing["last_seen"] = today
            if existing.get("needs_llm_fix") and category == "broken_link":
                # Broken-link entries: deduplicate-and-accumulate, preserving any existing
            # ``fixed`` flag and ``context`` from previous runs.
                existing_samples = _get_samples(existing)
                existing_map = {}  # name → sample_dict
                for s in existing_samples:
                    s = _normalize_sample(s)
                    existing_map[_sample_name(s)] = s
                for s in samples:
                    s = _normalize_sample(s)
                    name = _sample_name(s)
                    if " → " in name:
                        target = name.split(" → ")[-1].strip("[]")
                    else:
                        target = name.strip("[]")
                    if target not in existing_map:
                        existing_map[target] = s
                new_samples_list = [
                    v if isinstance(v, dict) else {"name": k, "fixed": False}
                    for k, v in sorted(existing_map.items(), key=lambda x: x[0])
                ]
                _set_samples(existing, new_samples_list)
                _set_count(existing, len(existing_map))
            else:
                _set_count(existing, item_count)
                _set_samples(existing, samples)
            if existing.get("status") == "closed":
                existing["status"] = "open"
                existing["pass_count"] = 0
                print(f"  📔 Error {existing['id']} reappeared, reopened")
            changed = True
        else:
            # Create a new entry.
            max_id = 0
            for e in errors:
                eid = e.get("id", "")
                if eid.startswith("E") and eid[1:].isdigit():
                    max_id = max(max_id, int(eid[1:]))
            new_id = f"E{max_id + 1:02d}"

            new_error = {
                "id": new_id,
                "category": category,
                "description": template["description"],
                "constraint": template["constraint"],
                "status": "open",
                "count": item_count,
                "samples": samples,
                "discovered_at": today,
                "last_seen": today,
                "pass_count": 0,
            }
            for flag_key in ("brief", "needs_llm_fix"):
                if template.get(flag_key):
                    new_error[flag_key] = True
            errors.append(new_error)
            changed = True
            print(f"  📔 New error {new_id}: {template['description']} ({item_count} issues)")

    # Bump the pass_count of pre-existing entries that did not show up this round.
    seen_categories = set()
    for issue_key in issues:
        if issue_key in _ISSUE_TEMPLATES:
            seen_categories.add(_ISSUE_TEMPLATES[issue_key]["category"])

    for e in errors:
        if e.get("status") == "open" and e.get("category") not in seen_categories:
            if e.get("needs_llm_fix"):
                continue  # entries needing LLM repair are not closed via pass_count
            e["pass_count"] = e.get("pass_count", 0) + 1
            if e["pass_count"] >= 2:
                e["status"] = "closed"
                e["closed_at"] = today
                print(f"  📔 Error {e['id']} not seen for 2 checks, closed")
            changed = True

    # Hard-delete entries that have been closed for more than 30 days.
    _cleanup_old_closed(errors)

    if changed:
        save_error_book(errors)


def record_sample_with_context(category: str, sample_name: str,
                                context: dict | None = None,
                                template: dict | None = None):
    """Record a single error-book sample together with its repair context.

    Used during ingestion to proactively log issues (e.g. a digest missing
    ``source_article`` records the candidate stems for later repair).
    """
    errors = load_error_book()
    today = datetime.now().strftime("%Y-%m-%d")

    existing = None
    for e in errors:
        if e.get("category") == category and e.get("status") != "closed":
            existing = e
            break

    if existing is None:
        if template is None:
            template = {
                "description": f"{category} (auto-recorded)",
                "constraint": "",
                "needs_llm_fix": True,
            }
        max_id = 0
        for e in errors:
            eid = e.get("id", "")
            if eid.startswith("E") and eid[1:].isdigit():
                max_id = max(max_id, int(eid[1:]))
        new_id = f"E{max_id + 1:02d}"
        existing = {
            "id": new_id,
            "category": category,
            "description": template.get("description", ""),
            "constraint": template.get("constraint", ""),
            "status": "open",
            "count": 0,
            "samples": [],
            "discovered_at": today,
            "last_seen": today,
            "pass_count": 0,
        }
        if template.get("needs_llm_fix"):
            existing["needs_llm_fix"] = True
        errors.append(existing)

    # Skip if a sample with the same key already exists.
    current_samples = _get_samples(existing)
    for s in current_samples:
        if _sample_name(s) == sample_name:
            if not _sample_is_fixed(s):
                if context and isinstance(s, dict):
                    s.setdefault("context", {}).update(context)
                _set_samples(existing, current_samples)
                save_error_book(errors)
            return

    new_sample = {"name": sample_name, "fixed": False}
    if context:
        new_sample["context"] = context
    current_samples.append(new_sample)
    _set_samples(existing, current_samples)
    _set_count(existing, len(current_samples))
    existing["last_seen"] = today
    save_error_book(errors)


# ─── Fix-state management ───

def _normalize_sample(s) -> dict:
    """Upgrade a legacy plain-string sample to the dict shape with fix state.

    Accepts both shapes:
      - legacy: a plain page-name string;
      - current: ``{"name": "...", "fixed": False}``.
    """
    if isinstance(s, dict):
        return s
    return {"name": str(s), "fixed": False}


def _sample_name(s) -> str:
    """Extract the page name from a sample (dict or plain string)."""
    if isinstance(s, dict):
        return s.get("name", "")
    return str(s)


def _sample_is_fixed(s) -> bool:
    """Return True if the sample has been marked as fixed."""
    if isinstance(s, dict):
        return bool(s.get("fixed", False))
    return False


def get_unfixed_samples(error_entry: dict) -> list[str]:
    """Return the names of all unfixed samples in an error-book entry."""
    return [_sample_name(s) for s in _get_samples(error_entry)
            if not _sample_is_fixed(s)]


def has_unfixed_samples(category: str = None) -> bool:
    """Return True if there are unfixed samples of the given category (or any category)."""
    errors = load_error_book()
    for e in errors:
        if e.get("status") == "closed":
            continue
        if category and e.get("category") != category:
            continue
        if get_unfixed_samples(e):
            return True
    return False


def mark_samples_fixed(category: str, fixed_names: list[str]):
    """Mark matching samples in the given category as fixed.

    ``fixed_names`` is a list of sample names that were repaired.
    Matching rules:
      - Exact match.
      - Prefix match (e.g. passing "sources/digests/foo.md" matches a sample
        "sources/digests/foo.md: missing ...").
      - Match after stripping the directory prefix.
      - For "source_page -> [[target_link]]" entries, match the target part.
      - Pipe / comma alias split (e.g. "hotels/A | B, C" splits into A, B, C
        and each piece is matched).
      - Suffix match (e.g. ``fixed_name`` matches "dir/fixed_name").
    """
    if not fixed_names:
        return
    errors = load_error_book()
    fixed_set = set(fixed_names)
    changed = False

    for e in errors:
        if e.get("category") != category:
            continue
        if e.get("status") == "closed":
            continue
        new_samples = []
        for s in _get_samples(e):
            s = _normalize_sample(s)
            name = _sample_name(s)
            # Strip the directory prefix.
            name_bare = name.rsplit("/", 1)[-1] if "/" in name else name
            # For "source -> [[target]]" format, extract the target.
            name_target = ""
            name_target_bare = ""
            name_target_parts = []  # pipe-separated alias parts
            if " → " in name:
                name_target = name.split(" → ")[-1].strip("[]")
                name_target_bare = name_target.rsplit("/", 1)[-1] if "/" in name_target else name_target
                # Handle pipe-alias format: e.g. "hotels/A | B, C".
                if " | " in name_target_bare:
                    main_part = name_target_bare.split(" | ")[0].strip()
                    alias_part = name_target_bare.split(" | ", 1)[1].strip()
                    name_target_parts = [main_part] + [a.strip() for a in alias_part.split(",")]
                elif ", " in name_target_bare:
                    name_target_parts = [a.strip() for a in name_target_bare.split(",")]
            matched = (name in fixed_set
                       or name_bare in fixed_set
                       or name_target in fixed_set
                       or name_target_bare in fixed_set
                       or any(p in fixed_set for p in name_target_parts)
                       or any(name.startswith(fn) for fn in fixed_set)
                       or any(name.endswith("/" + fn) for fn in fixed_set))
            if matched and not s.get("fixed"):
                s["fixed"] = True
                changed = True
            new_samples.append(s)
        # Update the still_active count after marking a fix.
        if "still_active" in e:
            old_fixed_count = sum(1 for s in _get_samples(e) if _sample_is_fixed(s))
            new_fixed_count = sum(1 for s in new_samples if _sample_is_fixed(s))
            _set_samples(e, new_samples)
            e["still_active"] = sum(1 for s in new_samples if not _sample_is_fixed(s))
            e["resolved_by_page_creation"] = e.get("resolved_by_page_creation", 0) + (new_fixed_count - old_fixed_count)
        else:
            _set_samples(e, new_samples)

    if changed:
        save_error_book(errors)
        print(f"  📔 Marked {len(fixed_names)} {category} sample(s) as fixed")


# ─── Summary printing ───

def print_error_book():
    """Print a human-readable error-book summary."""
    errors = load_error_book()
    if not errors:
        print("  📔 Error book is empty")
        return

    open_errors = [e for e in errors if e.get("status") != "closed"]
    closed_errors = [e for e in errors if e.get("status") == "closed"]

    print(f"\n{'='*60}")
    print(f"  📔 Error Book ({len(open_errors)} active / {len(closed_errors)} closed)")
    print(f"{'='*60}")

    if open_errors:
        print(f"\n  🔴 Active errors:")
        for e in open_errors:
            eid = e.get("id", "?")
            desc = e.get("description", "")
            category = e.get("category", "")

            if category == "broken_link" and "still_active" in e:
                # Current schema: show detailed fix statistics.
                total = e.get("total_discovered", 0)
                by_creation = e.get("resolved_by_page_creation", 0)
                by_removal = e.get("resolved_by_link_removal", 0)
                still_active = e.get("still_active", 0)
                print(f"    [{eid}] {desc}")
                print(f"         total={total}, created={by_creation}, removed={by_removal}, still_active={still_active}")
            else:
                # Legacy schema or other types.
                count = _get_count(e)
                unfixed = get_unfixed_samples(e)
                fixed_count = count - len(unfixed)
                print(f"    [{eid}] {desc}")
                print(f"         {count} total, {fixed_count} fixed, {len(unfixed)} remaining")

            for s in _get_samples(e)[:3]:
                tag = "✅" if _sample_is_fixed(s) else "❌"
                print(f"         {tag} {_sample_name(s)[:80]}")

    if closed_errors:
        print(f"\n  🟢 Closed: {len(closed_errors)} errors")

    print()


# ─── Expiry cleanup ───

def _cleanup_old_closed(errors: list[dict], max_age_days: int = 30):
    """Hard-delete entries that have been closed for more than ``max_age_days`` days.

    Closed entries are no longer injected into the prompt; keeping them indefinitely wastes space.
    """
    today = datetime.now()
    to_remove = []
    for e in errors:
        if e.get("status") != "closed":
            continue
        closed_at = e.get("closed_at", "")
        if not closed_at:
            continue
        try:
            closed_dt = datetime.strptime(closed_at, "%Y-%m-%d")
            age = (today - closed_dt).days
            if age > max_age_days:
                to_remove.append(e)
        except ValueError:
            pass

    for e in to_remove:
        errors.remove(e)
        print(f"  🗑️ Cleaned up expired error {e.get('id', '?')} (closed {(today - datetime.strptime(e['closed_at'], '%Y-%m-%d')).days} days ago)")


# ─── Repair log (lint_ledger.jsonl) ───

def _get_ledger_path() -> Path:
    """Path to the repair-log JSONL file."""
    wiki_dir = config.WIKI_DIR
    if wiki_dir:
        return wiki_dir / "lint_ledger.jsonl"
    return Path("lint_ledger.jsonl")


def append_ledger(
    issue_type: str,
    file: str = "",
    auto_fixed: bool = True,
    fix_method: str = "",
    note: str = "",
    count: int = 1,
):
    """Append one repair-log entry to ``lint_ledger.jsonl``.

    Args:
      issue_type   issue category (e.g. broken_link, digest_incomplete).
      file         file path or page name involved.
      auto_fixed   True if fixed by code, False if fixed by the LLM.
      fix_method   description of the fix (e.g. delete_link, llm_create_page).
      note         free-form note.
      count        number of issues fixed in this entry (default 1; >1 for
                   batched fixes).
    """
    path = _get_ledger_path()
    entry = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "issue_type": issue_type,
        "file": file,
        "auto_fixed": auto_fixed,
        "fix_method": fix_method,
        "note": note,
        "count": count,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # log write failures must not interrupt the main pipeline


def load_ledger() -> list[dict]:
    """Load all repair-log entries."""
    path = _get_ledger_path()
    if not path.exists():
        return []
    entries = []
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def get_unfixed_samples_full(category: str) -> list[dict]:
    """Return all unfixed samples (with context) for the given category, for use by the repair pass."""
    errors = load_error_book()
    out: list[dict] = []
    for e in errors:
        if e.get("category") != category or e.get("status") == "closed":
            continue
        for s in _get_samples(e):
            s_n = _normalize_sample(s)
            if not _sample_is_fixed(s_n):
                out.append(s_n)
    return out
