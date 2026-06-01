# Wiki Schema — Structured Knowledge Base

## Directory layout

Page types are declared in `wiki/page_types.yaml`. On first ingestion the LLM
analyses a sample of source articles and proposes a per-corpus directory
schema; you can also set it manually.

**Fixed infrastructure directories:**

```
wiki/
├── sources/               # source-page root
│   ├── digests/           #   structured summaries of source paragraphs
│   └── articles/          #   verbatim copies of the original source text
├── syntheses/             # distilled query-result pages
├── index.md               # listing of directories with descriptions + page counts
├── overview.md            # global overview (regenerated periodically)
├── log.md                 # append-only operation log
└── page_types.yaml        # registry of page types
```

**Dynamic page-type directories** (defined in `page_types.yaml`, generated on
first ingestion; one per corpus):

```
wiki/
├── {type_a}/              # e.g. entities/, events/, concepts/, relations/
├── {type_b}/
└── {type_c}/              # exact names and descriptions vary per corpus
```

## Frontmatter

```yaml
---
type: source|synthesis|{dynamic_type}
aliases: [list of aliases / variant names]
tags: [list of tags]
---
```

> `type` matches the directory name.
> `aliases` is used at retrieval time — list common aliases, foreign names and
> alternative translations.
> `created` and `updated` are injected by the engine; the LLM does not need to
> emit them.

### Generic knowledge-page template (`{dir}/`)

```markdown
---
type: {dir}
aliases: [aliases]
tags: [tags]
---

# Page title

## Basic info
- Key attribute 1:
- Key attribute 2:

## Core facts
- Fact 1 (use full noun phrases — avoid pronouns)
- Fact 2

## Related pages
- [[dir/page]] — relationship description

## Related sources
- [[sources/digests/YYYY-MM-DD-slug]] — what this source contributes
```

### Source-digest template (`sources/digests/`)

> All five sections below are **mandatory** — write them even when the
> source paragraph is short.

```markdown
---
type: source
source_date: YYYY-MM-DD
source_article: <source file stem, without path prefix or .md extension>
tags: [tags]
---

# Source title

> Source: {origin} | {date}

## Summary
A summary of at most 200 words. (required)

## Core claims
- Author/origin asserts that … (judgements/opinions, not just encyclopedic facts) (required)

## Key quotes
- "Direct quotation from the source." — speaker, if applicable (required, ≥1)

## Key facts
- Fact 1
- Fact 2
(required, ≥2)

## Mentioned entities
- [[entity name]]
(required, ≥1)
```

## Writing style (the wiki is consumed by an LLM for retrieval)

- **Information density first.** Prefer structured fact lists over prose paragraphs.
- **Use full noun phrases.** Avoid pronouns ("he", "it") — they break entity matching.
- **Informative headings.** Use "Life and style" rather than "About"; "Structural analysis" rather than "Closer look".
- **Always declare aliases.** Common nicknames, foreign-language names and alternative spellings go into the frontmatter `aliases` field.
- **Link sources by full path.** `[[sources/digests/YYYY-MM-DD-slug]]`.

## Naming conventions

- Knowledge pages: use the canonical English page name (e.g. `Tan_Dun.md`).
- Source digest pages: `YYYY-MM-DD-slug.md` under `sources/digests/`.
- Avoid filesystem-unsafe characters: `/` → `-`, `"` → removed, `|` → `-`.
- Wiki links use `[[page]]` syntax (no `.md` extension).
