# LLM Wiki Agents

## LLM Wiki

This workspace includes an LLM-maintained wiki — a persistent, interlinked knowledge base
built from source documents. The LLM writes and maintains the wiki; the user curates sources
and directs analysis.

### Source Directories (Immutable — never modify)

- **`Literature/`** — Primary source documents (articles, papers, books)
- **`Notes/`** — Research notes, clippings, and personal synthesis

### Wiki Output (`wiki/`)

All wiki pages live here. The LLM owns this directory entirely. Key files:

- `wiki/index.md` — Master index, updated on every ingest. Read this first when searching.
- `wiki/log.md` — Append-only chronological log of all operations.

### Wiki Page Conventions

- **Frontmatter:** Every page has YAML frontmatter with `title`, `type`
  (concept/summary/comparison/synthesis), `tags`, `sources`, `created`, `updated`.
- **Wikilinks:** Use `[[Page Title]]` for cross-references (Obsidian-compatible). Always bidirectional.
- **Filenames:** Lowercase kebab-case, `.md` extension. No spaces or special characters.

### Skills

You have access to the following specialized skills. Activate them when the user's intent matches the trigger condition:

| Skill | Purpose | When to Activate |
|-------|---------|------------------|
| **ingest** | Process new/modified sources, create/update wiki pages | When the user asks you to ingest new sources, process new files, or mentions adding new files to `Literature/` or `Notes/`. |
| **search** | Search wiki via qmd and synthesize answers with citations | When the user asks a knowledge question that requires searching the wiki. |
| **optimize** | Compact, merge, reorganize, strengthen cross-references | When the user asks you to optimize, clean up, or garden the wiki. |
| **health** | Audit for broken links, orphans, contradictions, stale content | When the user asks for a health check, audit, or review of wiki quality. |

### Search (qmd)

The wiki is indexed by qmd (local markdown search engine) with collection name `wiki`.
Skills use qmd MCP tools automatically. To reindex manually: `qmd update`.
To build embeddings: `qmd embed`.
