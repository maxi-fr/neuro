# Reasearch LLM Wiki

A collection of coding agent skills for knowledge management, wiki building, and more.

## Skills

### LLM Wiki

A pattern for building persistent, interlinked knowledge bases using LLMs. Instead of re-deriving knowledge from raw documents on every query (like RAG), the LLM incrementally builds and maintains a structured wiki of markdown files. The wiki compounds over time as you add sources and ask questions.

Based on [Andrej Karpathy's LLM Wiki idea](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). See also the [local copy with annotations](LLMWiki.md).

| Skill | Purpose |
|-------|---------|
| `ingest` | Process new/modified sources into the wiki (change detection) |
| `search` | Search wiki via qmd and synthesize answers with citations |
| `optimize` | Compact, merge, reorganize, strengthen cross-references |
| `health` | Audit for broken links, orphans, contradictions, source drift |

**How it works:**

1. You drop source files (articles, papers, notes) into `Literature/` or `Notes/`.
2. Ask the agent to **ingest** the wiki -- the LLM reads each source, creates summary pages, concept pages, and maintains cross-references using `[[wikilinks]]`.
3. Query the wiki by asking the agent to **search** -- get synthesized answers with citations.
4. The wiki keeps getting richer with every source you add and every question you ask.

**Works great with [Obsidian](https://obsidian.md/)** -- the wiki is just a folder of interlinked markdown files. Open it in Obsidian and use graph view to see the shape of your knowledge base.

## Installation

This wiki setup is optimized for **Gemini CLI**.

1. Ensure you have [Gemini CLI](https://github.com/google/gemini-cli) installed.
2. Clone this repository.
3. The `GEMINI.md` file in the root will automatically provide the agent with the necessary context and instructions.

## Prerequisites

- **Gemini CLI** (running on Windows/macOS/Linux)
- **[qmd](https://github.com/tobi/qmd)** (optional but recommended) -- local markdown search engine with hybrid BM25/vector search. Install with `npm install -g @tobilu/qmd`. Without qmd, skills fall back to index-based search and grep.

## Folder Structure

Source directories (never modified by the LLM):

```text
Literature/           # Primary source documents (papers, PDFs)
Notes/                # Research notes, clippings, and synthesis
```

Wiki output (owned by the LLM):

```text
wiki/
├── index.md          # Master index of all pages (updated on every ingest)
├── log.md            # Chronological log of all operations
├── .manifest.json    # MD5 hashes for source change detection
├── concepts/         # Conceptual/topic pages
├── sources/          # One summary page per ingested source document
├── comparisons/      # Comparison tables, side-by-side analyses
└── synthesis/        # Cross-cutting analyses, evolving theses
```

skills/               # Skills for the LLM to use
...
└── <skill-name>/
    └── SKILL.md

## Page Conventions

Every wiki page has YAML frontmatter:

```yaml
---
title: Page Title
type: concept | source | comparison | synthesis
tags: [tag1, tag2]
sources: [path/to/original.md]
created: 2026-04-06
updated: 2026-04-06
---
```

Cross-references use Obsidian-compatible `[[wikilinks]]`. All links are bidirectional.

## Change Detection

The ingest skill tracks every source file via MD5 checksums in `wiki/.manifest.json`. On Windows, it uses `CertUtil` or PowerShell for hashing. On each run it detects:

- **New files** -- not in manifest, queued for ingest
- **Modified files** -- MD5 changed, queued for re-ingest (updates existing wiki pages)
- **Deleted files** -- flagged for user attention
- **Unchanged files** -- skipped

The health skill also checks for source drift and reports stale wiki pages.

## Usage

### Setup

The root `GEMINI.md` file contains the base instructions for the agent.

### Skills

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `ingest` | Process new/modified sources, create/update wiki pages | After adding new files to `Literature/` or `Notes/` |
| `search` | Search wiki via qmd and synthesize answers with citations | When asking questions against the knowledge base |
| `optimize` | Compact, merge, reorganize, strengthen cross-references | Periodically as wiki grows (every ~10-20 ingests) |
| `health` | Audit for broken links, orphans, contradictions, stale content | Periodically to maintain wiki quality |

To use these skills, simply ask the agent to perform the task (e.g., "Ingest the new sources" or "Search the wiki for X").

### Workflow

1. Drop source files into `Literature/` or `Notes/` (use Obsidian Web Clipper for articles).
2. Ask the agent to **ingest** the new sources.
3. Ask the agent to **search** the wiki when you have questions.
4. Periodically ask for a **health check** to maintain quality.
5. Periodically ask to **optimize** the wiki as it grows.

### Search (qmd)

The wiki is indexed by qmd (local markdown search engine) with collection name `wiki`.
Skills use qmd MCP tools automatically. To reindex manually: `qmd update`.
To build embeddings: `qmd embed`.

### Tips

- **[Obsidian Web Clipper](https://obsidian.md/clipper)** is a browser extension that converts web articles to markdown. Very useful for quickly getting sources into `Literature/` or `Notes/`.
- **Download images locally.** In Obsidian, set attachment folder path to `Notes/attachments/` and bind a hotkey to download attachments. This lets the LLM view images directly.
- **Obsidian's graph view** is the best way to see the shape of your wiki -- what's connected, which pages are hubs, which are orphans.
- **Good answers can be filed back.** When a search produces a valuable analysis, the agent may offer to save it as a wiki page so your explorations compound too.
- **The wiki is just markdown files.** It's a git repo, version-controlled, portable, and works with any tool that reads markdown.

## Adding Your Own Skills

To add a new skill to this plugin:

1. Create a directory under `skills/` with your skill name (kebab-case).
2. Add a `SKILL.md` file with YAML frontmatter (`name`, `description`, `allowed-tools`).
3. The skill will be automatically discovered when the plugin is installed.

## Credits

The LLM Wiki pattern is based on [Andrej Karpathy's idea](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) of using LLMs to incrementally build and maintain persistent knowledge bases instead of re-deriving knowledge on every query.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
