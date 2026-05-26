---
name: optimize
description: Optimize the LLM Wiki. Compacts verbose pages, merges near-duplicates, reorganizes misplaced content, strengthens cross-references, improves consistency, and generates missing synthesis pages. Run periodically as the wiki grows.
user-invokable: true
allowed-tools: read_file, write_file, replace, run_shell_command, grep_search, glob, list_directory, update_topic, invoke_agent, mcp__plugin_qmd_qmd__query, mcp__plugin_qmd_qmd__status, mcp__plugin_qmd_qmd__get, mcp__plugin_qmd_qmd__multi_get
---

# LLM Wiki: Optimize

Compact, reorganize, and improve the wiki. This is the "gardening" operation — it makes the wiki better without adding new sources.

## Step 1: Audit

Read all wiki pages and build a picture of the current state:

1. Read `wiki/index.md` for the master inventory.
2. Glob all `.md` files in `wiki/` recursively.
3. For each page, read frontmatter and scan for wikilinks.
4. Build an in-memory graph: pages as nodes, wikilinks as edges.

Identify issues:

### Compaction Candidates

- Pages that are unnecessarily verbose (ratio of content to information is high)
- Source summaries that just repeat the original without synthesis
- Redundant information repeated across multiple pages

### Merge Candidates

- Pages covering the same concept under different names
- Concept pages that are too small to stand alone — could be sections of a larger page
- Near-duplicate content across pages

### Reorganization Candidates

- Pages in the wrong directory (e.g., a source in `concepts/`)
- Pages missing frontmatter or with incorrect `type` fields
- Filenames that don't follow kebab-case convention

### Cross-Reference Gaps

- Pages that mention concepts without wikilinks
- Concept pages mentioned in sources but lacking backlinks
- Important relationships between pages that aren't explicitly linked

### Synthesis Opportunities

- Topics covered by 3+ sources that lack a synthesis page
- Emerging patterns or themes across multiple pages
- Comparisons that would be valuable but haven't been written

## Step 2: Plan

Present findings to the user as a categorized list:

- **Compact:** N pages to tighten
- **Merge:** N page groups to consolidate
- **Reorganize:** N pages to move/rename
- **Link:** N cross-reference gaps to fill
- **Synthesize:** N synthesis pages to create

Ask the user which categories to proceed with (default: all).

## Step 3: Execute

Process each category. Use parallel subagents where changes are independent.

### Compaction

- Rewrite verbose pages to be more concise while preserving all facts
- Remove redundant passages that exist elsewhere in the wiki
- Tighten source summaries to focus on unique insights

### Merging

- When merging pages, combine all information and sources from both
- Update all wikilinks across the wiki to point to the merged page
- Delete the redundant page
- Update index.md

### Reorganization

- Move misplaced files to correct directories
- Rename files to follow kebab-case convention
- Fix frontmatter `type` fields
- Update all wikilinks that referenced the old name/path

### Cross-Reference Strengthening

- Add missing wikilinks within page content
- Ensure bidirectional links (A→B implies B→A mention)
- Add "Related" sections where they're missing

### Synthesis Generation

- For topics with rich coverage, write synthesis pages in `wiki/synthesis/`
- Synthesis pages should: compare perspectives across sources, identify consensus vs. disagreement, note gaps in coverage, present an evolving thesis
- Follow the standard frontmatter template with `type: synthesis`

## Step 4: Rebuild Index

After all changes, rebuild `wiki/index.md` from scratch by scanning all wiki pages.

## Step 5: Reindex qmd

```bash
qmd update
```

## Step 6: Log

Append to `wiki/log.md`:

```markdown
## [YYYY-MM-DD] optimize | Wiki optimization pass
- Pages compacted: N
- Pages merged: N (list)
- Pages reorganized: N
- Cross-references added: N
- Synthesis pages created: N (list)
```

## Step 7: Report

Summarize all changes made. Highlight any synthesis pages created — these are often the most valuable output of an optimization pass.
