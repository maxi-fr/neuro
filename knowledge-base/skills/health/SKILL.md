---
name: health
description: Check LLM Wiki health. Finds orphan pages, broken wikilinks, contradictions, stale content, missing pages, cross-reference gaps, and suggests improvements. Run periodically to keep the wiki in good shape.
user-invokable: true
allowed-tools: read_file, write_file, replace, run_shell_command, grep_search, glob, list_directory, update_topic, invoke_agent, mcp__plugin_qmd_qmd__query, mcp__plugin_qmd_qmd__status, mcp__plugin_qmd_qmd__get, mcp__plugin_qmd_qmd__multi_get
---

# LLM Wiki: Health Check

Audit the wiki for structural issues, content gaps, and maintenance needs.

## Step 1: Inventory

1. Glob all `.md` files in `wiki/` recursively.
2. For each file, read frontmatter and extract:
   - `title`, `type`, `tags`, `sources`, `created`, `updated`
3. Scan each file's content for all `[[wikilinks]]`.
4. Build a page inventory and a link graph.

## Step 2: Structural Checks

### Broken Wikilinks

- Find all `[[Target]]` references across the wiki
- Check if a page with that title exists (case-insensitive filename match)
- Report any broken links (target page doesn't exist)

### Orphan Pages

- Pages with **zero inbound links** from other wiki pages
- Exclude `index.md` and `log.md` from this check
- Orphans are pages nobody references — they may be forgotten or poorly integrated

### Missing Pages

- Wikilinks that point to non-existent pages — these are implicit "wanted pages"
- Rank by how many pages link to them (higher = more important to create)

### Frontmatter Issues

- Pages missing required frontmatter fields (`title`, `type`, `created`)
- Pages with `type` that doesn't match their directory (e.g., `type: comparison` in `concepts/`)
- Pages with empty or placeholder content

### Naming Convention Violations

- Files not using lowercase kebab-case
- Files with spaces or special characters
- Extremely long filenames

## Step 3: Content Checks

### Stale Content

- Pages where `updated` date is more than 30 days old AND newer sources exist that mention the same concepts
- Source summaries whose original source file has been modified since the summary was written

### Contradictions

- Use parallel subagents to spot-check key claims across pages:
  - Find pages about the same concept
  - Compare claims, numbers, dates between them
  - Flag any inconsistencies with specific quotes from each page

### Thin Pages

- Pages with very little content (< 100 words excluding frontmatter)
- Pages that are just stubs with no real information
- Source summaries that lack key takeaways

### Coverage Gaps

- Concepts that appear frequently in tags but have no dedicated page
- Topics where only one source covers them (single-source risk)

## Step 4: Source Drift Check (Manifest)

Read `wiki/.manifest.json` and verify source integrity:

1. For each entry in the manifest, compute the current MD5 of the source file:

   ```powershell
   (Get-FileHash <file> -Algorithm MD5).Hash
   ```

2. Compare against the stored hash. Report:
   - **Modified sources:** MD5 differs — the source has changed since last ingest. Wiki pages may be stale. Suggest running the **ingest** skill to re-process.
   - **Deleted sources:** File no longer exists on disk. Wiki pages referencing it may be orphaned.
   - **Untracked sources:** Files in `raw/`, `docs/`, `notes/` that aren't in the manifest at all — never ingested.
3. Summarize: N modified, N deleted, N untracked out of N total tracked sources.

## Step 5: Index & Log Checks

### Index Accuracy

- Compare `wiki/index.md` entries against actual files
- Pages in the wiki but missing from the index
- Index entries pointing to deleted/renamed pages

### Log Completeness

- Check that recent file changes have corresponding log entries
- Identify files with no ingest log entry (possible manual additions)

## Step 6: qmd Health

Check qmd status:

```bash
qmd status
```

- Is the wiki collection registered?
- How many documents are indexed vs. exist on disk?
- Are embeddings up to date?

If out of sync, suggest running `qmd update` and/or `qmd embed`.

## Step 7: Generate Report

Present findings organized by severity:

### Critical (fix now)

- Broken wikilinks
- Contradictions between pages
- Index out of sync
- Modified sources with stale wiki pages (run **ingest**)

### Warning (fix soon)

- Orphan pages
- Stale content
- Thin pages
- Frontmatter issues

### Info (nice to have)

- Missing pages (wanted but not created)
- Coverage gaps
- Synthesis opportunities
- Naming convention issues

### Statistics

- Total pages: N (concepts: N, sources: N, comparisons: N, synthesis: N)
- Total wikilinks: N (broken: N)
- Orphan pages: N
- Average page age: N days
- Most connected pages (top 5 by link count)
- Least connected pages (excluding new pages)

### Suggested Actions

Prioritized list of maintenance tasks:

1. Fix broken wikilinks (list specific fixes)
2. Resolve contradictions (list specific pages)
3. Create wanted pages (list top 5 by demand)
4. Update stale pages (list specific pages)
5. Write synthesis pages for well-covered topics

## Step 8: Log

Append to `wiki/log.md`:

```markdown
## [YYYY-MM-DD] health | Wiki health check
- Total pages: N
- Broken links: N
- Orphans: N
- Contradictions: N
- Stale pages: N
```
