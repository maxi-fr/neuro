---
name: ingest
description: Ingest new sources into the LLM Wiki. Reads unprocessed files from Literature/ and Notes/, creates source summaries, updates concept pages, maintains cross-references, and updates the index and log. Use when new files have been added.
argument-hint: "[optional: specific file or folder path to ingest]"
user-invokable: true
allowed-tools: read_file, write_file, replace, run_shell_command, grep_search, glob, list_directory, update_topic, invoke_agent, mcp__plugin_qmd_qmd__query, mcp__plugin_qmd_qmd__status, mcp__plugin_qmd_qmd__get, mcp__plugin_qmd_qmd__multi_get
---

# LLM Wiki: Ingest

Process new source files and integrate their knowledge into the wiki.

## Step 1: Identify New and Modified Sources

The wiki tracks every ingested file via an MD5 manifest at `wiki/.manifest.json`. Format:

```json
{
  "docs/research/cross-platform-tech-stacks-2026.md": {
    "md5": "A1B2C3D4E5F6...",
    "ingested": "2026-04-06",
    "wiki_pages": ["sources/cross-platform-tech-stacks-2026.md", "concepts/expo.md"]
  }
}
```

### Detection logic

If the user provided a specific path as an argument, ingest only that file/folder (skip detection, force re-ingest).

Otherwise:

1. Read `wiki/.manifest.json` (create empty `{}` if it doesn't exist).
2. Scan these directories for markdown, PDF, and other readable files:
   - `Literature/` (recursively)
   - `Notes/` (recursively)
3. For each file found, compute its MD5 hash:

   ```powershell
   (Get-FileHash <file> -Algorithm MD5).Hash
   ```

4. Compare against the manifest:
   - **New:** file path not in manifest → queue for ingest
   - **Modified:** file path in manifest but MD5 differs → queue for re-ingest (update existing wiki pages)
   - **Unchanged:** file path in manifest and MD5 matches → skip
   - **Deleted:** file path in manifest but no longer on disk → flag for user attention (don't auto-delete wiki pages)

If no new or modified files are found, report "Nothing new to ingest" and exit.

Present the list of files to ingest, grouped by status (new/modified/deleted), and ask the user for confirmation before proceeding.

### Re-ingesting modified files

When a previously ingested file has changed:

- Read the `wiki_pages` list from the manifest to find all wiki pages that were created/updated from this source
- Re-read the source and update those wiki pages with the new content
- Preserve information from OTHER sources on those pages — only update the parts that came from the modified source
- Note in the log that this was a re-ingest due to source modification

## Step 2: Read the Wiki State

Before processing sources:

1. Read `wiki/index.md` to understand existing pages and structure.
2. Optionally use qmd to search for relevant existing pages that might need updating.

## Step 3: Process Each Source

For each new source file, in order:

### 3a. Read and Analyze

- For markdown or text files, read the full source file.
- For PDF files (`.pdf`):
  1. Execute `uv run skills/ingest/process_pdf.py "<path_to_pdf>" -o Literature/<pdf_name>/`. This will create a markdown file and an `attachments` directory in a new folder alongside the PDF.
  2. Read the generated markdown file.
  3. **Invoke a subagent** to clean up the generated markdown. Automated PDF-to-Markdown conversion can introduce formatting issues or OCR artifacts. Ask the subagent to fix broken formatting, correct typos, ensure headings/lists/tables are well-formed, and improve overall readability.
  4. Write the cleaned markdown back to the generated file.
  5. Use the cleaned markdown content for the remainder of the analysis.
- If the file references images in `Notes/attachments/`, read key images for additional context
- Identify: core concepts, claims/data points, relationships. **Crucially, track the specific locations (e.g., section headings, line numbers) where each claim is made.**
- **Identify and extract mathematical descriptions and formulas** if they are necessary for a complete understanding of the concepts or claims. Use standard LaTeX syntax (`$$...$$` for blocks, `$ ... $` for inline).

### 3b. Create Source Summary

Write a summary page in `wiki/sources/`:

```markdown
---
title: "Source Title"
type: source
tags: [relevant, tags]
sources: [relative/path/to/original.md]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Source Title

**Source:** `relative/path/to/original.md`
**Ingested:** YYYY-MM-DD

## Summary
2-4 paragraph summary of key content. **Integrate mathematical formulas or models logically into the summary if they are central to the source's findings.**

## Key Takeaways
- Bullet points of the most important facts, claims, or insights. **Include citations to specific locations in the raw file using alias syntax for readability (e.g., `[[relative/path/to/original.md#Section Name | Source Name]]`).**

## Related Concepts
- [[concepts/concept-name.md|Concept Name]] — brief note on relevance

## Raw Notes
Any additional details, quotes, or data worth preserving.
```

### 3c. Create or Update Concept Pages

For each significant concept mentioned in the source:

- If `wiki/concepts/<concept>.md` exists: **update it** — add new information from this source, update the `sources` list in frontmatter, add `updated` date, ensure cross-references are correct. Always cite the specific source and location for new claims.
- If it doesn't exist: **create it** with information from this source. Add wikilinks to related concepts.

Concept page template:

```markdown
---
title: Concept Name
type: concept
tags: [relevant, tags]
sources: [source1.md, source2.md]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Concept Name

Definition and explanation of the concept. **Integrate formal mathematical definitions or models here if they are essential to defining the concept.**

## Key Points
Information aggregated from all sources. **Every distinct claim must include a citation to the specific source and location using alias syntax for readability (e.g., `[[path/to/original.md#Heading | Source Name]]`).** **Integrate mathematical descriptions naturally into the key points where they provide necessary formal backing.**

## Related Concepts
- [[concepts/other-concept.md|Other Concept]] — relationship description

## Sources
- [[sources/source-summary.md|Source Name]] — what this source says about the concept (include references to specific sections)
```

### 3d. Update Cross-References

After creating/updating pages for a source:

- Check all pages that were created or updated
- Ensure bidirectional wikilinks exist and use full relative paths with aliases (e.g., `[[concepts/concept-name.md|Concept Name]]`)
- Add links to the source summary from all concept pages it touches

## Step 4: Update Index

Rebuild `wiki/index.md`:

- Read all `.md` files in `wiki/` subdirectories
- Group by type (concept, source, comparison, synthesis)
- For each page, list: `- [[Page Title]] — one-line description` (read from frontmatter or first paragraph)
- Sort alphabetically within each group

## Step 5: Update Manifest

After processing each source file, update `wiki/.manifest.json`:

```powershell
# Compute MD5 for the ingested file
(Get-FileHash <file> -Algorithm MD5).Hash
```

Update the manifest entry:

```json
{
  "relative/path/to/file.md": {
    "md5": "<current md5 hash>",
    "ingested": "YYYY-MM-DD",
    "wiki_pages": ["sources/page.md", "concepts/concept.md"]
  }
}
```

The `wiki_pages` array lists all wiki pages that were created or updated from this source. This is used by future re-ingests to know which pages to update when the source changes.

Read the existing manifest, merge in the new/updated entries, and write it back. Use the Bash tool to read/write JSON if needed.

## Step 6: Update Log

Append to `wiki/log.md`:

```markdown
## [YYYY-MM-DD] ingest | filename.md
- Source: relative/path/to/file.md
- Status: new | modified (previous hash: abc123)
- Pages created: list of new pages
- Pages updated: list of updated pages
- New concepts: count
```

## Step 7: Reindex qmd

If qmd is available:

```bash
qmd update
```

## Step 8: Report

Print a summary for the user:

- Files ingested
- Pages created (with links)
- Pages updated (with what changed)
- Any contradictions or notable findings between new and existing content

## Important Notes

- **Never modify source files** in `Literature/` or `Notes/`. They are immutable.
- **Flag contradictions.** If new source data conflicts with existing wiki claims, note the contradiction on both pages and mention it in the report.
- **Be thorough but concise.** Source summaries should capture all key information but not be longer than necessary.
- **Use parallel subagents** when ingesting multiple files to speed up processing.
- When updating existing pages, preserve information from previous sources. Add, don't replace.
