---
name: search
description: Search the LLM Wiki using qmd (hybrid BM25/vector search). Finds relevant wiki pages, reads them, and synthesizes an answer with citations. Use when asking questions against the knowledge base.
argument-hint: "<search query>"
user-invokable: true
allowed-tools: read_file, write_file, replace, run_shell_command, grep_search, glob, list_directory, update_topic, invoke_agent, mcp__plugin_qmd_qmd__query, mcp__plugin_qmd_qmd__status, mcp__plugin_qmd_qmd__get, mcp__plugin_qmd_qmd__multi_get
---

# LLM Wiki: Search

Search the wiki and synthesize an answer from the knowledge base.

## Step 1: Parse Query

The user's argument is the search query. If no argument is provided, ask the user what they'd like to search for.

## Step 2: Search Strategy

Use a multi-pronged search approach:

### 2a. qmd Search (Primary)

Use the qmd MCP tools to search the `wiki` collection:

```json
{
  "searches": [
    { "type": "lex", "query": "<key terms extracted from query>" },
    { "type": "vec", "query": "<full natural language query>" }
  ],
  "collections": ["wiki"],
  "intent": "<what the user is looking for>",
  "limit": 15,
  "minScore": 0.3
}
```

For complex queries, add a `hyde` sub-query with a hypothetical answer (50-100 words).

### 2b. Index Scan (Fallback/Supplement)

Read `wiki/index.md` and scan for relevant page titles and descriptions. This catches pages that qmd might miss due to indexing lag.

### 2c. Grep (Precision)

If the query contains specific terms, names, or identifiers, use Grep to find exact matches across wiki pages.

## Step 3: Read Relevant Pages

From the search results, identify the most relevant pages (typically 3-10). Read them in full using the Read tool or `qmd get`.

For each page, note:

- How it relates to the query
- Key facts or claims
- Which sources it draws from
- What wikilinks it contains that might lead to more relevant pages

If the initial pages reference other wiki pages that seem highly relevant, follow those links too (one level deep).

## Step 4: Synthesize Answer

Compose an answer that:

1. **Directly answers the query** based on wiki content
2. **Cites sources** using wikilinks: `[[Page Title]]`
3. **Notes confidence level** — how well-covered is this topic in the wiki?
4. **Identifies gaps** — what's missing from the wiki that would help answer this better?
5. **Suggests follow-ups** — related questions the user might want to explore

Format the answer in clean markdown. Use headers, bullet points, and tables where they aid clarity.

## Step 5: Offer to File

If the answer is substantial and adds value beyond what's already in the wiki, offer to file it:

> "This analysis could be saved as a wiki page. Would you like me to file it in `wiki/comparisons/` or `wiki/synthesis/`?"

If the user agrees, write the page following standard wiki conventions (frontmatter, wikilinks, proper filename) and update the index.

Append to `wiki/log.md`:

```markdown
## [YYYY-MM-DD] query | <short query description>
- Query: <full query>
- Pages consulted: [[Page1]], [[Page2]], ...
- Filed as: wiki/path/to/new-page.md (if applicable)
```

## Tips

- **Combine search types.** Lexical search finds exact terms; vector search finds related concepts. Use both.
- **Follow the links.** Wiki pages are richly interlinked — the best answer often comes from reading 2-3 connected pages.
- **Check source freshness.** Note the `updated` date in frontmatter. If a page hasn't been updated recently, mention this.
- **Be honest about gaps.** If the wiki doesn't cover something well, say so. Don't hallucinate from outside the wiki unless the user asks.
