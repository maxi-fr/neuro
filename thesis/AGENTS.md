# Thesis instructions

Treat `thesis/` as a document workspace. For work within this directory, parent-repository instructions do not apply here.

Keep file inspection and changes scoped to `thesis/`. Inspect parent-repository files only when the requested task requires them, such as checking the implementation against the thesis content. Inherited instructions do not expand the task's scope.

Use the latex-lint command when needed:

```bash
usage: uv run latex-lint [-h] {compile,check,rule,install-skills,refs,format} ...

positional arguments:
  {compile,check,rule,install-skills,refs,format}
    compile             compile LaTeX document via pdflatex and biber recipe (<root> [-o/--output-dir PATH])
    check               check an explicit UTF-8 root document (<root> [--ignore RULES])
    rule                show an implemented rule and examples (<rule_id>)
    install-skills      install bundled agent skills ([preset] [-d/--dest PATH] [-f/--force])
    refs                manage reference papers and text extraction (fetch, extract)
    format              format LaTeX files with one sentence per line (<paths>...)

options:
  -h, --help            show this help message and exit
```

Add all temporary files to tmp/

When creating or editing figures follow [FIGURE_INSTRUCTIONS.md]
