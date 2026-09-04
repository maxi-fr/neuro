# Scientific writing

## Latex

Use pdflatex -> biber -> pdflatex -> pdflatex to compile into out/

## Titles and headings

Use american capitalization convention for titles and section heads — headline case ("A Really Awesome Thesis").

## Structure

Tell the story of the problem and how it was solved — not an enumeration of technical facts like a report.

Bridge every section into the next with a transition; a reader should never land in a new section without knowing why it follows the last one.
End every section, subsection, and subsubsection on a sentence, never on an equation.
Nest chapters no deeper than three levels (1.2.3).
The conclusion states no new results — only a summary of the key results and, if useful, an outlook.

## Prose

No comma before a restrictive "that".
Never let "This" stand alone as a subject — name what it refers to: "This $SUBJECT is stupid" not "This is stupid."
Fold an equation into the grammar of the sentence it belongs to; never end a sentence with a colon before the equation.
Use the /unslop skill to avoid AI sounding language

## Claims and citations

Back every claim with a proof, an experiment, or a citation; a well-known keyword is enough for a standard step ("follows by partial integration").
Paraphrase rather than quote — a literal quotation is unusual in science and engineering.

## Equations

Number an equation only if the text refers back to it later; otherwise use an unnumbered environment (`equation*`). Reference a numbered equation with `\eqref{}`, and drop the word "equation" except at the start of a sentence.
Never use `\frac` inline — reserve it for displayed equations.
A bare expression is incomplete: give it a relation sign, e.g. `v(t) = gt`, not just `gt`.
Separate variables within a formula, and a value from its unit, with `\,` (e.g. `10\,\si{V}` via siunitx).
Set matrices bold and uppercase, vectors bold and lowercase, scalars italic and lowercase.
An index is italic when it counts (a running index, e.g. `x_i`); upright when it names a word or abbreviation (e.g. `U_\mathrm{rms}`); and never italic when it's a number (e.g. `x_1`).
Keep code and math notation separate in running text — write `A^{-1}b`, not `A\b`.
Introduce every variable in the surrounding text, including each element of a vector.
Use `\linebreak` rather than let a formula split awkwardly across a line.
Organize abbreviations with the `acro` package: spell one out on first use, abbreviate it from then on; don't introduce more than the reader needs, since a page thick with acronyms breaks the flow.

Always update the symbols and acronyms table

## Figures

A generated figure keeps its generating script next to it, sharing its name:
`figures/03_foundations/excitability_regimes.py` writes `excitability_regimes.pdf` into its
own directory (`Path(__file__).with_suffix(".pdf")`). Run it from the repo root with `uv
run`.

Every figure needs to have a grid on the background
Every figure floats, carries a long caption, and is referenced from the text.
Label every axis with its unit, and add a legend when a plot holds more than one curve.
Write the caption so the figure is understandable on its own; end it with a full stop.
