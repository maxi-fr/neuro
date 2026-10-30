# Figures

Design figures at their final physical size and insert them at native size: `\input{...}` for TikZ and `\includegraphics{...}` for generated PDFs. Fit the complete figure, including labels and strokes, within the available `\linewidth`; change the layout rather than scaling the finished figure. Keep explicit width scaling for external artwork such as logos only.

Use the thesis's XCharter roman text and Computer Modern mathematics. Primary figure text is 9 TeX pt (`\footnotesize` in this 11 pt document); secondary annotations, ticks, and legends are 8 TeX pt (`\scriptsize`). These are thesis design conventions, not template requirements. Mathematical subscripts retain normal TeX sizing.

For TikZ, place `\tikzsetnextfilename{<figure_stem>}` immediately before `\begin{tikzpicture}`. Apply `thesis figure` to each picture, `thesis heading` to bold headings, and `thesis secondary` to secondary labels. A later `font=...` replaces the complete font setting, so use the shared styles to retain the intended size. Set node widths, wrapping, and spacing to fit; preserve the text sizes when adjusting geometry.

For Python, use `figures/style.py` and its size/font constants. Size the entire canvas with `style.figsize(...)`, use constrained layout, and preserve the canvas with `savefig.bbox = None`. Generate narrower figures at the intended width ratio rather than shrinking them during inclusion.
