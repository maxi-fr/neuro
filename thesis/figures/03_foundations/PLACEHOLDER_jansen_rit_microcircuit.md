# Placeholder: `jansen_rit_microcircuit`

Referenced from `chapters/03_foundations.tex` as `\ref{fig:microcircuit}`. The figure is
currently drawn inline with TikZ in that file; replace the TikZ block with a
`\includegraphics{figures/03_foundations/jansen_rit_microcircuit}` if it is redrawn
externally.

What it has to show — the classic Jansen-Rit block diagram:

- Three population blocks: pyramidal cells (PY), excitatory interneurons (eIN),
  inhibitory interneurons (iIN).
- Each block is a pulse-to-wave PSP filter ($h_\mathrm{e}$ or $h_\mathrm{i}$) followed by a
  wave-to-pulse sigmoid $S(\cdot)$.
- Arrows labelled with the four intra-columnar contact numbers $c_1$ to $c_4$:
  PY $\xrightarrow{c_1}$ eIN, eIN $\xrightarrow{c_2}$ PY, PY $\xrightarrow{c_3}$ iIN,
  iIN $\xrightarrow{c_4}$ PY (the last one inhibitory, drawn with a filled circle head).
- The summing junction at the pyramidal soma forming $v = x_2 - x_3$, with the stimulation
  drive $s_i$ added at the same junction.
- The background input $I$ and the network coupling $I_{\mathrm{coup},i}$ entering the
  excitatory PSP branch, and the noise $\zeta_i$ on the same branch.
- The output tap $y_{\mathrm{LFP},i} = x_{i,2} - x_{i,3}$.
