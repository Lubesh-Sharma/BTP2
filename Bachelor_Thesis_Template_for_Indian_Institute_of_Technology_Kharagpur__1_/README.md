# BTP report

`BTP_Report_Current_Pipeline.pdf` is the compiled report. `BTP_Report_Source.zip` contains the LaTeX sources and figure assets for editing or uploading to Overleaf, with `main.tex` as the main document.

## Document contents

- `main.tex`: institutional front matter, abstract, chapter order and bibliography.
- `Chapters/Chapter1.tex`: introduction.
- `Chapters/Chapter2.tex`: related work and attribution.
- `Chapters/Chapter3.tex`: preprocessing, orientation, ASMAE, visual features, objectives and HADM.
- `Chapters/Chapter4.tex`: dataset context and correspondence evaluation.
- `Chapters/Chapter5.tex`: conclusion followed by a paragraph-based ablation overview, with links to the detailed appendix sections.
- `Appendices/AppendixA.tex`: detailed experimental progression, placed after the bibliography.
- `Bibliography.bib`: research references.
- `Pictures/current/`: four figures in high-resolution PNG and editable draw.io formats.

The report describes module functions and mathematical relationships without numerical training settings or source revision identifiers. The original institute preamble, front-matter order, submission details and formatting are retained.

## Figures

1. `pipeline-overview`: correspondence pipeline, supervision and a schematic colour-transfer operation.
2. `preprocessing`: surface points, neighbourhood graph, spectral heat descriptors and XYZ channels.
3. `orientation-module`: pairwise encoding, angular prediction and orientation supervision.
4. `asmae-pipeline`: two-branch AAMG masking, shared encoding and cross-reconstruction in one continuous diagram.

Each figure appears in its corresponding methodology section. Labels use black text; module boxes, point clouds and feature matrices use coordinated colours. Point clouds, matrix shading and colour transfer illustrate the architecture; they are not benchmark output panels. Explanatory text is placed in the figure captions.

## Build

With a TeX distribution containing the packages required by the institute template:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively:

```bash
mkdir -p build
tectonic --keep-logs --keep-intermediates --outdir build main.tex
```

The local figure-generation and source-check scripts are in `scripts/`. Figure regeneration uses Python, NumPy, Matplotlib and the repository's input shape data. The exported figures are included in the source ZIP, so compiling the report does not require the data or project code.

## Template credit

The IIT Kharagpur template was developed by Sarthak Jain using ShareLaTeX/LaTeXTemplates. Its original attribution remains in `Thesis.cls`. Missing environment delimiters in the supplied template were repaired without changing its visual layout.
