#!/bin/sh
# Figures of the MethodsX paper. Run after ../methodsx/make_validation_tables.py.
#   sh build_figures.sh
# R packages: ggplot2, sf, maptiles, tidyterra, patchwork. Python: pandas, pyarrow (as make_validation_tables.py).
set -e
cd "$(dirname "$0")"
python3 export_fig_data.py                 # data/*.csv, fig_numbers.tex
Rscript fig_sites_map.R                    # out/fig_sites_map.pdf    Figure 3(b)
Rscript fig_validation.R                   # out/fig_validation.pdf     validation figure (RePV)
Rscript fig_refinement.R                   # out/fig_refinement_*.pdf Figure 5(a-c)
pdflatex -interaction=nonstopmode graphical_abstract.tex >/dev/null   # graphical_abstract.pdf
# fig_configs.tex (Figure 2) and fig_selection.tex (Figure 3a) are compiled with main.tex.
