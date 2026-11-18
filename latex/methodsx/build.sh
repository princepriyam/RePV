#!/bin/sh
# PDF only. main.tex and supplement.tex cross-reference each other through xr, so the
# supplement is built before and after the article. Outputs go to build/.
set -e
cd "$(dirname "$0")"
latexmk -pdf -bibtex -interaction=nonstopmode -outdir=build supplement.tex
latexmk -pdf -bibtex -interaction=nonstopmode -outdir=build main.tex
latexmk -g -pdf -bibtex -interaction=nonstopmode -outdir=build supplement.tex
cp build/main.pdf build/supplement.pdf .
