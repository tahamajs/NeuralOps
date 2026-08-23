# =========================================================================
# PROFESSIONAL LATEX MAKEFILE
# Uses latexmk to automatically handle multiple compilation passes,
# bibliography generation, and cross-referencing.
# =========================================================================

# Main document filename (without extension)
DOC = main

# LaTeX compiler options
LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -interaction=nonstopmode -synctex=1 -file-line-error

.PHONY: all clean fullclean watch

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex preamble.tex title.tex references.bib sections/*.tex
	$(LATEXMK) $(LATEXMK_FLAGS) $(DOC).tex

watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(DOC).tex

clean:
	$(LATEXMK) -c
	rm -f *.bbl *.blg *.aux *.log *.out *.toc *.lof *.lot *.synctex.gz *.fls *.fdb_latexmk
	rm -rf sections/*.aux sections/*.log

fullclean: clean
	rm -f $(DOC).pdf
