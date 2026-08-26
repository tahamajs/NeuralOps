# =========================================================================
# PROFESSIONAL LATEX MAKEFILE FOR NEURALOPS TECHNICAL REPORT
# =========================================================================

DOC = main
LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -interaction=nonstopmode -synctex=1 -file-line-error

.PHONY: all clean fullclean watch

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex preamble.tex title.tex references.bib sections/*.tex appendices/*.tex
	$(LATEXMK) $(LATEXMK_FLAGS) $(DOC).tex

watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(DOC).tex

clean:
	$(LATEXMK) -c
	rm -f *.bbl *.blg *.aux *.log *.out *.toc *.lof *.lot *.synctex.gz *.fls *.fdb_latexmk
	rm -rf sections/*.aux sections/*.log appendices/*.aux appendices/*.log

fullclean: clean
	rm -f $(DOC).pdf
