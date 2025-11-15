RESULT_DIR = ../result
LATEX = pdflatex -synctex=1 -shell-escape --enable-write18 -interaction=nonstopmode
BIBTEX = biber
MAKEINDEX = makeindex
MAKEGLOSSARY = makeglossaries

# Abbreviations are enabled by default, glossary and symbols commented out
INCLUDES = chapters/abbreviations.tex abstract.tex docinfo.tex preamble.tex titlepage.sty literature.bib images/*.pdf hma.cls
# Glossary and symbols support commented out for now
# chapters/glossary.tex chapters/symbols.tex

# Appendices commented out for now
# CHAPTERS = chapters/introduction.tex chapters/related_work.tex chapters/methods.tex chapters/results.tex chapters/discussion.tex chapters/conclusion.tex chapters/appendix*.tex
CHAPTERS = chapters/introduction.tex chapters/related_work.tex chapters/methods.tex chapters/results.tex chapters/discussion.tex chapters/conclusion.tex

TEMP_FILES = *.slg *.sls *.glg *.gls *.acr *.alg  *.bbl *.lol *.fdb_latexmk *.log *.synctex.gz *.fls *.idx *.lof *.lol *.lot *.toc *.acn *.glo *.run.xml *.slo *.syg *.bcf *.mw *.out *.aux *.glsdefs thesis.ist *.bbl *.blg *.ilg *.ind *.loc *.soc *.syi thesis.pdf thesis.run.xml thesis.synctex.gz

define latex-it
$(eval FILE = $(firstword $^))
$(eval BASENAME = $(basename $(FILE)))
@echo "Building $(FILE)"
@echo $(FILE)
@$(LATEX) -draftmode $(FILE)
$(MAKEINDEX) -s $(BASENAME).ist -t $(BASENAME).alg -o  $(BASENAME).acr $(BASENAME).acn
$(MAKEINDEX) -s  $(BASENAME).ist -t $(BASENAME).glg -o  $(BASENAME).gls $(BASENAME).glo
$(MAKEINDEX) -s  $(BASENAME).ist -t $(BASENAME).slg -o  $(BASENAME).syi $(BASENAME).syg
$(BIBTEX) $(BASENAME)
@$(LATEX) -draftmode $(FILE) > /dev/null
@$(LATEX) -draftmode -interaction batchmode $(FILE) > /dev/null
@$(LATEX) -interaction batchmode $(FILE)
endef

.PHONY: all
all: $(RESULT_DIR)/paper.pdf

.PHONY: clean
clean:
	rm -rf $(RESULT_DIR)
	rm -f $(TEMP_FILES)

../thesis-overleaf.zip: $(INCLUDES) $(CHAPTERS) thesis.tex
	rm -f $(TEMP_FILES)
	rm $@
	zip -r $@ .

thesis.pdf: thesis.tex $(INCLUDES) $(CHAPTERS)
	$(latex-it)

$(RESULT_DIR)/paper.pdf: thesis.pdf
	mkdir -p $(RESULT_DIR)
	@cp $< $@
