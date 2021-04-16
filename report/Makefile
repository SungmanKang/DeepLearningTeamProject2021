
PACKAGE=dl21

PDF = $(PACKAGE).pdf

all:  ${PDF}

%.pdf:  %.tex
	xelatex $<
	- bibtex $*
	xelatex $<
	- makeindex -s gind.ist -o $*.ind $*.idx
	- makeindex -s gglo.ist -o $*.gls $*.glo
	xelatex $<
	while ( grep -q '^LaTeX Warning: Label(s) may have changed' $*.log) \
	do xelatex $<; done

clean:
	$(RM)  *.pdf *.aux *.log

.PHONY: all docclean clean archive zip
