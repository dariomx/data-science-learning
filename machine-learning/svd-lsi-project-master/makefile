hw = lsi

tex = $(shell find . -name "*.tex" -print)

all: $(hw).pdf

$(hw).pdf: $(tex) $(hw).bib
	pdflatex $(hw).tex
	bibtex $(hw).aux
	pdflatex $(hw).tex
	pdflatex $(hw).tex

clean:
	rm -f *.aux *.log *.pdf *.out *.bbl *.blg


