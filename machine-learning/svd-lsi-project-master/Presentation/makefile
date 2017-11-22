hw = slidesSVD

tex = $(shell find . -name "*.tex" -print)
bib = $(shell find . -name "*.bib" -print)

all: $(hw).pdf

$(hw).pdf: $(tex) $(bib)
	pdflatex $(hw).tex
	bibtex $(hw).aux
	pdflatex $(hw).tex
	pdflatex $(hw).tex

clean:
	rm -f *.aux *.log *.pdf *.out *.nav *.snm *.toc *.vrb *.bbl *.blg




