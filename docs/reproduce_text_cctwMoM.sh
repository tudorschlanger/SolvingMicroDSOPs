#/bin/bash
scriptDir="$(realpath $(dirname $0))" ; cd $scriptDir
pdflatex -shell-escape -output-format dvi cctwMoM
bibtex -terse cctwMoM
pdflatex -shell-escape -output-format dvi cctwMoM
pdflatex -shell-escape -output-format dvi cctwMoM
rm -f economics.bib; make4ht  --utf8 --config cctwMoM.cfg --format html5 cctwMoM "svg            "   "-cunihtf -utf8"
