## Local latexmk configuration for this thesis.
##
## Portability note: this file is safe to commit / upload to Overleaf.
## The biber override below only activates on a machine where the thinned
## biber copy exists (i.e. this Mac). On Overleaf/Linux the file is absent,
## so latexmk falls back to the system biber and nothing changes.
##
## Background: on macOS 26/27 the newer `lipo` CLI breaks biber's universal
## self-extraction launcher ("extracting arm64 binary with lipo failed").
## We point latexmk at an arch-thinned biber that runs natively.
## Regenerate it if biber is ever updated:
##   lipo /Library/TeX/texbin/biber -thin arm64 -output ~/.local/bin/biber

my $local_biber = "$ENV{HOME}/.local/bin/biber";
if (-x $local_biber) {
    $biber = "$local_biber %O %S";
}

## Build a PDF via pdflatex + biber, rerunning as needed.
$pdf_mode = 1;
