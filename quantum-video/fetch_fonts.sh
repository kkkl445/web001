#!/usr/bin/env bash
# Download the open-licensed (SIL OFL) fonts used by the renderer into fonts/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fonts
G=https://raw.githubusercontent.com/google/fonts/main/ofl
get() { [ -f "fonts/$2" ] || curl -fsSL -o "fonts/$2" "$G/$1"; }
get "notoserifsc/NotoSerifSC%5Bwght%5D.ttf"                    NotoSerifSC.ttf
get "notosanssc/NotoSansSC%5Bwght%5D.ttf"                      NotoSansSC.ttf
get "cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf"        CormorantGaramond.ttf
get "cormorantgaramond/CormorantGaramond-Italic%5Bwght%5D.ttf" CormorantGaramond-Italic.ttf
get "jost/Jost%5Bwght%5D.ttf"                                  Jost.ttf
get "stixtwotext/STIXTwoText%5Bwght%5D.ttf"                    STIXTwoText.ttf
get "stixtwotext/STIXTwoText-Italic%5Bwght%5D.ttf"             STIXTwoText-Italic.ttf
get "notosansmath/NotoSansMath-Regular.ttf"                    NotoSansMath.ttf
echo "fonts ready"
