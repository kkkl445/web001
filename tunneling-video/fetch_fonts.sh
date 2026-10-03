#!/usr/bin/env bash
# Download the open-licensed (SIL OFL) fonts used by this film into fonts/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fonts
G=https://raw.githubusercontent.com/google/fonts/main/ofl
get() { [ -f "fonts/$2" ] || curl -fsSL -o "fonts/$2" "$G/$1"; }
get "notoserifsc/NotoSerifSC%5Bwght%5D.ttf"        NotoSerifSC.ttf
get "notosanssc/NotoSansSC%5Bwght%5D.ttf"          NotoSansSC.ttf
get "intertight/InterTight%5Bwght%5D.ttf"          InterTight.ttf
get "ibmplexmono/IBMPlexMono-Regular.ttf"          IBMPlexMono-Regular.ttf
get "ibmplexmono/IBMPlexMono-Medium.ttf"           IBMPlexMono-Medium.ttf
get "stixtwotext/STIXTwoText%5Bwght%5D.ttf"        STIXTwoText.ttf
get "stixtwotext/STIXTwoText-Italic%5Bwght%5D.ttf" STIXTwoText-Italic.ttf
get "notosansmath/NotoSansMath-Regular.ttf"        NotoSansMath.ttf
echo "fonts ready"
