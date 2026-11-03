#!/usr/bin/env bash
# Profile the HAR-kNN program with gprof and gcov (checkpoint 1.1).
#
# Usage (from anywhere, in Git Bash / Linux):
#   scripts/profile.sh [SCENARIO] [K] [DT]
#     SCENARIO: wisdm | pamap2   (default: wisdm)
#     K:        3 | 5 | 7 | 20   (default: 3)
#     DT:       1 = double, 2 = float (default: 1)
#   Env overrides: OPT (default -O2), SPECIALIZED (1), USE_SQRT (1),
#                  VTUNE (path to vtune; 0 = skip VTune)
#
# Outputs go to results/profiling/<scenario>_k<K>_<type>/:
#   program_output.txt   - program stdout (accuracy, timings)
#   gprof_<OPT>.txt      - gprof flat profile + call graph (build with OPT)
#   gprof_O0.txt         - gprof at -O0, no inlining -> complete call graph
#   gcov/*.gcov          - line-by-line execution counts (-O0 --coverage)
#   gcov_summary.txt     - per-file / per-function coverage and branch summary
#   vtune_summary.txt    - VTune hotspots summary (build with OPT -g, no -pg)
#   vtune_hotspots.csv   - VTune per-function CPU time
#   vtune_topdown.txt    - VTune call tree (compare with the gprof call graph)
set -euo pipefail

SCENARIO=${1:-wisdm}
K=${2:-3}
DT=${3:-1}
OPT=${OPT:--O2}
SPECIALIZED=${SPECIALIZED:-1}
USE_SQRT=${USE_SQRT:-1}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/knn/src
SRCS=(knn.c utils.c timer.c io.c features.c main.c)

case $SCENARIO in
  wisdm)  NUM_FEATURES=43; SCENARIO_FEATURES=1 ;;
  pamap2) NUM_FEATURES=90; SCENARIO_FEATURES=2 ;;
  *) echo "unknown scenario: $SCENARIO" >&2; exit 1 ;;
esac
DATA=$ROOT/data/$SCENARIO
[ "$DT" = 2 ] && TYPE=float || TYPE=double

OUT=$ROOT/results/profiling/${SCENARIO}_k${K}_${TYPE}
WORK=$ROOT/knn/run/${SCENARIO}_k${K}_${TYPE}
rm -rf "$OUT" "$WORK"
mkdir -p "$OUT/gcov" "$WORK"

# Same defines as knn/Makefile (READ=4)
DEFS=(-I"$DATA" -Wall -Wextra -std=gnu99
  -DK="$K" -DDT="$DT" -DSPECIALIZED="$SPECIALIZED" -DMATH_TYPE=1 -DDIST_METHOD=1
  -DUSE_SQRT="$USE_SQRT" -DNUM_FEATURES=$NUM_FEATURES -DSCENARIO_FEATURES=$SCENARIO_FEATURES
  -DREAD=4 -DDIMEM=0 -DVERIFY=0 -DSTREAMING=1 -DNORMALIZE=1
  -DTRAIN_PATH="\"$DATA/train2_norm.dat\"" -DTEST_PATH="\"$DATA/test2_notnorm.dat\"")

# -no-pie is required on MinGW-w64: with PIE/ASLR gprof reports an empty profile.
PG_FLAGS=(-g -pg)
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) PG_FLAGS+=(-no-pie) ;; esac

run_gprof() {  # $1 = label, rest = compiler flags
  local label=$1; shift
  local dir=$WORK/gprof_$label
  mkdir -p "$dir"
  ( cd "$SRC" && gcc "$@" "${DEFS[@]}" "${SRCS[@]}" -o "$dir/harknn.exe" -lm "${PG_FLAGS[@]}" )
  ( cd "$dir" && ./harknn.exe > program_output.txt && gprof -b harknn.exe gmon.out > "$OUT/gprof_$label.txt" )
}

echo "== $SCENARIO k=$K $TYPE: gprof ($OPT)"
run_gprof "${OPT#-}" $OPT
grep -v '^class id' "$WORK/gprof_${OPT#-}/program_output.txt" > "$OUT/program_output.txt"

echo "== $SCENARIO k=$K $TYPE: gprof (-O0, no inlining)"
run_gprof O0 -O0 -fno-inline

echo "== $SCENARIO k=$K $TYPE: gcov (-O0 --coverage)"
GC=$WORK/gcov
mkdir -p "$GC"
for f in "${SRCS[@]}"; do
  gcc -O0 --coverage "${DEFS[@]}" -c "$SRC/$f" -o "$GC/${f%.c}.o"
done
gcc --coverage "$GC"/*.o -o "$GC/harknn.exe" -lm
( cd "$GC" && ./harknn.exe > /dev/null )
( cd "$OUT/gcov" && gcov -b -c -f -o "$GC" "${SRCS[@]/#/$SRC/}" > "$OUT/gcov_summary.txt" )

# VTune is a native Windows program: give it Windows paths (cygpath is absent on Linux).
winpath() { cygpath -w "$1" 2>/dev/null || echo "$1"; }
VTUNE=${VTUNE:-$(command -v vtune || echo "/c/Program Files (x86)/Intel/oneAPI/vtune/latest/bin64/vtune.exe")}
if [ "$VTUNE" != 0 ] && [ -x "$VTUNE" ]; then
  echo "== $SCENARIO k=$K $TYPE: VTune hotspots ($OPT -g)"
  VT=$WORK/vtune
  mkdir -p "$VT"
  ( cd "$SRC" && gcc $OPT -g "${DEFS[@]}" "${SRCS[@]}" -o "$VT/harknn.exe" -lm )
  R=$(winpath "$VT/r_hotspots")
  ( cd "$VT" && "$VTUNE" -collect hotspots -r "$R" -- "$(winpath "$VT/harknn.exe")" > collect.log 2>&1 )
  "$VTUNE" -report summary  -r "$R" -report-output "$(winpath "$OUT/vtune_summary.txt")"
  "$VTUNE" -report hotspots -r "$R" -format csv -csv-delimiter comma \
    -report-output "$(winpath "$OUT/vtune_hotspots.csv")"
  "$VTUNE" -report top-down -r "$R" -report-output "$(winpath "$OUT/vtune_topdown.txt")"
else
  echo "== VTune not found, skipping (set VTUNE=/path/to/vtune)"
fi

echo "== done -> ${OUT#$ROOT/}"
sed -n '/accuracy/p;/time/p' "$OUT/program_output.txt"
sed -n '3,12p' "$OUT/gprof_${OPT#-}.txt"
