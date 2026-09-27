#!/bin/sh
# Sequential widths; numerical failures remain recorded, never replaced.
set -eu
: "${SDPX_CLI:?absolute native sdpx executable required}"
: "${SDPX_SOURCE_ID:?source commit plus dirty patch hash required}"
: "${SDPX_OUTPUT_DIR:?new output directory required}"
test -x "$SDPX_CLI" || { echo "SDPX_CLI is not executable" >&2; exit 2; }
JULIA=${JULIA:-julia}
export SDPX_CLI SDPX_SOURCE_ID
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir "$SDPX_OUTPUT_DIR" # Refuse an existing directory, including partial runs.
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
failed=0
for width in 1 2 4 8; do
    export SDPX_THREADS=$width RAYON_NUM_THREADS=$width
    export SDPX_OUTPUT="$SDPX_OUTPUT_DIR/width-$width.json"
    if "$JULIA" --startup-file=no -t1 --gcthreads=1 \
        "$script_dir/orthant.jl" >"$SDPX_OUTPUT_DIR/width-$width.log" 2>&1; then
        printf '%s\n' "width $width passed"
    else
        failed=1
        printf '%s\n' "width $width failed; inspect retained log/receipt" >&2
    fi
done
exit "$failed"
