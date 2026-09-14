#!/bin/sh
# Sequential widths; numerical failures remain recorded, never replaced.
set -eu
: "${SDPX_LIBRARY:?explicit frozen library required}"
: "${SDPX_EXPECTED_SOURCE:?explicit frozen Julia package root required}"
: "${SDPX_SOURCE_ID:?source commit plus dirty patch hash required}"
: "${SDPX_OUTPUT_DIR:?new output directory required}"
SDPX_PROJECT=${SDPX_PROJECT:-/tmp/sdpx-rust-acceptance/float64-env-16a}
JULIA=${JULIA:-julia}
export SDPX_LIBRARY SDPX_EXPECTED_SOURCE SDPX_SOURCE_ID
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir "$SDPX_OUTPUT_DIR" # Refuse an existing directory, including partial runs.
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
failed=0
for width in 1 2 4 8; do
    export SDPX_THREADS=$width RAYON_NUM_THREADS=$width
    export SDPX_OUTPUT="$SDPX_OUTPUT_DIR/width-$width.json"
    if "$JULIA" --startup-file=no --project="$SDPX_PROJECT" -t1 --gcthreads=1 \
        "$script_dir/orthant.jl" >"$SDPX_OUTPUT_DIR/width-$width.log" 2>&1; then
        printf '%s\n' "width $width passed"
    else
        failed=1
        printf '%s\n' "width $width failed; inspect retained log/receipt" >&2
    fi
done
exit "$failed"
