#!/bin/bash
# Batch run low-resolution cloth simulations

set -e

PVBATCH=/home/zcy/Applications/ParaView-6.0.1-MPI-Linux-Python3.12-x86_64/bin/pvbatch
SIM=./build/src/sim_cloth
RENDER_SCRIPT=./scripts/render_cloth.py

# Test configurations: "rows cols size name"
configs=(
    "1 1 1.0 1x1"
    "2 1 1.0 2x1"
    "2 2 1.0 2x2"
    "3 3 1.0 3x3"
    "5 5 0.5 5x5"
)

for config in "${configs[@]}"; do
    read -r rows cols size name <<< "$config"

    echo "========================================"
    echo "Running $name (rows=$rows, cols=$cols)"
    echo "========================================"

    outdir="output/test_${name}"
    mkdir -p "$outdir"

    # Run simulation
    $SIM $rows $cols $size \
        --steps 50 \
        --stretch-backend cpu-ref \
        --export "$outdir" \
        --iter 50 \
        --bend 0.01 \
        --dt 0.01 \
        2>&1 | tee "$outdir/log.txt"

    # Render final frame with ParaView
    if [ -f "$outdir/frame_final.ply" ]; then
        echo "Rendering with ParaView..."
        $PVBATCH $RENDER_SCRIPT "$outdir/frame_final.ply" "$outdir/final.png" 2>&1 || echo "ParaView render failed"
    fi

    echo ""
done

echo "All tests complete. Check output/test_*/"
