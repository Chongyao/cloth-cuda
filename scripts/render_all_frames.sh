#!/bin/bash
# Render all PLY frames in a directory

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ply_dir> <output_dir>"
    exit 1
fi

PLY_DIR=$1
OUT_DIR=$2
PVBATCH=/home/zcy/Applications/ParaView-6.0.1-MPI-Linux-Python3.12-x86_64/bin/pvbatch
RENDER_SCRIPT=/home/zcy/workspace/projects/cuda-ms/scripts/render_cloth.py

mkdir -p "$OUT_DIR"

# Render each frame
for ply in "$PLY_DIR"/frame_*.ply; do
    [ -f "$ply" ] || continue
    basename=$(basename "$ply" .ply)
    echo "Rendering $basename..."
    $PVBATCH "$RENDER_SCRIPT" "$ply" "$OUT_DIR/${basename}.png" 2>/dev/null
done

echo "All frames rendered to $OUT_DIR/"
