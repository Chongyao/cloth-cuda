#!/usr/bin/env python3
"""
Render PLY files to PNG images using matplotlib (no extra deps beyond numpy/matplotlib).
"""
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def parse_ply(filepath):
    """Parse a simple ASCII PLY file."""
    vertices = []
    faces = []
    with open(filepath, 'r') as f:
        # Header
        line = f.readline().strip()
        assert line == "ply", f"Not a PLY file: {filepath}"

        num_verts = None
        num_faces = None
        in_header = True

        while in_header:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                num_verts = int(line.split()[2])
            elif line.startswith("element face"):
                num_faces = int(line.split()[2])
            elif line == "end_header":
                in_header = False

        # Read vertices
        for _ in range(num_verts):
            parts = f.readline().strip().split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

        # Read faces
        for _ in range(num_faces):
            parts = f.readline().strip().split()
            n = int(parts[0])
            faces.append([int(p) for p in parts[1:1+n]])

    return np.array(vertices), faces

def render_ply_to_png(ply_path, png_path, elev=30, azim=45):
    """Render a PLY file to PNG with a fixed camera angle."""
    verts, faces = parse_ply(ply_path)

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh polygons
    polys = [[verts[idx] for idx in face] for face in faces]

    # Add mesh collection with lighting-like coloring
    poly3d = Poly3DCollection(polys, alpha=0.9, facecolor='lightblue',
                               edgecolor='darkblue', linewidth=0.3)
    ax.add_collection3d(poly3d)

    # Auto-scale
    ax.set_xlim(verts[:,0].min(), verts[:,0].max())
    ax.set_ylim(verts[:,1].min(), verts[:,1].max())
    ax.set_zlim(verts[:,2].min(), verts[:,2].max())

    # Equal aspect ratio
    max_range = np.array([verts[:,0].max()-verts[:,0].min(),
                          verts[:,1].max()-verts[:,1].min(),
                          verts[:,2].max()-verts[:,2].min()]).max() / 2.0
    mid_x = (verts[:,0].max()+verts[:,0].min()) * 0.5
    mid_y = (verts[:,1].max()+verts[:,1].min()) * 0.5
    mid_z = (verts[:,2].max()+verts[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame: {ply_path.split("/")[-1]}')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ply_file_or_directory> [output.png]")
        print(f"       {sys.argv[0]} output/ply/          # batch convert all PLY files")
        sys.exit(1)

    input_path = sys.argv[1]

    if input_path.endswith('.ply'):
        # Single file
        out_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.ply', '.png')
        render_ply_to_png(input_path, out_path)
    else:
        # Directory batch mode
        import os
        ply_files = sorted(glob.glob(f"{input_path}/frame_*.ply"))
        if not ply_files:
            ply_files = sorted(glob.glob(f"{input_path}/*.ply"))

        out_dir = sys.argv[2] if len(sys.argv) > 2 else f"{input_path}_png"
        os.makedirs(out_dir, exist_ok=True)

        for ply in ply_files:
            basename = os.path.basename(ply).replace('.ply', '.png')
            render_ply_to_png(ply, f"{out_dir}/{basename}")

        print(f"\nAll frames saved to: {out_dir}/")

if __name__ == "__main__":
    main()
