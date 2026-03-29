#!/usr/bin/env python3
"""
Render PLY files to PNG with better quality using pyvista.
Install: pip install pyvista
"""
import sys
import glob
import os

def render_with_pyvista(ply_path, png_path):
    import pyvista as pv

    mesh = pv.read(ply_path)

    plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
    plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='darkblue')
    plotter.camera_position = 'iso'
    plotter.add_title(os.path.basename(ply_path))
    plotter.screenshot(png_path)
    print(f"Saved: {png_path}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ply_file_or_directory> [output.png]")
        sys.exit(1)

    input_path = sys.argv[1]

    if input_path.endswith('.ply'):
        out_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.ply', '.png')
        render_with_pyvista(input_path, out_path)
    else:
        ply_files = sorted(glob.glob(f"{input_path}/frame_*.ply"))
        if not ply_files:
            ply_files = sorted(glob.glob(f"{input_path}/*.ply"))

        out_dir = sys.argv[2] if len(sys.argv) > 2 else f"{input_path}_png"
        os.makedirs(out_dir, exist_ok=True)

        for ply in ply_files:
            basename = os.path.basename(ply).replace('.ply', '.png')
            render_with_pyvista(ply, f"{out_dir}/{basename}")

        print(f"\nAll frames saved to: {out_dir}/")

if __name__ == "__main__":
    main()
