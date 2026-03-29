#!/usr/bin/env pvbatch
"""
ParaView batch script to render cloth PLY files.
Usage: pvbatch render_cloth.py <ply_file> <output_png>
"""
from paraview.simple import *
import sys

def render_cloth(ply_path, png_path):
    # Reset session
    Disconnect()
    Connect()

    # Load PLY file using PLYReader
    reader = PLYReader(FileNames=[ply_path])

    # Get active view
    view = GetActiveViewOrCreate('RenderView')
    view.ViewSize = [1280, 720]

    # Display the mesh
    display = Show(reader, view)
    display.Representation = 'Surface With Edges'
    display.ColorArrayName = [None, '']
    display.DiffuseColor = [0.7, 0.8, 1.0]  # Light blue
    display.EdgeColor = [0.1, 0.2, 0.4]     # Dark blue edges
    display.LineWidth = 1.5

    # Set camera
    view.CameraViewUp = [0, 1, 0]
    view.CameraFocalPoint = [0, 0, 0]
    view.CameraPosition = [5, 2, 5]
    view.ResetCamera()

    # Set background
    view.Background = [0.95, 0.95, 0.95]

    # Render and save
    Render()
    SaveScreenshot(png_path, view, ImageResolution=[1280, 720])
    print(f"Saved: {png_path}")

    # Cleanup
    Delete(reader)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: pvbatch {sys.argv[0]} <ply_file> <output_png>")
        sys.exit(1)

    render_cloth(sys.argv[1], sys.argv[2])
