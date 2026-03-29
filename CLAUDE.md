# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Prerequisites**: CUDA Toolkit, Eigen3. GLFW, ImGui, GLM are auto-fetched via CMake FetchContent.

Run a single test executable:
```bash
./build/test_mesh
./build/test_topology
./build/test_constraints
./build/test_pd_constraints
```

## Rendering Simulation Results

### Option 1: ParaView (Recommended for batch rendering)

```bash
# Batch render PLY sequence
~/Applications/ParaView-6.0.1-MPI-Linux-Python3.12-x86_64/bin/pvbatch \
    scripts/render_cloth.py \
    output/ply/frame_final.ply \
    output/final.png
```

### Option 2: Matplotlib (No extra deps)

```bash
python3 scripts/ply_to_png.py output/ply/ output/png/
```

### ParaView Installation

Download from https://www.paraview.org/download/ and extract to `~/Applications/`.

## Executables

| Binary | Usage |
|--------|-------|
| `sim_cloth <rows> <cols> <size> [opts]` | GPU PD simulation, supports PLY sequence export |
| `sim_cloth ... --stretch-backend cpu-ref` | CPU reference solver (DiffCloth-aligned, uses sparse direct solve) |
| `view_cloth <rows> <cols> <size> [type]` | Static OpenGL mesh viewer |
| `gen_cloth <rows> <cols> <size> [type]` | Generate mesh and print stats |
| `cuda_ms` | Main framework (Phase 5: real-time rendering, not yet wired) |

## Architecture

This is a **CUDA Projective Dynamics (PD) cloth simulator**. PD transforms implicit time integration into an iterative constraint projection problem:

1. **Predict**: `y = x + h·v + h²·g` (inertial position, g is acceleration not force)
2. **Local Step** (fully parallel): project each constraint independently
3. **Global Step** (Jacobi): `x_new = rhs / diag`, where `diag` is a precomputed constant diagonal
4. **Apply fixed constraints**
5. **Update velocity**: `v = (x_new - x_old) / h`

**Key invariant**: `bend_k << stretch_k` — violating this causes Jacobi divergence.

### Module Layers

```
ClothMesh        — GPU buffers for geometry + simulation state (pos/vel/mass/tris/Dm_inv/rest_area)
MeshTopology     — CPU-only: edges, vert_to_tris, tri_to_tris, bending hinges
SimConstraints   — GPU buffers for constraint params + precomputed Jacobi diagonal
Constraints      — Fixed-point (pinned vertex) constraints
PDSolver         — CUDA kernel pipeline; owns temporary per-frame GPU buffers
```

**Struct layout rule**: Members of `ClothMesh` and `SimConstraints` that are shared between `.cu` and `.cpp` must be declared before any `#ifndef __CUDACC__` guards, ensuring identical memory layout in both compilation units.

### PDSolver GPU Buffers

PDSolver owns these temporary buffers (allocated once, reused each frame):
- `d_predict_` — inertial prediction y
- `d_rhs_` — RHS accumulator for Global Step
- `d_prev_pos_` — position at frame start (for velocity update)
- `d_new_pos_` — Jacobi ping-pong output (never aliased with `d_prev_pos_`)
- `d_tri_stretch_proj_` — per-triangle Stiefel projection R (3×2)
- `d_bend_proj_` — per-hinge 4-vertex projection

### Stretch Constraint (Stiefel manifold projection)

```
F = Ds · Dm_inv          (3×2 deformation gradient)
R = U · V^T              (nearest rotation via SVD of F^T·F)
p_i = A · G_col · R      (A = stiffness × rest_area, G = Dm_inv)
```

v0 accumulates `-(g00+g10)` and `-(g01+g11)` column contributions; v1/v2 accumulate individual columns.

### Bending Constraint

4-vertex hinge (v0–v1 shared edge, v2/v3 wing vertices). Projects wing vertices by rotating toward rest dihedral angle θ_rest, capped at ±10° per iteration to prevent Jacobi divergence. Only v2/v3 contribute to the Jacobi diagonal (v0/v1 projections are fixed at current positions).

### Mesh Generation (type 3 default)

Type 3 "米字格" adds a center vertex per quad → 4 triangles per quad, isotropic (no diagonal bias):
```
v0──v1
|╲ ╱|   → (v0,v1,c), (v1,v3,c), (v3,v2,c), (v2,v0,c)
| c |
|╱ ╲|
v2──v3
```

## Phase Status (v0.5)

- ✅ Phase 1–3: Core framework, GPU PD solver (stretch + bending), sim_cloth tool
- ✅ Phase Refactor: Split God-struct ClothMesh → ClothMesh / MeshTopology / SimConstraints
- 🔲 Phase 4: Chebyshev acceleration — `chebyshev_accelerate()` computes ω but blend kernel not yet written; needs `d_cheby_prev_` buffer + `x = ω·(x_jacobi - x_prev) + x_prev` kernel
- 🔲 Phase 5: CUDA–OpenGL interop for real-time rendering in `cuda_ms`
- 🔲 Phase 6: Warp-level reduce, vertex reordering, A-Jacobi unrolling
