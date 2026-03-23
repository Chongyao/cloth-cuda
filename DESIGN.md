# CUDA Projective Dynamics 布料仿真框架设计文档

> 版本：v0.5
> 目标：基于 CUDA 的 Projective Dynamics 布料仿真，暂不处理碰撞

---

## 一、理论基础概述

### 1.1 Projective Dynamics 核心思想

PD 将隐式积分中的能量最小化问题转化为**约束投影问题**：

```
x_{n+1} = argmin  (1/2h²)||x - y||²_M + Σ w_i ||A_i x - p_i||²

其中 y = x_n + h·v_n + h²·g  (惯性预测位置，g 为重力加速度)
```

- **Local Step**: 对每个约束计算投影 `p_i = projection(A_i x)` —— 完全并行
- **Global Step**: 求解固定线性系统 `(M + h²L)x = M y + h² Σ w_i A_i^T p_i`

系统矩阵对角线 `diag_i = m_i + h²·Σ w_c` 是**常数**，预计算后每帧只需 Jacobi 除法。

### 1.2 时间积分流程

```
每帧:
1. 预测: y = x_n + h·v_n + h²·g        ← g 是加速度，不除质量
2. for iter = 0..max_iter:
   a. Local Step  (并行): p_c = project_constraint(x)
   b. Global Step (Jacobi): rhs = M·y + h²·Σ w_c·p_c
                             x_new = rhs / diag
   c. Chebyshev 加速 (TODO Phase 4)
   d. 应用固定约束
3. 速度: v_{n+1} = (x_{n+1} - x_n) / h
```

### 1.3 Jacobi 迭代公式

对每个顶点 `i`：
```
x_i^{new} = (m_i·y_i + h²·Σ_{c∈C_i} w_c·p_c^i) / (m_i + h²·Σ_{c∈C_i} w_c)
```

分母 `diag_i = m_i + h²·Σ w_c` 是常数，预计算一次存储。

### 1.4 Chebyshev 加速（待实现）

```
ω_1 = 1,  ω_2 = 2/(2 - ρ²)   (ρ ≈ 0.9~0.99，Jacobi 谱半径估计)
k ≥ 2:  ω_{k+1} = 4/(4 - ρ²·ω_k)

x^{k+1} = ω_{k+1}·(Jacobi(x^k) - x^{k-1}) + x^{k-1}
```

当前 `chebyshev_accelerate()` 已计算 ω 序列，但混合更新尚未实现（TODO）。

### 1.5 布料约束类型

| 约束 | 投影算子 | 实现状态 |
|------|----------|---------|
| 三角形拉伸 | Stiefel 流形投影（SVD 近似旋转 R） | ✅ |
| 弯曲（二面角） | 4 顶点绕共享边旋转至 rest 角度 | ✅ |

**三角形拉伸投影（Stiefel manifold）**：
```
F = Ds · Dm_inv        (变形梯度, 3×2)
R = U · V^T            (via SVD of F^T·F，取最近旋转)
p_0 = -A·((g00+g10)R0 + (g01+g11)R1)   (v0 贡献)
p_1 =  A·(g00·R0 + g01·R1)             (v1 贡献)
p_2 =  A·(g10·R0 + g11·R1)             (v2 贡献)
```
其中 A = stiffness × rest_area，Dm_inv = G = [[g00,g01],[g10,g11]]。

**弯曲投影**：
计算共享边两侧翼顶点的当前二面角 θ，将差值 Δ = θ − θ_rest 对称地分配给两侧翼顶点，单步旋转上限 ±10°（防 Jacobi 发散）。

> ⚠️ **收敛条件**：Jacobi-PD 要求 `bend_k << stretch_k`，否则迭代发散。

---

## 二、工程现状与文件结构

### 2.1 模块分层

重构后（v0.5）将原 God struct `ClothMesh` 拆为三层职责清晰的模块：

```
include/
  cloth_mesh.h       ← 几何 + 仿真状态 GPU buffer
  mesh_topology.h    ← CPU-only 拓扑分析
  sim_constraints.h  ← 约束数据 + Jacobi 对角线 GPU buffer
  constraints.h      ← 固定点（pinned vertex）约束
  pd_solver.h        ← PDSolverConfig + PDSolver
  mesh_generator.h   ← 网格生成
  utils/cuda_helper.h

src/
  cloth_mesh.cpp     ← ClothMesh 实现
  mesh_topology.cpp  ← MeshTopology 实现
  sim_constraints.cpp← SimConstraints 实现
  constraints.cpp    ← Constraints 实现
  mesh_generator.cpp ← 网格生成实现
  pd_solver.cu       ← 全部 CUDA kernel + PDSolver
  tools/
    sim_cloth.cpp    ← 命令行仿真工具
    gen_cloth.cpp    ← 网格生成工具
    view_cloth.cpp   ← 静态 OpenGL 查看器
  viewer/            ← GLFW + GLAD + ImGui 查看器库
  tests/             ← 单元测试
```

### 2.2 各模块职责

**`ClothMesh`**（`cloth_mesh.h`）
- GPU buffer：`d_pos`, `d_vel`, `d_mass`（仿真状态）；`d_tris`, `d_Dm_inv`, `d_rest_area`（FEM 几何）
- CPU 数据：`rest_pos`, `triangles`, `Dm_inv`, `rest_area`, `mass`
- 方法：`load_obj`, `precompute_rest_state`, `upload_to_gpu`, `free_gpu`

**`MeshTopology`**（`mesh_topology.h`，CPU-only）
- 数据：`edges`（含 tri_a/tri_b/rest_len）、`vert_to_tris`、`tri_to_tris`
- 工厂：`MeshTopology::build(mesh)` — 单次构建，O(T)

**`SimConstraints`**（`sim_constraints.h`）
- GPU buffer：`d_tri_stretch_k`, `d_jacobi_diag`, `d_bend_quads/rest/k`
- CPU build 方法：`build_stretch`, `build_bend`, `precompute_jacobi_diag`
- `upload_to_gpu()`, `free_gpu()`

**`PDSolver`**（`pd_solver.h`）
- 构造：`PDSolver(config, mesh, sim_cons)` — 分配临时 GPU buffer
- 步进：`step(mesh, sim_cons, pin_cons)` — 每帧调用一次

### 2.3 典型使用流程

```cpp
// 1. 构建 mesh
ClothMesh mesh;
generate_square_cloth(rows, cols, size, /*type=*/3, mesh);
mesh.precompute_rest_state(density);

// 2. 拓扑分析（CPU，一次性）
MeshTopology topo = MeshTopology::build(mesh);

// 3. 构建约束
SimConstraints sim_cons;
sim_cons.build_stretch(mesh, stretch_k);
sim_cons.build_bend(mesh, topo, bend_k);   // 可选，需 bend_k << stretch_k

// 4. 上传 GPU
mesh.upload_to_gpu();
sim_cons.upload_to_gpu();
sim_cons.precompute_jacobi_diag(mesh, dt); // 必须在 upload 之后调用

// 5. 固定约束
Constraints pin_cons;
pin_cons.pin_top_row(mesh, ncols);
pin_cons.upload_to_gpu();

// 6. 求解器
PDSolverConfig cfg;  cfg.dt = dt;  cfg.max_iterations = 50;
PDSolver solver(cfg, mesh, sim_cons);

// 7. 仿真循环
solver.step(mesh, sim_cons, pin_cons);
```

### 2.4 可执行文件

| 程序 | 说明 |
|------|------|
| `sim_cloth <rows> <cols> <size> [opts]` | GPU PD 仿真，支持 PLY 序列导出 |
| `view_cloth <rows> <cols> <size> [type]` | 静态网格 OpenGL 查看器 |
| `gen_cloth <rows> <cols> <size> [type]` | 生成网格并打印统计 |
| `cuda_ms` | 主程序框架（Phase 5 接入实时渲染） |

---

## 三、网格生成

`generate_square_cloth(nrows, ncols, size, type, mesh)`

| type | 说明 | 顶点数 | 三角形数 |
|------|------|--------|---------|
| 0 | 统一 `\` 对角线 | R·C | 2·(R-1)(C-1) |
| 1 | 棋盘格 `\/` 交替 | R·C | 2·(R-1)(C-1) |
| 2 | 统一 `/` 对角线 | R·C | 2·(R-1)(C-1) |
| **3** | **米字格（默认）** | R·C+(R-1)(C-1) | 4·(R-1)(C-1) |

**米字格（type 3）**：每个 quad 中心增加一个顶点，各向同性，无对角线方向偏差：
```
v0──v1
|╲ ╱|   → (v0,v1,c), (v1,v3,c), (v3,v2,c), (v2,v0,c)
| c |
|╱ ╲|
v2──v3
```

---

## 四、PD Solver 实现细节

### 4.1 GPU 缓冲区分工

```
PDSolver 私有缓冲区（每帧临时）：
  d_predict_          [N×3]   惯性预测位置 y
  d_rhs_              [N×3]   Global Step RHS 累加器
  d_prev_pos_         [N×3]   帧开始时旧位置，用于 v = (x_new - x_old)/h
  d_new_pos_          [N×3]   Jacobi ping-pong 输出缓冲
  d_tri_stretch_proj_ [T×6]   Stiefel 投影 R（每三角形 3×2 矩阵）
  d_bend_proj_        [E_b×4×3] 弯曲约束四顶点投影

ClothMesh GPU 数据（持久，仿真状态）：
  d_pos, d_vel, d_mass        当前运动状态
  d_tris, d_Dm_inv, d_rest_area  FEM 几何

SimConstraints GPU 数据（持久，约束参数）：
  d_tri_stretch_k [T]         每三角形拉伸刚度
  d_jacobi_diag   [N]         预计算系统对角线
  d_bend_quads    [E_b×4]     弯曲约束顶点四元组
  d_bend_rest     [E_b]       静止二面角
  d_bend_k        [E_b]       弯曲刚度
```

### 4.2 已实现 CUDA Kernels

| Kernel | 线程粒度 | 功能 |
|--------|---------|------|
| `predict_kernel` | 每顶点 | `y = x + h·v + h²·g` |
| `tri_stretch_project_kernel` | 每三角形 | SVD 求最近旋转 R |
| `accumulate_tri_stretch_rhs_kernel` | 每三角形 | atomicAdd 拉伸 RHS |
| `bend_project_kernel` | 每内边 | 翼顶点旋转投影 |
| `accumulate_bend_rhs_kernel` | 每内边 | atomicAdd 弯曲 RHS（仅 v2/v3）|
| `clear_rhs_kernel` | 每顶点 | 清零 RHS 缓冲 |
| `add_inertial_rhs_kernel` | 每顶点 | atomicAdd M·y 到 RHS |
| `jacobi_divide_kernel` | 每顶点 | `x_new = rhs / diag` |
| `update_velocity_kernel` | 每顶点 | `v = (x_new - x_old)/h · (1-damping)` |
| `apply_constraints_kernel` | 每固定点 | 位置/速度强制置回 |

### 4.3 踩坑记录

- **重力预测**：`y += h²·g`，g 是加速度，**不是** `h²/m·g`（力）。
- **Jacobi 对角线**：三角形拉伸对 v0 贡献 `h²·wA·(s0²+s1²)`，对 v1/v2 贡献 `h²·wA·‖g_col‖²`；弯曲约束仅对翼顶点 v2/v3 贡献 `h²·w`，共享边顶点 v0/v1 不贡献（其投影固定在当前位置）。
- **Buffer 管理**：`d_prev_pos_` 专用于帧开始旧位置；Jacobi ping-pong 使用独立的 `d_new_pos_`，两者不混用。
- **CUDA/C++ 布局一致性**：`ClothMesh` 和 `SimConstraints` 的 GPU 指针和标量成员必须在 `#ifndef __CUDACC__` 之前，保证 `.cu`/`.cpp` 的 struct 内存偏移一致。
- **Chebyshev 尚未完成**：`chebyshev_accelerate()` 计算了 ω 值但未执行混合更新，需要额外 `d_cheby_prev_` buffer 和 blend kernel。

---

## 五、实现路线图

### ✅ Phase 1–2: 基础框架
- 网格数据结构、OBJ IO、拓扑构建
- CMake 多目标构建，独立测试

### ✅ Phase 2.5: 静态可视化
- GLFW + GLAD + ImGui OpenGL 查看器
- 静态网格渲染、相机控制

### ✅ Phase 3: GPU PD 求解器
- 三角形 Stiefel 拉伸约束（SVD 近似旋转）
- 二面角弯曲约束（`bend_project_kernel`）
- 完整 CUDA kernel 流水线
- 米字格网格（type 3），各向同性
- `sim_cloth` 命令行工具，支持 PLY 序列导出

### ✅ Phase Refactor: 代码重构（v0.5）
- 将 God struct `ClothMesh` 拆分为三层：`ClothMesh` / `MeshTopology` / `SimConstraints`
- 统一拓扑系统（删除旧双系统）
- 清理 `pd_solver.cu` 中的死代码（旧边拉伸 kernels、空桩方法）
- 修复 `constraints.cpp` extern 声明和 `reset_to_rest` 语义

**实测结果**（20×20 米字格，761 顶点，dt=0.01，50 iter/frame）：
- 固定行 y 坐标恒为 0 ✓
- 自由顶点从 y=0 下落，之后弹性回摆 ✓
- 全部 5 组单元测试通过（test_mesh, test_topology, test_constraints, test_pd_constraints）

---

### 🔲 Phase 4: Chebyshev 加速完整实现

**目标**：减少达到相同精度所需迭代次数（预期减少 40%）。

**当前状态**：`chebyshev_accelerate()` 已计算 ω 序列，混合更新未实现。

**需要**：
```cuda
// 额外缓冲 d_cheby_prev_ [N×3]，记录上一次 Jacobi 输出
// 新 kernel:
x = ω · (x_jacobi - x_prev) + x_prev
```

**验证**：
```bash
./sim_cloth 30 30 0.05 --iter 20 --steps 100 --verbose
# 对比 use_chebyshev=true/false
# 预期: Chebyshev 所需迭代数 < 60% 纯 Jacobi
```

---

### 🔲 Phase 5: CUDA–OpenGL Interop + 实时渲染

**目标**：把 PD 仿真接入 viewer，实时交互显示。

**关键修改**：
- `viewer.h/cpp`：增加 `update_positions_from_device(float* d_pos, int N)` 接口
  - 简单方案：`cudaMemcpy` device→host，再更新 VBO
  - 高性能方案：`cudaGraphicsGLRegisterBuffer` 零拷贝
- `main.cpp`：主循环每帧调用 `solver.step(mesh, sim_cons, pin_cons)`，再调用 viewer 更新
- ImGui 面板：运行时调节 dt、迭代次数、拉伸刚度、弯曲刚度、重力

**验证**：
```bash
./cuda_ms --cloth 30 30 --pin top
# 预期: >30 FPS，布料实时下垂，ImGui 参数可调
```

---

### 🔲 Phase 6: 性能优化（可选）

- Warp-level reduce 替代 `atomicAdd`
- 顶点重排提升 cache 局部性
- A-Jacobi 多步展开（减少 kernel launch 开销）

---

## 六、依赖库

| 库 | 用途 | 状态 |
|----|------|------|
| CUDA Toolkit | GPU 计算 | ✅ |
| Eigen 3 | CPU 预处理、线性代数 | ✅ |
| GLFW 3.3.8 | 窗口/输入 | ✅ (FetchContent) |
| GLAD | OpenGL 加载 | ✅ (vendored) |
| ImGui v1.90 | UI 面板 | ✅ (FetchContent) |
| GLM 1.0.1 | GPU 侧矩阵 | ✅ (FetchContent) |

---

*文档 v0.5：Phase 3 + Refactor 完成；下一步 Phase 4 Chebyshev 加速*
