# CUDA Projective Dynamics 布料仿真框架设计文档

> 版本：v0.4
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

系统矩阵 `A = (M + h²L)` 是**常数矩阵**，对角线可预计算，每帧只需 Jacobi 回代。

### 1.2 时间积分流程

```
每帧:
1. 预测: y = x_n + h·v_n + h²·g        ← g 是加速度，不除质量
2. for iter = 0..max_iter:
   a. Local Step  (并行): p_c = project_constraint(x)
   b. Global Step (Jacobi): rhs = M·y + h²·Σ w_c·p_c
                             x_new = rhs / diag
   c. Chebyshev 加速 (Phase 4)
   d. 应用固定约束
3. 速度: v_{n+1} = (x_{n+1} - x_n) / h
```

### 1.3 Jacobi 迭代公式

对每个顶点 `i`：
```
x_i^{new} = (m_i·y_i + h²·Σ_{c∈C_i} w_c·p_c^i) / (m_i + h²·Σ_{c∈C_i} w_c)
```

分母 `diag_i = m_i + h²·Σ w_c` 是常数，每条约束对每个端点贡献 `w`（不含额外系数），预计算存储。

### 1.4 Chebyshev 加速

```
ω_1 = 1,  ω_2 = 2/(2 - ρ²)   (ρ ≈ 0.9~0.99，Jacobi 谱半径估计)
k ≥ 2:  ω_{k+1} = 4/(4 - ρ²·ω_k)

x^{k+1} = ω_{k+1}·(Jacobi(x^k) - x^{k-1}) + x^{k-1}
```

### 1.5 布料约束类型

| 约束 | 类型 | 投影算子 |
|------|------|----------|
| 拉伸 | 边长度 | 两端点投影至目标长度 |
| 弯曲 | 二面角 | 4 顶点投影至目标二面角 |

**拉伸投影**（已实现）：
```
center = (x_i + x_j) / 2
dir    = (x_j - x_i) / |x_j - x_i|
p_i = center - (L_rest/2)·dir,   p_j = center + (L_rest/2)·dir
```

**弯曲投影**（Phase 4）：
基于共享边 (v0,v1) 两侧三角形的二面角，沿梯度方向微调 4 个顶点使 θ → θ_rest。

---

## 二、工程现状与文件结构

### 当前实际文件树

```
cuda-ms/
├── CMakeLists.txt
├── DESIGN.md
├── include/
│   ├── mesh.h              ✅ GPU/CPU 布局一致；含 PD 约束 GPU 指针
│   ├── mesh_generator.h    ✅ 4 种三角化类型（0-3，默认米字格）
│   ├── constraints.h       ✅ 固定点约束 + GPU upload/apply
│   ├── pd_solver.h         ✅ PDSolverConfig + PDSolver 接口
│   └── utils/cuda_helper.h ✅
├── src/
│   ├── CMakeLists.txt      ✅ pd_solver 静态库 + sim_cloth 可执行
│   ├── mesh.cpp            ✅ 约束构建、Jacobi 对角线预计算
│   ├── mesh_generator.cpp  ✅ 含米字格 type 3
│   ├── constraints.cpp     ✅ GPU upload、apply_gpu
│   ├── pd_solver.cu        ✅ 全 CUDA kernel 实现
│   ├── viewer/             ✅ Phase 2.5 OpenGL 查看器
│   └── tools/
│       ├── gen_cloth.cpp   ✅
│       ├── view_cloth.cpp  ✅
│       └── sim_cloth.cpp   ✅ 命令行 PD 仿真 + PLY 导出
└── shaders/                ✅
```

### 可执行文件

| 程序 | 说明 |
|------|------|
| `sim_cloth <rows> <cols> <size> [opts]` | GPU PD 仿真，支持 PLY 序列导出 |
| `view_cloth <file.obj>` | 静态网格 OpenGL 查看器 |
| `gen_cloth` | 生成 OBJ 网格文件 |
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

**米字格（type 3）**：每个 quad 中心增加一个顶点，分裂为 4 个等腰三角形，各向同性，无对角线方向偏差：
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
  d_predict_      [N×3]      惯性预测位置 y
  d_rhs_          [N×3]      Global Step RHS 累加器
  d_prev_pos_     [N×3]      帧开始时旧位置，用于 v = (x_new - x_old)/h
  d_new_pos_      [N×3]      Jacobi ping-pong 输出缓冲
  d_stretch_proj_ [E_s×2×3]  拉伸约束 Local Step 投影

ClothMesh GPU 数据（持久）：
  d_pos, d_vel, d_mass        当前运动状态
  d_stretch_edges/rest/k      拉伸约束参数
  d_jacobi_diag [N]           预计算 m_i + h²·Σw_c（常数，构建时算）
  d_bend_quads/rest/k         弯曲约束（Phase 4）
```

### 4.2 已实现 CUDA Kernels

| Kernel | 线程粒度 | 功能 |
|--------|---------|------|
| `predict_kernel` | 每顶点 | `y = x + h·v + h²·g` |
| `stretch_project_kernel` | 每边 | 计算两端投影点 p0, p1 |
| `clear_rhs_kernel` | 每顶点 | 清零 RHS 缓冲 |
| `accumulate_stretch_rhs_kernel` | 每边 | `atomicAdd h²·w·p` 到 RHS |
| `add_inertial_rhs_kernel` | 每顶点 | `atomicAdd m·y` 到 RHS |
| `jacobi_divide_kernel` | 每顶点 | `x_new = rhs / diag` |
| `update_velocity_kernel` | 每顶点 | `v = (x_new - x_old)/h`，更新 pos |
| `apply_constraints_kernel` | 每固定点 | 位置/速度强制置回 |

### 4.3 实现注意事项（踩坑记录）

- **重力预测**：`y += h²·g`，g 是加速度，**不是** `h²/m·g`（力）。误用后导致每步 4000× 位移过大。
- **Jacobi 对角线**：每条拉伸约束对两端顶点各贡献 `h²·w` 一次，无额外系数。之前误用 `2·constraint_wt·w` 导致分母过大、弹簧太软。
- **Buffer 管理**：`d_prev_pos_` 专用于帧开始旧位置；Jacobi ping-pong 使用独立的 `d_new_pos_`，两者不混用。
- **mesh.h 布局**：GPU 指针和标量成员必须在 `#ifndef __CUDACC__` 之前，保证 `.cu`/`.cpp` 的 struct 内存偏移一致。

---

## 五、实现路线图

### ✅ Phase 1–2: 基础框架
- 网格数据结构、OBJ IO、拓扑构建
- CMake 多目标构建，独立测试

### ✅ Phase 2.5: 静态可视化
- GLFW + GLAD + ImGui OpenGL 查看器
- 静态网格渲染、相机控制

### ✅ Phase 3: GPU PD 求解器（拉伸）
- `build_stretch_constraints()` — 从三角形提取唯一边约束
- `precompute_jacobi_diag()` — 预计算 Jacobi 分母
- Constraints GPU upload / apply
- 完整 CUDA kernel 流水线
- 米字格网格（type 3），各向同性
- `sim_cloth` 命令行工具，支持 PLY 序列导出

**实测结果**（20×20 米字格，761顶点，dt=0.01，50 iter/frame）：
- 固定行 y 坐标恒为 0 ✓
- 自由顶点从 y=0 下落约 −0.33m（第 50 步），之后弹性回摆 ✓
- 能量量级合理（峰值约 −1 J，对应 0.1 kg 布料下落 ~1 m）✓

---

### 🔲 Phase 4: 弯曲约束 + Chebyshev 加速

**目标**：布料具有抗弯刚度（折叠有阻力）；Chebyshev 加速减少所需迭代次数。

#### 4.1 弯曲约束

基于 `inner_edges`（已构建）的二面角约束，每条内边关联四顶点 (v0,v1,v2,v3)：

**CPU 端**（已有骨架，需完善 `build_bend_constraints()`）：
```cpp
// 计算并存储 rest_angle = dihedral(v0,v1,v2,v3)
// 数据已上传：d_bend_quads, d_bend_rest, d_bend_k
```

**GPU 端需新增**：
```cuda
// 每线程处理一条弯曲约束
__global__ void bend_project_kernel(
    const float* pos,
    const int* quads,      // [E_bend×4]
    const float* rest_angles,
    const float* stiffness,
    float3* projections,   // [E_bend×4] 四顶点投影
    int num_bends);

// 弯曲 RHS 累加（同 stretch，用 atomicAdd）
__global__ void accumulate_bend_rhs_kernel(...);
```

**`precompute_jacobi_diag` 扩展**：每条弯曲约束对 4 个顶点各贡献 `h²·w_bend`。

#### 4.2 Chebyshev 完整实现

当前 `chebyshev_accelerate()` 已计算 ω 值，但尚未执行混合更新。需要：
- 额外缓冲 `d_cheby_prev_` [N×3]，记录上一次 Jacobi 输出
- 新 kernel：`x = ω·(x_jacobi - x_prev) + x_prev`

#### 验证检查点

```bash
# 4.1: 弯曲刚度对比（有 vs 无 bend 约束）
./sim_cloth 20 20 0.05 --stiffness 10 --bend 0.01 --steps 300 --export ./out/
# 预期: 布料边缘不再完全折叠，保持曲率

# 4.2: Chebyshev 加速效果（相同精度所需迭代数）
./sim_cloth 30 30 0.05 --iter 20 --steps 100 --verbose
# 对比 use_chebyshev=true/false
# 预期: Chebyshev 所需迭代数 < 60% 纯 Jacobi
```

---

### 🔲 Phase 5: CUDA–OpenGL Interop + 实时渲染

**目标**：把 Phase 3/4 的 PD 仿真接入 Phase 2.5 的 viewer，实时交互显示。

**关键修改**：
- `viewer.h/cpp`：增加 `update_positions_from_device(float* d_pos, int N)` 接口
  - 简单方案：`cudaMemcpy` device→host，再更新 VBO
  - 高性能方案：`cudaGraphicsGLRegisterBuffer` 零拷贝
- `main.cpp`：主循环中每帧调用 `solver.step(mesh, cons)`，再调用 viewer 更新
- ImGui 面板：运行时调节 dt、迭代次数、拉伸刚度、弯曲刚度、重力

#### 验证检查点

```bash
./cuda_ms --cloth 30 30 --pin top
# 预期: >30 FPS，布料实时下垂，ImGui 参数可调
```

---

### 🔲 Phase 6: 性能优化（可选）

- Warp-level reduce 替代 `atomicAdd`（按 warp 聚合后再 atomic）
- 顶点重排提升 cache 局部性
- A-Jacobi 多步展开（减少 kernel launch 开销）
- 层次化 Jacobi / 多分辨率收敛

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

**无需** cuSPARSE：PD Jacobi 不涉及稀疏矩阵求解。

---

*文档 v0.4：Phase 3 完成；Phase 4（弯曲 + Chebyshev）为当前下一步*
