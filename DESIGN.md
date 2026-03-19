# CUDA FEM 布料仿真框架设计文档

> 作者：仿真工程师视角
> 版本：v0.1
> 目标：基于 CUDA 的 FEM 三角面片布料仿真，暂不考虑碰撞处理

---

## 一、理论基础概述

### 1.1 问题描述

布料建模为嵌入三维空间的二维薄壳。我们将布料离散为三角形网格：

- **参考构型（Reference / Rest Configuration）**：布料静止时的形状，定义在材料空间（2D 或展平的 3D 坐标）
- **当前构型（Deformed Configuration）**：仿真过程中的世界坐标（3D）

对每个三角形，FEM 的核心任务是：
1. 计算变形梯度 **F**（3×2 矩阵，从材料空间到世界空间）
2. 由本构模型计算应变能密度 Ψ(F)
3. 计算弹性力（能量对节点位置的梯度）
4. （隐式积分时）计算切线刚度矩阵（能量的 Hessian）

### 1.2 变形梯度

设三角形三顶点在参考构型中的 2D 坐标为 X₁, X₂, X₃ ∈ ℝ²，在当前构型中的 3D 坐标为 x₁, x₂, x₃ ∈ ℝ³。

定义参考边矩阵（2×2）：
```
D_m = [X₂ - X₁ | X₃ - X₁]    (2×2)
```

当前构型边矩阵（3×2）：
```
D_s = [x₂ - x₁ | x₃ - x₁]    (3×2)
```

变形梯度（3×2）：
```
F = D_s · D_m⁻¹               (3×2)
```

注意：D_m 是方阵可直接求逆，且在预处理阶段计算一次即可存储。

### 1.3 本构模型（膜力）

#### St. Venant-Kirchhoff（StVK）

Green 应变张量（2×2）：
```
E = (FᵀF - I) / 2
```

应变能密度：
```
Ψ = (λ/2) tr(E)² + μ tr(EᵀE)
```

其中 λ, μ 为 Lamé 参数，可由杨氏模量 E_Y 与泊松比 ν 换算：
```
λ = E_Y·ν / ((1+ν)(1-2ν))
μ = E_Y / (2(1+ν))
```

StVK 对大压缩不稳定，但对布料的小至中等形变已经足够。

#### Neo-Hookean（备选，更稳健）

```
Ψ = (μ/2)(tr(FᵀF) - 2) - μ·ln(J) + (λ/2)·ln(J)²
```

其中 J = sqrt(det(FᵀF)) 为面积比。对布料偏好 StVK 或各向异性模型，Neo-Hookean 更适合体积仿真，但也可用。

### 1.4 弯曲模型

纯膜力（in-plane）不产生抵抗弯折的力。布料的弯曲通过**离散铰链弹簧**（Discrete Hinge / Dihedral Angle）建模：

对每条内边（连接两个三角形的共享边），计算相邻三角形的二面角θ，弯曲能量：
```
E_bend = k_bend · (θ - θ_rest)² · l_e / (A₁ + A₂)
```

其中 l_e 为边长，A₁, A₂ 为两三角形面积。

> 弯曲力的推导较膜力复杂，将在实现阶段详细展开。

### 1.5 时间积分

#### 显式 Newmark / Verlet（入门首选）

```
v_{n+1/2} = v_n + (Δt/2) · a_n
x_{n+1}   = x_n + Δt · v_{n+1/2}
a_{n+1}   = M⁻¹ · f(x_{n+1})
v_{n+1}   = v_{n+1/2} + (Δt/2) · a_{n+1}
```

优点：实现简单，CUDA 完全并行。
缺点：条件稳定，时间步长受材料刚度限制（Δt < 2/ω_max，其中 ω_max 为最大频率）。

#### 隐式 Backward Euler（生产首选）

求解：
```
M·(x_{n+1} - 2x_n + x_{n-1}) / Δt² = f(x_{n+1})
```

等价于求解非线性方程组，通常用 Newton 法线性化：
```
(M/Δt² - K(x_k)) · Δx = f(x_k) - M·(x_k - 2x_n + x_{n-1}) / Δt²
```

每个 Newton 步需在 GPU 上求解稀疏线性系统，使用**共轭梯度法（PCG）**。

---

## 二、模块划分

```
cuda-cloth-sim/
├── CMakeLists.txt
├── include/
│   ├── mesh.h              # 网格数据结构
│   ├── material.h          # 本构模型参数与接口
│   ├── fem_engine.h        # FEM 力/刚度计算接口
│   ├── bending.h           # 弯曲力计算接口
│   ├── sparse_matrix.h     # GPU 稀疏矩阵（CSR）
│   ├── linear_solver.h     # GPU 共轭梯度求解器
│   ├── integrator.h        # 时间积分器接口
│   └── simulation.h        # 顶层仿真管理器
├── src/
│   ├── mesh.cpp            # 网格 IO、拓扑构建（Eigen 用于预处理）
│   ├── fem_engine.cu       # CUDA 核函数：力/刚度
│   ├── bending.cu          # CUDA 核函数：弯曲力
│   ├── sparse_matrix.cu    # CSR 构建、矩阵向量乘法
│   ├── linear_solver.cu    # PCG 求解器（cuSPARSE / 手写）
│   ├── integrator.cu       # 显式/隐式积分器
│   └── main.cpp            # 驱动程序
└── utils/
    ├── obj_io.cpp          # OBJ 网格读写
    ├── ply_io.cpp          # PLY 输出（帧序列）
    └── cuda_helper.h       # CUDA 错误检查宏
```

---

## 三、各模块详细设计

### 3.1 Mesh 模块

#### 数据结构

```cpp
// include/mesh.h

struct ClothMesh {
    // --- CPU 端原始数据（预处理用）---
    std::vector<Eigen::Vector3f> rest_pos;    // 参考构型顶点（3D，展平后可投影到2D）
    std::vector<Eigen::Vector3i> triangles;   // 三角形顶点索引
    std::vector<Eigen::Vector2i> edges;       // 内边（弯曲用），存 (tri_a, tri_b)

    // --- 预计算量（每个三角形，CPU→GPU）---
    std::vector<Eigen::Matrix2f> Dm_inv;      // D_m 的逆 (2×2)，每个三角形一个
    std::vector<float>           rest_area;   // 参考构型面积
    std::vector<float>           mass;        // 节点质量（由面积分布）

    // --- GPU 端镜像（device 指针）---
    float*  d_pos;          // 当前位置 [N×3]
    float*  d_vel;          // 当前速度 [N×3]
    int*    d_tris;         // 三角形索引 [T×3]
    float*  d_Dm_inv;       // [T×4]（2×2 矩阵按列主序展平）
    float*  d_rest_area;    // [T]
    float*  d_mass;         // [N]（集中质量）

    int num_verts;
    int num_tris;
    int num_inner_edges;
};
```

#### 关键操作

- `load_obj(path)` — 读取 OBJ 文件，构建拓扑
- `build_topology()` — 构建内边表，标记边界点
- `precompute_rest_state()` — 计算每个三角形的 Dm_inv 和 rest_area（Eigen 处理）
- `compute_mass_matrix()` — 均匀密度下将面积质量分配到顶点（Lumped Mass）
- `upload_to_gpu()` — 将预计算数据传送至 GPU

#### 关于 Eigen 的使用策略

Eigen 仅用于 **CPU 端预处理**（矩阵求逆、拓扑构建），不引入 GPU 端依赖。
GPU 端所有数据为简单 float 数组，手工实现 2×2、3×2 矩阵运算（inline device 函数）。

---

### 3.2 Material 模块

```cpp
// include/material.h

enum class MaterialModel { StVK, NeoHookean };

struct MaterialParams {
    MaterialModel model = MaterialModel::StVK;
    float young_modulus = 1e5f;   // 杨氏模量 (Pa)
    float poisson_ratio = 0.3f;   // 泊松比
    float density       = 0.1f;   // 面密度 (kg/m²)
    float damping_alpha = 0.01f;  // Rayleigh 阻尼系数（质量项）
    float damping_beta  = 0.001f; // Rayleigh 阻尼系数（刚度项）

    // 派生 Lamé 参数
    float lambda() const;
    float mu() const;
};
```

弯曲参数单独管理：
```cpp
struct BendingParams {
    float stiffness = 1e-3f;   // 弯曲刚度
};
```

---

### 3.3 FEM Engine 模块

这是计算核心，所有 CUDA kernel 在此定义。

#### 3.3.1 数据流

```
per-triangle kernel
  输入：d_pos, d_tris, d_Dm_inv, d_rest_area
  输出：d_force（原子加法到顶点）
        d_K_entries（稀疏矩阵非零元素填充）
```

#### 3.3.2 力计算 Kernel（显式积分用）

```cuda
// 每个线程处理一个三角形
__global__ void compute_elastic_forces(
    const float* __restrict__ pos,      // [N×3]
    const int*   __restrict__ tris,     // [T×3]
    const float* __restrict__ Dm_inv,   // [T×4]
    const float* __restrict__ rest_area,// [T]
    float*       forces,                // [N×3]，原子累加
    int num_tris,
    MaterialParams params
);
```

每个三角形的处理流程：

```
1. 读取三顶点位置 x0, x1, x2
2. 构建 Ds = [x1-x0 | x2-x0]   (3×2)
3. F = Ds * Dm_inv               (3×2)
4. 计算 FᵀF                     (2×2)
5. 计算 Green 应变 E = (FᵀF-I)/2
6. 计算第一 PK 应力 P(F)         (3×2)
   StVK: P = F * (2μE + λ·tr(E)·I)
7. 计算节点力 H = -A₀ * P * Dm_inv^T  (3×2)
   f1 = H[:,0], f2 = H[:,1], f0 = -(f1+f2)
8. atomicAdd 到 forces[i0], forces[i1], forces[i2]
```

#### 3.3.3 刚度矩阵 Kernel（隐式积分用）

每个三角形贡献一个 9×9 局部刚度矩阵（3顶点×3自由度），散射到全局 CSR 稀疏矩阵。

```cuda
__global__ void compute_stiffness_entries(
    const float* pos,
    const int*   tris,
    const float* Dm_inv,
    const float* rest_area,
    float*       K_vals,    // 稀疏矩阵非零值（预分配结构）
    int num_tris,
    MaterialParams params
);
```

刚度计算方法：
- 解析推导（高效，需要推导 ∂P/∂F 的表达式）
- 或数值差分（调试用，生产环境不推荐）

---

### 3.4 Bending 模块

```cuda
// 每个线程处理一条内边 (共享边)
__global__ void compute_bending_forces(
    const float* pos,
    const int*   inner_edges,   // [E×4]: 每条内边的4个相关顶点索引
    const float* rest_angles,   // [E]: 参考构型二面角
    const float* rest_lengths,  // [E]: 边长
    float*       forces,
    int num_inner_edges,
    BendingParams params
);
```

每条内边涉及 4 个顶点（两个三角形共 4 个，去除共享边 2 个后还剩 4 个），需原子累加到这 4 个顶点的力。

---

### 3.5 Sparse Matrix 模块

使用 **CSR（Compressed Sparse Row）** 格式存储全局刚度矩阵 K（3N × 3N）。

```cpp
// include/sparse_matrix.h

struct CsrMatrix {
    int   num_rows;
    int   nnz;          // 非零元素数量
    int*  d_row_ptr;    // [num_rows+1]
    int*  d_col_ind;    // [nnz]
    float* d_val;       // [nnz]
};
```

#### 稀疏矩阵构建策略

1. **预分析阶段（CPU）**：根据网格拓扑确定刚度矩阵的稀疏模式（哪些位置非零），分配 CSR 结构
2. **每时间步（GPU）**：仅更新非零值，结构不变

稀疏模式由网格拓扑决定：若顶点 i 和 j 属于同一三角形，则 K[3i:3i+3, 3j:3j+3] 非零（9个条目）。

---

### 3.6 Linear Solver 模块（PCG）

隐式积分的核心需要求解：

```
A·x = b
A = M/Δt² + K
```

使用**预条件共轭梯度（PCG）**，预条件器选用块对角（Jacobi 块预条件，每块 3×3）。

```cpp
// include/linear_solver.h

class PCGSolver {
public:
    void solve(
        const CsrMatrix& A,
        const float*     b,      // [3N]
        float*           x,      // [3N]，输出
        int              max_iter = 200,
        float            tol     = 1e-6f
    );
private:
    // 工作向量（预分配，避免每步 malloc）
    float *d_r, *d_p, *d_q, *d_z;
    // cuSPARSE handle
    cusparseHandle_t sparse_handle;
};
```

矩阵向量乘法（SpMV）直接调用 **cuSPARSE**（`cusparseSpMV`）。

---

### 3.7 Integrator 模块

提供统一接口，支持切换积分方法：

```cpp
// include/integrator.h

class Integrator {
public:
    virtual void step(ClothMesh& mesh, float dt) = 0;
};

class ExplicitVerlet : public Integrator {
public:
    void step(ClothMesh& mesh, float dt) override;
    // GPU 上完全并行，无需线性求解
};

class ImplicitEuler : public Integrator {
public:
    void step(ClothMesh& mesh, float dt) override;
    // 调用：力计算 → 刚度计算 → 组装 A → PCG 求解 → 更新状态
private:
    PCGSolver solver;
    CsrMatrix system_matrix;
};
```

#### 显式 Verlet 步骤（CUDA 流程）

```
1. kernel: compute_elastic_forces  → d_force
2. kernel: compute_bending_forces  → d_force (累加)
3. kernel: add_gravity             → d_force (累加)
4. kernel: apply_damping           → d_force (Rayleigh)
5. kernel: integrate_verlet        → 更新 d_pos, d_vel
6. kernel: apply_constraints       → 固定顶点归零
```

#### 隐式 Euler 步骤（CUDA 流程）

```
Newton 迭代（通常 1～3 步近似即可）：
1. 计算当前力 f(x_k)
2. 计算刚度矩阵 K(x_k)
3. 组装系统矩阵 A = M/Δt² + K（+ 阻尼项）
4. 组装右端项 b = M·(x_n - x_k)/Δt² + f(x_k)
5. PCG 求解 A·Δx = b
6. 更新 x_k ← x_k + Δx
7. 检查收敛（|Δx| < tol），否则返回 1
更新速度 v = (x_new - x_old) / Δt
```

---

### 3.8 Constraint 模块

```cuda
// 将被 pin 住的顶点的 force 和 velocity 清零
__global__ void apply_pin_constraints(
    float* force,
    float* vel,
    const int* pinned_indices,
    int num_pinned
);
```

约束以索引列表形式存储，每步最后执行，实现简单。

---

### 3.9 Simulation 模块（顶层）

```cpp
// include/simulation.h

class ClothSimulation {
public:
    void init(const std::string& obj_path, const SimConfig& config);
    void step();
    void save_frame(const std::string& path);

private:
    ClothMesh       mesh;
    MaterialParams  material;
    BendingParams   bending;
    Integrator*     integrator;   // 指向 Explicit 或 Implicit
    int             frame_id = 0;
};

struct SimConfig {
    float dt           = 1e-3f;
    int   substeps     = 10;
    float gravity      = -9.8f;
    std::vector<int> pinned_verts;
    std::string integrator_type = "explicit";  // "explicit" or "implicit"
};
```

---

## 四、GPU 并行化设计要点

### 4.1 线程组织

| Kernel | 线程映射 | Grid 大小 |
|--------|----------|----------|
| 弹性力计算 | 1 thread / triangle | (T + 128 - 1) / 128 |
| 弯曲力计算 | 1 thread / inner edge | (E + 128 - 1) / 128 |
| 向量操作 | 1 thread / DOF (3N) | (3N + 256 - 1) / 256 |
| SpMV | cuSPARSE 自动 | — |

### 4.2 力的原子累加

每个三角形 kernel 将力原子累加（`atomicAdd`）到对应顶点。对于现代 GPU （sm_60+），float 的 atomicAdd 已有硬件支持，性能可接受。

若原子操作成为瓶颈，可改用**图着色（Graph Coloring）**方案：将三角形染色使同色三角形不共享顶点，每种颜色串行执行但同色内完全并行。

### 4.3 内存布局

顶点数据使用 **SOA（Structure of Arrays）** 而非 AOS：
```
x[0], x[1], ..., x[N-1]   // 所有 x 分量
y[0], y[1], ..., y[N-1]   // 所有 y 分量
z[0], z[1], ..., z[N-1]   // 所有 z 分量
```
或交错的 **AOS** 也可（`x0, y0, z0, x1, y1, z1, ...`），cuSPARSE 的 SpMV 对两种布局都支持。

---

## 五、依赖库

| 库 | 用途 | 是否必须 |
|----|------|----------|
| CUDA Toolkit | GPU 计算核心 | 必须 |
| cuSPARSE | GPU 稀疏矩阵运算 | 强烈推荐 |
| cuBLAS | GPU 向量点积等 | 可选（PCG 内部用） |
| Eigen 3 | CPU 端预处理（矩阵求逆、IO） | 推荐 |
| tinyobjloader | OBJ 文件读取 | 推荐 |
| CMake 3.18+ | 构建系统 | 必须 |

Eigen 仅参与 CPU 端预处理，**不进入任何 .cu 文件**，避免 nvcc 与 Eigen 模板的兼容问题。

---

## 六、实现路线图

### 阶段一：基础框架（显式积分）
- [ ] Mesh 加载与预处理（CPU + Eigen）
- [ ] GPU 内存管理封装
- [ ] 显式 Verlet 积分器 CUDA kernel
- [ ] StVK 膜力 kernel（无弯曲）
- [ ] 固定约束
- [ ] OBJ/PLY 帧输出
- [ ] **验证**：单三角形拉伸，与解析解对比

### 阶段二：弯曲与验证
- [ ] 内边拓扑构建
- [ ] 离散铰链弯曲力 kernel
- [ ] **验证**：悬挂布料静态平衡，与 ANSYS/FEniCS 参考解对比

### 阶段三：隐式积分
- [ ] CSR 稀疏矩阵构建
- [ ] 刚度矩阵 kernel（解析推导 ∂P/∂F）
- [ ] PCG 求解器（cuSPARSE SpMV + 手写 PCG）
- [ ] 隐式 Euler 积分器
- [ ] **验证**：大时间步稳定性测试

### 阶段四：性能优化（按需）
- [ ] 图着色替代 atomicAdd
- [ ] CUDA Stream 重叠计算与传输
- [ ] 共享内存优化 kernel
- [ ] Profiling（Nsight Compute）

---

## 七、验证方案

### 7.1 片元级测试

单三角形从参考态拉伸，检查：
- 力的方向与大小（对称性验证）
- 零能量模式（刚体运动不产生力）
- 力对位置的数值梯度 ≈ 解析刚度矩阵

### 7.2 系统级测试

| 测试场景 | 预期行为 |
|----------|----------|
| 自由落体（无弹力） | 纯重力加速，与解析解精确一致 |
| 两端固定悬链 | 收敛到悬链线静态解 |
| 四角固定方形布 | 中心下沉量与材料参数一致 |
| 动能守恒（无阻尼显式） | 总能量在机器精度内守恒 |

---

## 八、性能预期

以典型布料网格（10k 顶点，20k 三角形）为例，RTX 3080 上预期：

| 操作 | 耗时（估计） |
|------|-------------|
| 膜力计算（20k tri） | < 0.5 ms |
| 弯曲力计算（30k edges） | < 1 ms |
| PCG 求解（100 iter）| 5～20 ms |
| 整体单步（显式）| < 2 ms |
| 整体单步（隐式，Δt=10ms）| 10～30 ms |

---

---

## 九、可视化方案

### 9.1 技术选型对比

| 候选方案 | CUDA-GL 零拷贝 | 实时 60fps | GUI 支持 | 集成难度 | 推荐度 |
|---------|---------------|-----------|---------|---------|-------|
| **OpenGL + Dear ImGui** | 原生支持 | 是 | 优秀（ImGui） | 中等 | **首选** |
| Polyscope | Beta/部分支持 | 中小网格可 | 优秀（内置） | 低 | 第二选择 |
| libigl | 需 hack | 是 | 良好 | 低 | 不推荐用于实时 |
| VTK/ParaView | 不支持 | 可 | 无内置 | 高 | 过度设计 |
| Open3D | **被 Filament 阻断** | 否 | 一般 | 中 | 不推荐 |

### 9.2 推荐方案：OpenGL + Dear ImGui

核心原因：
1. **零拷贝是刚需**：CUDA-OpenGL interop 允许 `cudaGraphicsGLRegisterBuffer()` 将 OpenGL VBO 映射为 CUDA device pointer，kernel 直接写入，无 CPU 中转
2. **完全控制渲染循环**：map → kernel launch → unmap → draw，精确控制同步
3. **Dear ImGui 是标准**：参数滑块（刚度/重力/时间步）、播放/暂停按钮、能量实时绘图（ImPlot），～100 行代码搞定

#### CUDA-OpenGL 互操作流水线

```
CUDA kernel (布料求解器) ───> CUDA device buffer (顶点位置)
              │
              │ cudaGraphicsGLRegisterBuffer
              ▼
      OpenGL VBO (GL_DYNAMIC_DRAW)
              │
              ▼
     Phong shader pipeline (光照+法线)
              │
              ▼
     Dear ImGui overlay ────> GLFW window
```

#### 关键 API 调用

```cpp
// 1. 创建 OpenGL VBO
glGenBuffers(1, &vbo);
glBindBuffer(GL_ARRAY_BUFFER, vbo);
glBufferData(GL_ARRAY_BUFFER, N*3*sizeof(float), nullptr, GL_DYNAMIC_DRAW);

// 2. 注册为 CUDA 图形资源
cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

// 3. 每帧更新（GPU 端）
cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, cuda_vbo_resource);
launch_cloth_kernel<<<grid, block>>>(dev_ptr, ...);  // CUDA kernel 直接写
move_vertices_from_simulation_data<<<>>>(d_sim_pos, dev_ptr, N);
cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

// 4. OpenGL 渲染
glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, 0);

// 5. Dear ImGui 绘制控件面板
ImGui::Begin("Simulation Control");
ImGui::SliderFloat("Time Step", &dt, 0.0001f, 0.01f);
ImGui::SliderFloat("Young's Modulus", &E, 1e4f, 1e6f);
if (ImGui::Button("Play/Pause")) running = !running;
ImGui::End();
```

### 9.3 可视化模块结构

```
include/
├── viewer.h              # 查看器接口
└── ui/
    ├── gui_manager.h     # Dear ImGui 封装
    └── energy_plotter.h  # 能量曲线实时绘制
src/
├── viewer.cu             # CUDA-GL 互操作核心
├── ui/
│   ├── gui_manager.cpp   # 控件面板实现
│   └── energy_plotter.cpp
└── shaders/
    ├── phong.vs          # 顶点着色器
    ├── phong.fs          # 片段着色器
    └── normal.geom       # 可选：几何着色器计算法线
```

### 9.4 Viewer 类接口

```cpp
// include/viewer.h

class ClothViewer {
public:
    void init(int window_w, int window_h, int num_verts, int num_tris);
    void update_mesh(const float* d_positions);  // GPU device pointer
    void draw(const ClothMesh& mesh, const SimState& state);
    bool should_close() const;

    // GUI 状态绑定
    float*        gui_dt;
    float*        gui_young_modulus;
    float*        gui_bending_stiffness;
    bool*         gui_running;

private:
    GLFWwindow*   window_;
    GLuint        vbo_, ebo_;
    cudaGraphicsResource_t cuda_vbo_resource_;
    ShaderProgram shader_;
    EnergyPlotter energy_plot_;
};
```

### 9.5 备选方案：Polyscope

若快速原型优先，且网格规模较小（< 50K 顶点）：
- 零代码 GUI：相机控制、颜色映射、标量场可视化
- **当前局限**：`updateVertexPositions()` 走 CPU 路径，大网格会掉帧
- **未来可期**：GPU buffer update API 正在 Beta 中，稳定后可替换

### 9.6 帧输出离线可视化

提供 PLY 帧序列导出，支持导入 ParaView/Blender 进行后处理渲染。

---

*文档将随实现进展持续更新。*
