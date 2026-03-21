# CUDA Projective Dynamics 布料仿真框架设计文档

> 作者：仿真工程师视角
> 版本：v0.3
> 目标：基于 CUDA 的 Projective Dynamics 布料仿真，暂不处理碰撞

---

## 一、理论基础概述

### 1.1 Projective Dynamics 核心思想

PD 将隐式积分中的能量最小化问题转化为**约束投影问题**：

```
x_{n+1} = argmin  (1/2h²)||x - y||²_M + Σ w_i ||A_i x - p_i||²

其中 y = x_n + h·v_n + h²·M⁻¹·f_ext  (惯性预测位置)
```

- **Local Step**: 对每个约束计算投影 `p_i = projection(A_i x)` —— 完全并行
- **Global Step**: 求解固定线性系统 `(M + h²L)x = M y + h² Σ w_i A_i^T p_i`

系统矩阵 `A = (M + h²L)` 是**常数矩阵**，可预分解或预计算，每帧只需回代。

### 1.2 布料约束类型

| 约束 | 类型 | 投影算子 |
|------|------|----------|
| 拉伸 | 边长度 | 将当前边向量投影到目标长度 |
| 弯曲 | 二面角 | 将4个顶点投影到目标二面角 |

#### 拉伸约束
对边 `(i,j)`，目标长度 `L_rest`：
```
p = (x_i + x_j)/2 ± (L_rest/2) · (x_j - x_i)/|x_j - x_i|
```

#### 弯曲约束
对共享边 `(i,j)` 的两个三角形，参考二面角 `θ_rest`：
```
通过旋转使当前二面角 = θ_rest，然后投影4个顶点
```

### 1.3 线性求解策略

PD 的 Global Step 需求解 `A x = b`。策略演进：

| 方法 | 复杂度 | 适合场景 |
|------|--------|----------|
| Jacobi | O(k·N) 每帧，k次迭代 | GPU 友好，易实现 |
| Jacobi + Chebyshev | O(k·N)，收敛更快 | 推荐默认方案 |
| 预分解直接求解 | O(N) 每帧，预计算O(N³) | 小网格，固定拓扑 |

**本框架采用 Jacobi + Chebyshev 加速**，GPU 友好且无需稀疏矩阵库。

### 1.4 时间积分流程

```
每帧:
1. 预测位置: y = x_n + h·v_n + h²·f_ext/m
2. For iteration = 0 to max_iter:
   a. Local Step (并行):  p_i = project_constraint(x)
   b. Global Step (Jacobi): x = (M y + h² Σ A_i^T p_i) / (M + h² Σ A_i^T A_i)
   c. Chebyshev 加速更新
3. 更新速度: v_{n+1} = (x_{n+1} - x_n) / h
4. 应用约束: 固定点位置重置
```

---

## 二、模块划分

```
cuda-ms/
├── CMakeLists.txt
├── include/
│   ├── mesh.h              # 网格数据结构
│   ├── mesh_generator.h    # 程序化网格生成
│   ├── constraints.h       # 固定约束管理
│   ├── material.h          # 材料参数 (拉伸/弯曲刚度)
│   ├── pd_solver.h         # PD 求解器核心
│   ├── simulation.h        # 顶层仿真管理
│   └── viewer.h            # OpenGL 可视化 (Phase 2.5 已实现)
├── src/
│   ├── CMakeLists.txt
│   ├── mesh.cpp            # 网格 IO、拓扑构建
│   ├── mesh_generator.cpp  # 程序化生成方形布料
│   ├── constraints.cpp     # 约束应用
│   ├── pd_solver.cu        # CUDA: Local/Global Step
│   ├── simulation.cpp      # 主循环
│   ├── viewer/             # Phase 2.5 已实现的 viewer
│   └── utils/
│       ├── obj_io.h/cpp
│       └── cuda_helper.h
└── shaders/                # GLSL 着色器
```

---

## 三、各模块详细设计

### 3.1 Mesh 模块 (已存在，需扩展)

```cpp
// include/mesh.h (新增约束相关)

struct ClothMesh {
    // === CPU 端 ===
    std::vector<Eigen::Vector3f> rest_pos;
    std::vector<Eigen::Vector3i> triangles;

    // 拉伸约束: (v0, v1, rest_length, stiffness)
    std::vector<Eigen::Vector4f> stretch_constraints;

    // 弯曲约束: (v0, v1, v2, v3, rest_angle, stiffness)
    // v0-v1 是共享边，v2/v3 是对顶点
    std::vector<float> bend_rest_angles;
    std::vector<float> bend_stiffness;
    std::vector<Eigen::Vector4i> bend_quads;

    // === GPU 端 ===
    float* d_pos;           // [N×3] 当前位置
    float* d_vel;           // [N×3] 速度
    float* d_prev_pos;      // [N×3] 上一帧位置 (Chebyshev 需要)
    float* d_mass;          // [N] 质量

    // 约束 GPU 数据
    int* d_stretch_edges;   // [E_stretch×2] 顶点索引
    float* d_stretch_rest;  // [E_stretch] 目标长度
    float* d_stretch_k;     // [E_stretch] 刚度

    int* d_bend_quads;      // [E_bend×4] 顶点索引 (v0,v1,v2,v3)
    float* d_bend_rest;     // [E_bend] 目标二面角
    float* d_bend_k;        // [E_bend] 刚度

    // Jacobi 求解辅助
    float* d_mass_lumped;   // [N] M_ii + h² * (Σ w_i * A_i^T A_i)_ii

    int num_verts;
    int num_stretch_cons;
    int num_bend_cons;
};
```

**新增方法**:
- `build_stretch_constraints()` — 从三角形生成边约束
- `build_bend_constraints()` — 从内边生成弯曲约束
- `precompute_jacobi_diagonal()` — 预计算 Jacobi 迭代的分母

### 3.2 Constraints 模块 (已存在)

```cpp
// include/constraints.h

struct Constraints {
    std::vector<int> pinned_indices;

    void pin_top_row(const ClothMesh& mesh, int ncols);
    void pin_corners(const ClothMesh& mesh, int nrows, int ncols);

    // GPU
    int* d_pinned_indices;
    int num_pinned;

    void apply_gpu(float* d_pos, float* d_vel);  // 固定点设为 rest_pos
};
```

### 3.3 PD Solver 模块 (核心)

```cpp
// include/pd_solver.h

struct PDSolverConfig {
    int max_iterations = 50;      // 每帧最大迭代数
    float tolerance = 1e-4f;      // 收敛阈值
    bool use_chebyshev = true;    // 是否启用 Chebyshev 加速
    float omega = 1.0f;           // Jacobi 松弛因子 (Chebyshev 动态调整)
    float gravity = -9.8f;
    float dt = 0.01f;             // 时间步长
};

class PDSolver {
public:
    PDSolver(const PDSolverConfig& config);
    ~PDSolver();

    // 主入口: 执行一帧 PD 迭代
    void step(ClothMesh& mesh, const Constraints& cons);

private:
    PDSolverConfig config_;

    // CUDA kernels
    void predict_positions(ClothMesh& mesh);           // y = x + h*v + h²*f_ext/m
    void local_step_stretch(const ClothMesh& mesh);    // 计算边投影
    void local_step_bend(const ClothMesh& mesh);       // 计算弯曲投影
    void global_step_jacobi(const ClothMesh& mesh);    // Jacobi 迭代
    void update_velocity_and_positions(ClothMesh& mesh); // v = (x_new - x)/h

    // Chebyshev 加速
    float omega_prev_ = 1.0f;
    float omega_curr_ = 1.0f;
    void chebyshev_update(float* d_pos, float* d_prev_pos, int num_verts);
    void reset_chebyshev();

    // 临时 GPU 缓冲区
    float* d_projections_;      // Local step 结果 [总约束数×3]
    float* d_rhs_;              // Global step 右端项 [N×3]
};
```

### 3.4 CUDA Kernels 设计

```cuda
// src/pd_solver.cu

// === Local Step Kernels ===

// 每线程处理一条拉伸约束
__global__ void stretch_projection_kernel(
    const float* __restrict__ pos,
    const int2* __restrict__ edges,
    const float* __restrict__ rest_lengths,
    const float* __restrict__ stiffness,
    float3* __restrict__ projections,  // 输出: 两个投影点
    int num_edges
);

// 每线程处理一条弯曲约束
__global__ void bend_projection_kernel(
    const float* __restrict__ pos,
    const int4* __restrict__ quads,    // (v0, v1, v2, v3)
    const float* __restrict__ rest_angles,
    const float* __restrict__ stiffness,
    float3* __restrict__ projections,  // 输出: 4个投影点
    int num_bends
);

// === Global Step Kernel ===

// Jacobi 迭代: 每个顶点一个线程
__global__ void jacobi_update_kernel(
    const float* __restrict__ pos,      // 当前位置
    const float* __restrict__ predict,  // y (惯性预测)
    const float* __restrict__ mass,     // 质量
    const float* __restrict__ jacobi_diag, // 预计算的 (M + h²L)_ii

    // 拉伸约束贡献
    const int2* __restrict__ stretch_edges,
    const float3* __restrict__ stretch_proj,
    const float* __restrict__ stretch_k,

    // 弯曲约束贡献
    const int4* __restrict__ bend_quads,
    const float3* __restrict__ bend_proj,
    const float* __restrict__ bend_k,

    float* __restrict__ new_pos,        // 输出
    int num_verts,
    float dt
);

// === Constraint Application ===

__global__ void apply_pinned_constraints_kernel(
    float* __restrict__ pos,
    float* __restrict__ vel,
    const int* __restrict__ pinned_indices,
    const float* __restrict__ rest_pos,
    int num_pinned
);
```

---

## 四、实现路线图 (更新版)

### Phase 1-2.5: 已完成
- [x] 基础框架、网格生成、静态可视化

### Phase 3: PD 核心求解器

**新增文件**:
- `include/pd_solver.h`, `src/pd_solver.cu`

**交付物**:
- [ ] `build_stretch_constraints()` — 从三角形生成边约束
- [ ] `build_bend_constraints()` — 从内边生成弯曲约束
- [ ] `precompute_jacobi_diagonal()` — 预计算 Jacobi 分母
- [ ] `stretch_projection_kernel` — 边约束投影
- [ ] `jacobi_update_kernel` — 基础 Jacobi 迭代
- [ ] `predict_positions` / `update_velocity` — 时间积分
- [ ] 纯命令行仿真工具 `sim_cloth` (无渲染，验证正确性)

**验证检查点**:
```bash
# 检查点 3.1: 单根弹簧振荡 (解析解对比)
./sim_cloth --test-spring --steps 1000
# 输出: 周期/振幅与理论值误差 < 1%

# 检查点 3.2: 固定点约束
./sim_cloth --test-pinned --steps 100
# 输出: 固定点位置不变，其余点运动合理

# 检查点 3.3: 能量衰减 (有阻尼)
./sim_cloth --test-damping --steps 500
# 输出: 总能量单调递减
```

---

### Phase 4: 弯曲约束 + Chebyshev 加速

**交付物**:
- [ ] `bend_projection_kernel` — 二面角约束投影
- [ ] `chebyshev_update` — Chebyshev 加速迭代
- [ ] 对比实验: Jacobi vs Chebyshev 收敛速度

**验证检查点**:
```bash
# 检查点 4.1: 纯弯曲测试
./sim_cloth --test-bend-plate --steps 200
# 输出: 板弯曲形状合理

# 检查点 4.2: Chebyshev 加速效果
./sim_cloth --bench --iterations 100
# 输出: Chebyshev 达到相同精度所需迭代次数 < 50% Jacobi

# 检查点 4.3: 悬挂布料
./sim_cloth --hang 20 20 --steps 1000
# 输出: 布料下垂形成自然悬链线形状
```

---

### Phase 5: CUDA Interop + 实时渲染

**修改文件**:
- `src/viewer/viewer.cpp` — 添加 `update_from_cuda()` 接口
- `src/main.cpp` — 接入 PD 求解器

**交付物**:
- [ ] `cudaGraphicsGLRegisterBuffer` 零拷贝 VBO 更新
- [ ] `ClothViewer::update_from_device(float* d_pos)`
- [ ] ImGui 面板: dt、迭代次数、刚度、重力调节
- [ ] 实时主循环

**验证检查点**:
```bash
# 检查点 5.1: 实时仿真
./cuda_ms --cloth 30 30 --hang
# 输出: >30 FPS，布料自然下垂

# 检查点 5.2: 参数调节
# ImGui 调节 Young's modulus，布料刚度实时变化

# 检查点 5.3: 导出序列帧
./cuda_ms --cloth 50 50 --hang --export ./frames/ --steps 300
```

---

### Phase 6: 性能优化 (可选)

**优化方向**:
- [ ] Warp-level 并行优化 (减少 divergence)
- [ ] 共享内存缓存邻接顶点数据
- [ ] A-Jacobi (论文 2022 方法)
- [ ] 多网格/层次化方法

---

## 五、关键算法细节

### 5.1 Jacobi 迭代公式

对每个顶点 `i`:
```
x_i^{new} = (m_i·y_i + h² · Σ_{c∈C_i} w_c · A_c^T p_c) / (m_i + h² · Σ_{c∈C_i} w_c · A_c^T A_c)
```

分母 `diag_i = m_i + h² · Σ w_c` 是常数，预计算存储。

### 5.2 Chebyshev 加速

```
ω_1 = 1
ω_2 = 2/(2 - ρ²)  (ρ 是 Jacobi 迭代矩阵谱半径，通常取 0.9~0.99)

k ≥ 2 时:
ω_{k+1} = 4/(4 - ρ²·ω_k)

更新公式:
x^{k+1} = ω_{k+1} · (Jacobi_update(x^k) - x^{k-1}) + x^{k-1}
```

### 5.3 拉伸约束投影

对边约束 `|x_j - x_i| = L_rest`:
```
center = (x_i + x_j) / 2
dir = (x_j - x_i) / |x_j - x_i|
p_i = center - (L_rest/2) · dir
p_j = center + (L_rest/2) · dir
```

### 5.4 弯曲约束投影 (简化版)

对二面角约束，使用 Grinspun et al. 的离散壳模型:
```
1. 计算当前二面角 θ
2. 计算法向 n1, n2
3. 梯度方向 ∂θ/∂x (4个顶点的权重)
4. 投影: 沿梯度方向旋转顶点使 θ = θ_rest
```

---

## 六、验证方案

| 阶段 | 检查命令 | 通过标准 |
|------|----------|----------|
| 3 | `--test-spring` | 周期/振幅误差 < 1% |
| 3 | `--test-pinned` | 固定点不动 |
| 4 | `--test-bend-plate` | 弯曲形状合理 |
| 4 | `--bench` | Chebyshev 加速比 > 1.5x |
| 4 | `--hang` | 悬链线与参考解对比 |
| 5 | `--cloth 30 30` | >30 FPS |

---

## 七、与原版 FEM 设计的区别

| 方面 | FEM 原版 | PD 新版 |
|------|----------|---------|
| 核心算法 | 显式/隐式 Newton 迭代 | Local-Global 交替迭代 |
| 线性求解 | 每帧构建/求解 K·Δx = f | 固定矩阵，Jacobi 迭代 |
| 拉伸模型 | StVK/Neo-Hookean 本构 | 边长度约束投影 |
| 弯曲模型 | 二面角能量梯度 | 二面角约束投影 |
| GPU 友好度 | 中 (稀疏矩阵) | 高 (完全并行) |
| 收敛性 | 条件稳定/无条件稳定 | 无条件稳定 |
| 实现复杂度 | 高 | 中 |

---

## 八、依赖库

| 库 | 用途 | 状态 |
|----|------|------|
| CUDA Toolkit | GPU 计算 | 已有 |
| Eigen 3 | CPU 预处理 | 已有 |
| GLFW + GLAD + ImGui | 可视化 | Phase 2.5 已完成 |

**无需**: cuSPARSE (PD 不需要稀疏矩阵求解器)

---

*文档 v0.3: 重构为 Projective Dynamics 方案*
