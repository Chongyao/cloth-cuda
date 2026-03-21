# CUDA FEM 布料仿真框架设计文档

> 作者：仿真工程师视角
> 版本：v0.2
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
│   ├── mesh_generator.h    # 程序化网格生成
│   ├── constraints.h       # 约束管理
│   ├── material.h          # 本构模型参数与接口
│   ├── fem_engine.h        # FEM 力/刚度计算接口
│   ├── bending.h           # 弯曲力计算接口
│   ├── sparse_matrix.h     # GPU 稀疏矩阵（CSR）
│   ├── linear_solver.h     # GPU 共轭梯度求解器
│   ├── integrator.h        # 时间积分器接口
│   ├── simulation.h        # 顶层仿真管理器
│   └── viewer.h            # OpenGL 可视化
├── src/
│   ├── CMakeLists.txt
│   ├── mesh.cpp            # 网格 IO、拓扑构建
│   ├── mesh_generator.cpp  # 程序化生成方形布料
│   ├── constraints.cpp     # 约束应用
│   ├── fem_engine.cu       # CUDA 核函数：力/刚度
│   ├── bending.cu          # CUDA 核函数：弯曲力
│   ├── sparse_matrix.cu    # CSR 构建
│   ├── linear_solver.cu    # PCG 求解器
│   ├── integrator.cu       # 显式/隐式积分器
│   ├── simulation.cpp      # 主循环
│   ├── viewer.cpp          # OpenGL 渲染
│   └── utils/              # 工具函数
│       ├── obj_io.h/cpp
│       └── cuda_helper.h
└── shaders/                # GLSL 着色器
    ├── phong.vs
    └── phong.fs
```

---

## 三、各模块详细设计

### 3.1 Mesh 模块

#### 数据结构

```cpp
// include/mesh.h

struct ClothMesh {
    // --- CPU 端原始数据（预处理用）---
    std::vector<Eigen::Vector3f> rest_pos;    // 参考构型顶点
    std::vector<Eigen::Vector3i> triangles;   // 三角形顶点索引
    std::vector<Eigen::Vector4i> inner_edges; // 内边（弯曲用）：(v0, v1, v2, v3)
                                              // v0-v1 是共享边，v2/v3 是对顶点

    // --- 预计算量（每个三角形）---
    std::vector<Eigen::Matrix2f> Dm_inv;      // D_m 的逆 (2×2)
    std::vector<float>           rest_area;   // 参考构型面积
    std::vector<float>           mass;        // 节点质量

    // --- 弯曲预计算 ---
    std::vector<float>           rest_angles; // 参考二面角
    std::vector<float>           edge_lengths;// 共享边长度

    // --- GPU 端镜像 ---
    float*  d_pos;          // [N×3]
    float*  d_vel;          // [N×3]
    float*  d_force;        // [N×3]
    int*    d_tris;         // [T×3]
    int*    d_inner_edges;  // [E×4]
    float*  d_Dm_inv;       // [T×4]
    float*  d_rest_area;    // [T]
    float*  d_mass;         // [N]
    float*  d_rest_angles;  // [E]
    float*  d_edge_lengths; // [E]

    int num_verts;
    int num_tris;
    int num_inner_edges;
};
```

#### 关键操作

- `load_obj(path)` — 读取 OBJ 文件
- `generate_square_cloth(nrows, ncols, size)` — 程序化生成规则网格
- `build_inner_edges()` — 构建内边表（边→两个三角形映射）
- `precompute_rest_state()` — 计算 Dm_inv、rest_area、mass
- `precompute_bending_state()` — 计算 rest_angles、edge_lengths
- `upload_to_gpu()` — 数据传送到 GPU

### 3.2 Constraints 模块

```cpp
// include/constraints.h

struct Constraints {
    std::vector<int> pinned_indices;    // 固定顶点索引
    std::vector<float> target_pos;      // 目标位置 [N×3]

    // 预设边界条件
    void pin_one_side(const ClothMesh& mesh, int ncols);      // 固定第一行
    void pin_corners(const ClothMesh& mesh, int ncols);       // 固定对角
    void pin_top_row(const ClothMesh& mesh, int ncols);       // 固定顶行
    void set_from_list(const std::vector<int>& indices);

    // GPU 应用
    void upload_to_gpu();
    void apply_gpu(float* d_pos, float* d_vel, float* d_force);
};
```

### 3.3 Material 模块

```cpp
// include/material.h

enum class MaterialModel { StVK, NeoHookean };

struct MaterialParams {
    MaterialModel model = MaterialModel::StVK;
    float young_modulus = 1e5f;
    float poisson_ratio = 0.3f;
    float density       = 0.1f;
    float damping_alpha = 0.01f;
    float damping_beta  = 0.001f;

    float lambda() const;
    float mu() const;
};

struct BendingParams {
    float stiffness = 1e-3f;
};
```

### 3.4 FEM Engine 模块

```cuda
// 弹性力计算（显式）
__global__ void compute_elastic_forces(
    const float* __restrict__ pos,
    const int*   __restrict__ tris,
    const float* __restrict__ Dm_inv,
    const float* __restrict__ rest_area,
    float*       forces,
    int num_tris,
    MaterialParams params
);

// 刚体运动零力验证辅助函数
__device__ float3 compute_stvk_force(...);
```

### 3.5 Bending 模块

```cuda
__global__ void compute_bending_forces(
    const float* __restrict__ pos,
    const int*   __restrict__ inner_edges,
    const float* __restrict__ rest_angles,
    const float* __restrict__ edge_lengths,
    float*       forces,
    int num_inner_edges,
    BendingParams params
);
```

### 3.6 Integrator 模块

```cpp
// include/integrator.h

class Integrator {
public:
    virtual void step(ClothMesh& mesh, const Constraints& cons, float dt) = 0;
    virtual ~Integrator() = default;
};

class ExplicitVerlet : public Integrator {
public:
    void step(ClothMesh& mesh, const Constraints& cons, float dt) override;
private:
    // 工作缓冲区
    float* d_force;
};

class ImplicitEuler : public Integrator {
    // ... PCG 求解器
};
```

#### 显式 Verlet 流程

```
1. kernel: clear_forces
2. kernel: compute_elastic_forces
3. kernel: compute_bending_forces
4. kernel: add_gravity
5. kernel: apply_damping
6. kernel: integrate_verlet
7. kernel: apply_constraints
```

### 3.7 Viewer 模块（OpenGL）

```cpp
// include/viewer.h

class ClothViewer {
public:
    bool init(int width, int height, int num_verts, int num_tris);
    void update_positions(const float* d_pos);  // CUDA device pointer
    void draw(const ClothMesh& mesh);
    void poll_events();
    bool should_close() const;
    void cleanup();

    // GUI 状态（绑定到 ImGui）
    float gui_dt = 1e-3f;
    float gui_young_modulus = 1e5f;
    bool  gui_running = true;
    bool  gui_step = false;
};
```

---

## 四、实现路线图与检查点

### Phase 1: 基础框架 ✓ (已完成)

**交付物**:
- [x] 分层 CMakeLists.txt（CUDA 自动探测，CPU-only 降级）
- [x] ClothMesh 数据结构（CPU Eigen + GPU 指针）
- [x] OBJ 加载器
- [x] Dm_inv、面积、质量预计算
- [x] GPU 数据上传

**验证检查点**:
```bash
./cuda_ms
# 输出: Mass verification PASS
# 输出: Dm_inv identity check PASS
```

---

### Phase 2: 网格生成与拓扑

**新增文件**:
- `include/mesh_generator.h`, `src/mesh_generator.cpp`
- `include/constraints.h`, `src/constraints.cpp`
- `src/mesh.cpp` 扩展 `build_inner_edges()`

**交付物**:
- [ ] `generate_square_cloth(nrows, ncols, size, type)` — 规则网格生成
  - type 0: 统一对角线方向
  - type 1: 交替对角线（棋盘式）
- [ ] `build_inner_edges()` — 内边拓扑构建
- [ ] `Constraints` 类 — 固定约束管理
  - [ ] `pin_one_side()`, `pin_corners()`, `pin_top_row()`

**验证检查点**:
```bash
# 检查点 2.1
./cuda_ms --gen-cloth 10 10 0.1
# 输出: 100 vertices, 162 triangles
# 输出: Inner edges: 81, Boundary edges: 38

# 检查点 2.2
./cuda_ms --test-constraints pin_top_row
# 输出: 10 vertices pinned
```

---

### Phase 3: 显式积分器 + 膜力 (StVK)

**新增文件**:
- `include/fem_engine.h`, `src/fem_engine.cu`
- `include/integrator.h`, `src/integrator.cu`
- `include/simulation.h`, `src/simulation.cpp`

**交付物**:
- [ ] `compute_stvk_forces_kernel` — 每线程一个三角形
- [ ] `verlet_integrate_kernel` — 时间积分
- [ ] `apply_constraints_kernel` — 约束应用
- [ ] `ExplicitVerlet` 类 — 完整显式积分器
- [ ] `Simulation` 类 — 顶层管理

**验证检查点**:
```bash
# 检查点 3.1: 单三角形拉伸（解析解对比）
./cuda_ms --test-stretch-single-tri
# 输出: Force error < 1%

# 检查点 3.2: 刚体运动零力
./cuda_ms --test-rigid-motion
# 输出: Max force < 1e-6

# 检查点 3.3: 自由落体
./cuda_ms --free-fall 100
# y(t) 与 y0 - 0.5*g*t^2 对比，误差 < 0.1%

# 检查点 3.4: 悬挂布料（无弯曲）
./cuda_ms --hang-membrane --steps 2000
# 输出: 能量收敛，形状合理
```

---

### Phase 4: 弯曲力 (Discrete Hinge)

**新增文件**:
- `include/bending.h`, `src/bending.cu`

**交付物**:
- [ ] `compute_bending_forces_kernel` — 每线程一条内边
  - [ ] 二面角计算（atan2 方式，arborecence 风格）
  - [ ] 能量梯度 ∂E/∂x
  - [ ] 4 顶点原子累加
- [ ] `precompute_bending_state()` — 参考二面角预计算

**验证检查点**:
```bash
# 检查点 4.1: 纯弯曲测试
./cuda_ms --test-pure-bend
# 输出: 弯矩-曲率关系符合理论

# 检查点 4.2: 悬挂布料（膜+弯曲）
./cuda_ms --hang-cloth --steps 2000
# 输出: 与 arborecence 结果对比，位移误差 < 5%

# 检查点 4.3: 能量守恒（无阻尼）
./cuda_ms --energy-conservation --steps 1000
# 输出: |E(t) - E(0)| / E(0) < 1e-4
```

---

### Phase 5: 实时可视化 (OpenGL)

**新增文件**:
- `include/viewer.h`, `src/viewer.cpp`
- `shaders/phong.vs`, `shaders/phong.fs`
- `src/main_viz.cpp`

**交付物**:
- [ ] `ClothViewer` 类 — GLFW + GLAD 初始化
- [ ] `cudaGraphicsGLRegisterBuffer` 零拷贝
- [ ] Phong 光照渲染
- [ ] 实时仿真循环
- [ ] 键盘/鼠标控制（旋转、缩放）
- [ ] 可选：ImGui 参数面板

**验证检查点**:
```bash
# 检查点 5.1: 静态网格显示
./cuda_ms_viz --mesh test.obj
# 应显示窗口，可旋转视角

# 检查点 5.2: 实时仿真
./cuda_ms_viz --sim --cloth 20 20
# 应显示布料实时下落变形，>30 FPS

# 检查点 5.3: 帧输出
./cuda_ms --offline --save-frames ./frames/ --steps 100
# 生成 frames/frame_*.ply
```

---

### Phase 6: 隐式积分（可选/进阶）

**新增文件**:
- `include/sparse_matrix.h`, `src/sparse_matrix.cu`
- `include/linear_solver.h`, `src/linear_solver.cu`
- `include/implicit_integrator.h`, `src/implicit_integrator.cu`

**交付物**:
- [ ] CSR 稀疏矩阵构建
- [ ] 刚度矩阵 kernel
- [ ] PCG 求解器（cuSPARSE + Jacobi 预条件）
- [ ] `ImplicitEuler` 类

**验证检查点**:
```bash
# 检查点 6.1: 大时间步稳定性
./cuda_ms --implicit --dt 0.01 --steps 100
# 显式此时应发散，隐式应稳定

# 检查点 6.2: 与显式对比
./cuda_ms --compare --dt 0.001 --steps 1000
# 隐式/显式结果应接近
```

---

## 五、验证方案汇总

| 阶段 | 检查命令 | 通过标准 |
|------|----------|----------|
| 1 | `./cuda_ms` | Mass/Dm_inv PASS |
| 2 | `--gen-cloth 10 10` | 顶点/三角形/内边数正确 |
| 3 | `--test-stretch-single-tri` | 力误差 < 1% |
| 3 | `--free-fall 100` | 位移匹配 y = 0.5*g*t^2 |
| 4 | `--hang-cloth` | 与参考解对比 |
| 5 | `--viz --sim` | 实时 >30 FPS |
| 6 | `--implicit --dt 0.01` | 大时间步稳定 |

---

## 六、依赖库

| 库 | 用途 | 是否必须 |
|----|------|----------|
| CUDA Toolkit | GPU 计算 | 必须 |
| cuSPARSE | 稀疏矩阵（隐式阶段） | Phase 6 |
| Eigen 3 | CPU 预处理 | 必须 |
| GLFW | 窗口/输入（可视化） | Phase 5 |
| GLAD | OpenGL 加载（可视化） | Phase 5 |
| Dear ImGui | GUI（可选） | Phase 5 |

---

*文档 v0.2：整合 ROADMAP，添加详细检查点*
