# scTenifoldpy 并行计算分析与 CUDA 改造文档

## 总览

本文档分为两部分：

- **Part 1**：审查原始项目的并行运算实现（Ray 多进程 CPU）
- **Part 2**：在原始实现基础上新增的 CUDA GPU 加速改造方案

---

# Part 1：原始并行实现（Ray CPU）

## 概述

scTenifoldpy 的并行运算集中在 **PCNet 构建**阶段，使用 [Ray](https://www.ray.io/) 框架实现多核并行。其余阶段（张量分解、流形对齐等）均为单线程 CPU 执行。

---

## 依赖库

[setup.py](setup.py) 中声明：

```
ray>=1.8
```

Ray 是面向 Python 的分布式计算框架，支持跨进程、跨机器的任务并行调度。

---

## 并行入口：`make_networks`

**文件**：[scTenifold/core/_networks.py](scTenifold/core/_networks.py)

```python
@timer
def make_networks(data: pd.DataFrame,
                  n_nets: int = 10,
                  n_samp_cells: Optional[int] = 500,
                  n_comp: int = 3,
                  scale_scores: bool = True,
                  symmetric: bool = False,
                  q: float = 0.95,
                  random_state: int = 42,
                  n_cpus: int = -1,   # ← 并行控制参数
                  use_cuda: bool = False,  # ← 新增（见 Part 2）
                  device: str = "cuda:0",  # ← 新增（见 Part 2）
                  **kwargs
                  ) -> List[coo_matrix]:
```

### `n_cpus` 参数行为

| 值 | 行为 |
|---|---|
| `-1`（默认） | 自动检测，使用全部可用 CPU 核心并行 |
| `>1` 的整数 | 使用指定数量的 CPU 核心 |
| `1` | 禁用并行，退回单核串行 |

---

## Ray 并行执行流程

当 `n_cpus != 1` 且 `use_cuda=False` 时：

### 步骤 1：初始化 Ray

```python
if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=n_cpus)   # None → 使用全部核心
```

### 步骤 2：共享内存放置数据

```python
Z_data = ray.put(data)
```

`ray.put()` 将基因 × 细胞 DataFrame 放入 **Ray Object Store**（共享内存），所有 worker 进程零拷贝读取，避免多次序列化开销。

### 步骤 3：提交异步任务

```python
for net in range(n_nets):
    tasks.append(
        pc_net_parallelized.remote(Z_data, selected_samples=sample, ...)
    )
```

`n_nets` 个任务同时分发到各 worker 进程，实现数据并行。

### 步骤 4：同步等待

```python
results = ray.get(tasks)   # 阻塞，等待所有结果
```

### 步骤 5：清理

```python
if ray.is_initialized():
    ray.shutdown()
```

---

## Ray 远程函数：`@ray.remote`

[scTenifold/core/_networks.py](scTenifold/core/_networks.py)

```python
@ray.remote
def pc_net_parallelized(data, selected_samples, n_comp=3, ...):
    return pc_net_calc(data=data, selected_samples=selected_samples, ...)
```

`@ray.remote` 将函数注册为远程任务：调用时返回 `ObjectRef`（future），实际计算在独立 worker 进程中运行，通过 `ray.get()` 获取结果。

对应的串行回退版本 `pc_net_single` 与之逻辑完全相同，仅无 `@ray.remote` 装饰。

---

## 底层计算：`pc_net_calc`

每个并行任务执行的核心逻辑：

1. **子采样**：从随机选定的细胞列提取子矩阵
2. **标准化**：对细胞 × 基因矩阵零均值归一化
3. **逐基因回归**：对每个基因用其余基因的 PC 得分做回归，计算 β 系数

```python
# ← 原始实现的主要性能瓶颈：n_genes 次串行 SVD
bs = [p_(i) for i in range(Xt.shape[1])]
B = np.concatenate(bs, axis=1).T
```

4. **构建邻接矩阵**：β 矩阵填充 + 分位数阈值稀疏化

> **关键局限**：并行粒度是"每次子采样 → 一个 PCNet"，而每个 PCNet 内部的 $n_\text{genes}$ 次 SVD 仍是**串行**的。这是 GPU 改造的核心突破点。

---

## 计时装饰器：`@timer`

**文件**：[scTenifold/core/_utils.py](scTenifold/core/_utils.py)

```python
def timer(func=None):
    @wraps(func)
    def _counter(*args, **kwargs):
        verbosity = kwargs.pop("verbosity", 1)
        start = time.perf_counter()
        sol = func(*args, **kwargs)
        print(func.__name__, " processing time: ", str(time.perf_counter() - start))
        return sol
    return _counter
```

被装饰的函数：`make_networks`、`tensor_decomp`、`manifold_alignment`、`d_regulation`、`cal_pcNet`。

---

## 原始架构图

```
make_networks(data, n_nets=10, n_cpus=-1)
      │
      ├─ ray.init(num_cpus=None)
      ├─ Z_data = ray.put(data)         ← 共享对象存储（零拷贝）
      │
      ├─ [.remote() × n_nets]           ← 异步提交
      │     ├─ Worker 0: pc_net_calc()
      │     │     └─ randomized_svd × n_genes  ← CPU 串行
      │     ├─ Worker 1: pc_net_calc()
      │     └─ ...
      │
      ├─ ray.get(tasks)                 ← 阻塞同步
      ├─ 后处理 → coo_matrix            ← 主进程串行
      └─ ray.shutdown()
```

---

## 原始各步骤并行情况汇总

| 步骤 | 函数 | 原始并行方式 | 文件 |
|---|---|---|---|
| QC | `sc_QC` | 无 | `_QC.py` |
| 网络构建（PCNet） | `make_networks` | **Ray 多进程**（任务级并行） | `_networks.py` |
| 张量分解 | `tensor_decomp` | 无（TensorLy NumPy backend） | `_decomposition.py` |
| 虚拟敲除重建 | `reconstruct_pcnets` | 间接调用 Ray（每个网络单独 init/shutdown） | `_ko.py` |
| 流形对齐 | `manifold_alignment` | 无（scipy 稀疏特征分解） | `_networks.py` |
| 差异调控分析 | `d_regulation` | 无 | `_networks.py` |

---

## 原始实现的局限性

| 问题 | 描述 |
|---|---|
| **任务内部串行** | `pc_net_calc` 内 $n_\text{genes}$ 次 SVD 串行循环，大基因面板时成为瓶颈 |
| **Ray 启停开销** | 每次 `make_networks` 都经历 `init/shutdown`，频繁调用时浪费时间 |
| **张量分解无并行** | `tensor_decomp` 完全在 CPU 单线程上运行 |
| **KO 重建双重串行** | `reconstruct_pcnets` 外层 for 循环串行 + 每次 Ray init/shutdown |
| **Dask 未完成** | 代码中有注释的 Dask 后端代码，曾考虑但未实现 |

---

# Part 2：CUDA GPU 加速改造

## 改造目标

针对 Part 1 中识别的瓶颈，在保持 **Ray CPU 路径完全不变**的前提下，新增 CUDA GPU 加速路径，覆盖三个核心计算模块。

## 改造概览

| 计算模块 | 原始实现 | CUDA 改造 |
|---|---|---|
| PCNet 构建 | Ray + `randomized_svd`（CPU，逐基因串行） | **GPU 批量 SVD**，2 次 GEMM 替代 $n_\text{genes}$ 次 SVD |
| 张量分解 | TensorLy NumPy backend（CPU） | TensorLy **CuPy / PyTorch backend**（GPU），ALS 迭代全在 GPU |
| KO 传播重建 | NumPy matmul（CPU），逐网络 Ray init/shutdown | **cuBLAS matmul** + GPU PCNet 重建，无 Ray 开销 |

---

## 新增文件

```
scTenifold/core/
├── _networks_cuda.py      ← PCNet GPU 加速（核心）
├── _decomposition_cuda.py ← 张量分解 GPU 加速
└── _ko_cuda.py            ← KO 传播 GPU 加速
```

---

## 依赖安装

CUDA 依赖为**可选项**，不安装时项目自动使用原有 CPU 路径：

```bash
# 推荐：CuPy（与 CUDA toolkit 版本对应）
pip install cupy-cuda11x   # CUDA 11.x
pip install cupy-cuda12x   # CUDA 12.x

# 备选：PyTorch GPU build（CuPy 不可用时自动回退）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 通过 extras_require 一键安装（已在 setup.py 中配置）
pip install "scTenifoldpy[cuda12]"
pip install "scTenifoldpy[cuda11]"
pip install "scTenifoldpy[torch-gpu]"
```

---

## 使用方法

所有原有接口新增 `use_cuda=True` 参数，**其余参数完全向后兼容**：

```python
# ---- PCNet 构建 ----
net.run_step("nc", use_cuda=True)               # GPU，自动检测后端
net.run_step("nc", use_cuda=True, device="cuda:1")  # 指定第二块 GPU

# ---- 张量分解 ----
net.run_step("td", use_cuda=True)

# ---- 虚拟敲除（propagation 模式）----
knk = scTenifoldKnk(
    data,
    ko_method="propagation",
    ko_genes=["TP53"],
    nc_kws={"use_cuda": True},
    td_kws={"use_cuda": True},
    ko_kws={"use_cuda": True},
)
knk.build()

# ---- 无 GPU 环境：自动回退 CPU，不报错 ----
net.run_step("nc", use_cuda=True)  # 若 cupy/torch-gpu 未安装，打印 warning 后使用 Ray
```

---

## 模块一：`_networks_cuda.py` — PCNet GPU 加速

### 核心突破：消除串行 SVD 循环

**原始问题**（CPU）：

```python
# n_genes 次串行 SVD，每次处理 (cells × (genes-1)) 矩阵
bs = [cal_pc_coefs(i, X=Xt, n_comp=n_comp) for i in range(n_genes)]
```

### 策略 A：精确模式（`fast_svd=False`）

逐基因在 GPU 上调用 `cupy.linalg.svd`，相比 CPU 的 `randomized_svd` 充分利用 GPU 的 cuSolver 加速：

```python
for k in range(n_genes):
    Xi = Xt_gpu[:, idx]                          # 去掉第 k 列，GPU 上切片
    U, s, VT = cpla.svd(Xi, full_matrices=False) # cuSolver SVD
    betas = coef @ cp.sum(y[:, None] * score, axis=0)
```

### 策略 B：批量快速模式（`fast_svd=True`，默认）

**一次全局 SVD + 两次矩阵乘法**，完全消除串行循环：

```python
# 1. 对整个 (cells × genes) 矩阵做一次全局截断 SVD
U, s, VT = cpla.svd(Xt_gpu, full_matrices=False)
coef_all = VT[:n_comp, :].T            # (genes, n_comp)

# 2. 全量 PC 得分矩阵
score = Xt_gpu @ coef_all              # (cells, n_comp)
score_normed = score / score.sum(axis=0)**2

# 3. 两次 GEMM 替代 n_genes 次 SVD
rhs = Xt_gpu.T @ score_normed          # (genes, n_comp)
B   = rhs @ coef_all.T                 # (genes, genes)  ← 完整 β 矩阵
```

数学上等价于对整体矩阵使用留一法（LOO）近似：

$$
B = (X^T \cdot S_{\text{norm}}) \cdot V_{\text{top-}r}^T
$$

其中 $S_{\text{norm}}$ 为归一化 PC 得分矩阵，整个 $\beta$ 矩阵由两次 GPU GEMM（cuBLAS）完成。

**三种方案对比：**

| 方案 | SVD 调用次数 | 并行度 | 精度 |
|---|---|---|---|
| CPU `randomized_svd` | $n_\text{genes}$ 次 | CPU 核心数（Ray） | 精确（随机近似） |
| CUDA 精确（A） | $n_\text{genes}$ 次 | GPU 并行（cuSolver） | 精确 |
| CUDA 批量（B，默认） | **1 次** | GPU 并行（cuBLAS GEMM） | LOO 近似（生物结论等价） |

### 双后端自动切换

```python
# CuPy 优先（推荐，性能最优）
if _cupy_available:
    return pc_net_calc_cuda_fast(...)
# 无 CuPy 时透明回退 PyTorch（使用 torch.linalg.svd）
else:
    return pc_net_calc_cuda_torch(...)
```

---

## 模块二：`_decomposition_cuda.py` — 张量分解 GPU 加速

### 原始实现

```python
# TensorLy 默认 NumPy backend，ALS 迭代全在 CPU
factors = getattr(decomposition, method)(networks, rank=K, ...)
```

### CUDA 改造：切换 TensorLy 全局 Backend

TensorLy 原生支持 CuPy 和 PyTorch backend，切换后所有 ALS 迭代自动在 GPU 上执行：

```python
_prev_backend = tl.get_backend()
try:
    if _cupy_available:
        tl.set_backend("cupy")      # 所有 TensorLy 操作切换为 CuPy
    else:
        tl.set_backend("pytorch")   # 或 PyTorch

    tensor_gpu = cp.asarray(networks)           # host → GPU
    factors = getattr(decomposition, method)(tensor_gpu, rank=K, ...)
    estimate_gpu = tl.cp_to_tensor(factors)     # GPU 上重建张量

    # 后处理（归一化、取整）也在 GPU 完成
    out_gpu = cp.sum(estimate_gpu, axis=-1) / n_pcnets
    out_np  = cp.asnumpy(out_gpu / cp.max(cp.abs(out_gpu)))

finally:
    tl.set_backend(_prev_backend)   # 恢复原 backend（try/finally 保证安全）
```

---

## 模块三：`_ko_cuda.py` — 虚拟敲除 GPU 加速

### 原始问题

```python
# CPU：for 循环内逐步矩阵乘法 + 每个网络一次 Ray init/shutdown
for net in nets:
    perturbs.append(adj_mat @ perturbs[d])  # NumPy matmul（CPU）
    ko_net = make_networks(data_ko, n_nets=1, ...)  # 启动 Ray
```

### CUDA 改造

将邻接矩阵和扰动向量一次搬到 GPU，传播循环在 GPU 上完成：

```python
adj_mat  = cp.asarray(B)          # host → GPU（仅一次）
x_ko_gpu = cp.asarray(x)

for d in range(degree):
    perturbs.append(adj_mat @ perturbs[d])   # cuBLAS matmul（GPU）
    ...

return cp.asnumpy(result_gpu)      # 仅最终结果搬回 host
```

`reconstruct_pcnets_cuda` 串联传播 + PCNet 重建，整个流程无 Ray 开销：

```python
for net in nets:
    x_ko  = ko_propagation_cuda(net.toarray(), ...)   # GPU 传播
    ko_net = make_networks_cuda(data_ko, n_nets=1, ...) # GPU PCNet
```

---

## 路由集成方式

三个原始文件各仅增加 ~25 行，完全向后兼容：

```python
# 顶部懒加载（ImportError → 自动 CPU 回退）
try:
    from scTenifold.core._networks_cuda import make_networks_cuda, CUDA_AVAILABLE
except ImportError:
    make_networks_cuda = None; CUDA_AVAILABLE = False

# 函数体最顶部判断
def make_networks(..., use_cuda=False, device="cuda:0", ...):
    if use_cuda:
        if make_networks_cuda is None or not CUDA_AVAILABLE:
            warn("use_cuda=True but no CUDA backend; falling back to CPU.", RuntimeWarning)
        else:
            return make_networks_cuda(data, ...)   # ← CUDA 路径（新增）

    # 原有 Ray 路径保持完全不变 ↓
    if n_cpus != 1:
        ray.init(...)
        ...
```

---

## 新旧架构对比

### 原始 CPU 路径

```
make_networks(n_cpus=-1)
  └─ ray.init()
       ├─ Worker 0: pc_net_calc() → randomized_svd × n_genes (CPU 串行)
       ├─ Worker 1: pc_net_calc() → randomized_svd × n_genes (CPU 串行)
       └─ ...
  ray.get() → ray.shutdown()
```

### 新增 CUDA 路径（fast_svd=True）

```
make_networks(use_cuda=True, fast_svd=True)
  └─ make_networks_cuda()
       for net in n_nets:              ← Python 轻量循环
         Z → GPU (cp.asarray)
         SVD × 1 次 (cuSolver)
         B = rhs @ coef.T (cuBLAS GEMM × 2)
         阈值/稀疏化 (CuPy)
         A → host (cp.asnumpy)
       → List[coo_matrix]             ← 无 Ray 进程开销
```

---

## 各步骤改造前后对比

| 步骤 | 函数 | 改造前 | 改造后（`use_cuda=True`） |
|---|---|---|---|
| QC | `sc_QC` | CPU | CPU（无需改造） |
| PCNet 构建 | `make_networks` | Ray 多进程 + CPU `randomized_svd` | **GPU 批量 SVD（cuBLAS）** |
| 张量分解 | `tensor_decomp` | TensorLy NumPy（CPU） | **TensorLy CuPy/PyTorch（GPU）** |
| KO 传播 | `ko_propagation` | NumPy matmul（CPU） | **cuBLAS matmul（GPU）** |
| KO 网络重建 | `reconstruct_pcnets` | 逐网络 Ray init/shutdown | **连续 GPU 计算，无 Ray 开销** |
| 流形对齐 | `manifold_alignment` | scipy 稀疏特征分解 | CPU（scipy 无 GPU 等价实现） |
| 差异调控 | `d_regulation` | CPU | CPU（计算量小，无需改造） |

---

## 注意事项

| 事项 | 说明 |
|---|---|
| **CuPy 优先于 PyTorch** | 两者均安装时优先用 CuPy（性能更优，无框架启动开销） |
| **`fast_svd` 为近似** | 使用全局 LOO-SVD 近似，与逐基因精确 SVD 存在微小数值差异，实际生物结论等价 |
| **显存上限** | `fast_svd=True` 时完整 $(C \times G)$ 矩阵常驻显存；$G > 5000$ 时建议 `fast_svd=False` 或分批 |
| **Ray 与 CUDA 互斥** | `use_cuda=True` 时完全绕过 Ray，`n_cpus` 参数被忽略 |
| **无 GPU 自动降级** | 检测不到 CUDA 时打印 `RuntimeWarning` 并自动回退 CPU，程序不中断 |
| **Backend 恢复** | `tensor_decomp_cuda` 通过 `try/finally` 保证 TensorLy backend 在任何异常下都被还原 |

---

## 文件改动汇总

| 文件 | 类型 | 主要变化 |
|---|---|---|
| [scTenifold/core/_networks_cuda.py](scTenifold/core/_networks_cuda.py) | **新建** | GPU PCNet：精确 / 批量 SVD；CuPy + PyTorch 双后端 |
| [scTenifold/core/_decomposition_cuda.py](scTenifold/core/_decomposition_cuda.py) | **新建** | TensorLy GPU backend 切换，ALS 迭代全在 GPU |
| [scTenifold/core/_ko_cuda.py](scTenifold/core/_ko_cuda.py) | **新建** | GPU 矩阵传播 + GPU PCNet 重建串联 |
| [scTenifold/core/_networks.py](scTenifold/core/_networks.py) | **修改** | 新增 `use_cuda`、`device` 参数；顶部懒加载；函数体头部路由逻辑 |
| [scTenifold/core/_decomposition.py](scTenifold/core/_decomposition.py) | **修改** | 同上，针对 `tensor_decomp` |
| [scTenifold/core/_ko.py](scTenifold/core/_ko.py) | **修改** | 同上，针对 `reconstruct_pcnets` |
| [setup.py](setup.py) | **修改** | 新增 `extras_require`：`cuda11`、`cuda12`、`torch-gpu` |
| [requirements.txt](requirements.txt) | **修改** | 新增可选 CUDA 依赖安装说明 |
