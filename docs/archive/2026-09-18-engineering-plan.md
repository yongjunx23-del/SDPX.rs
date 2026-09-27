# SDPX.rs：面向高精度与多核/分布式的工程执行方案

审阅日期：2026-09-18（Asia/Singapore）  
审阅基线：`yongjunx23-del/SDPX.rs@0c04025960fc5af5822fca62f0d2bb342ef26e56`  
实现参考：`davidsd/sdpb@fec8e934bf03eb59b0f35ad76dd9b205dde537e6`，tag `3.1.0`  
性质：针对当前源码的静态审阅、数学设计和待执行工程任务；不是已经完成的求解器改造或性能认证。

> 总决策：保留统一的 Julia → Rust HSD/predictor-corrector → 数值后端架构。先建立当前版本的正确性与成本证据，再增加精确 CRT 矩阵乘法、多右端项数据流、按结构选择的高精度分块分解以及可选 MPI 后端。不要把“3–10 倍”“减少 50%–60%”“64 核线性扩展”作为已经成立的事实或强制优化目标。

本文 `[R…]`、`[S…]` 对应 `SOURCES.md` 中固定版本的来源；没有标为来源事实的接口、目录、任务编号和指标是本方案建议，尚未在 SDPX 中实现。

## 0. 审阅边界与实施约束

本次通过 GitHub 读取当前分支身份、工作约定、源码关键路径、研究/构建协议和相关性能记录，并核对 SDPB 的 CRT/Q 计算实现。重点读取 `kktsystem.rs`、主迭代循环、KKT trait、`dense_block.rs`、`mpfr.rs`、`arithmetic/src/lib.rs` 和 condensed 结构选择入口。大型文件是定向范围阅读，不声称全仓逐行审查。

没有在当前环境编译、运行 SDPX，也没有重跑仓库报告的 Ising、Lambda11、64 核或 MPI 实验。仓库文档引用的外部 `/tmp` 实验原始文件没有随 GitHub 内容取得，因此相关数字属于“仓库记录”，而不是本次独立复现实测。包内 Python 程序只验证独立的数学参考实现。

执行 agent 必须先读目标工作区现行 `AGENTS.md`。若 HEAD 已变化，先做版本差异与已完成任务核查，不能直接把本计划的路径/缺口当成最新事实。不要覆盖未提交文件；默认主 agent 顺序执行，不自动启动 subagents、提交远程代码或提交集群作业。实际远程计算和发布另按本轮授权。[R1][R2][R10]

### 0.1 不得改变的数值契约

维持工作精度、原坐标返回、accepted-iterate recovery、不可行判定、regularization/refinement、direct/prepared 两类预处理语义。不得放宽收敛/外部验算门槛、把 AlmostSolved/AlmostOptimal 算作完整精度成功、按 benchmark 名称选算法，或通过近似降秩和精度阶梯取得速度。[R2][R3]

当前项目运行时检查沿用 Clarabel 体系；不要重新加入已退役的独立 certificate 阶段、状态提升或 SDPX 专用五方程“纠正门”。原始线性方程的 refinement 属于现有数值求解，继续保留；独立最终原坐标验算放在测试/benchmark 计时之外。这两者不能混淆。[R2][R8]

高精度 NT scaling 已恢复直接 SVD，不能再为省算力改成先算 `MᵀM` 再特征分解并钳制负特征值。现有病态 SPD 反例必须保留。[R3][R9]

### 0.2 四阶段修订

| 用户设想 | 审阅结论 | 可实施目标 |
|---|---|---|
| CRT + BLAS，单核 3–10 倍、打满 64 核 | 算子方向正确，收益依赖尺寸、精度、指数跨度和实际热点；不能直接外推全求解 | 同精度输入的精确 dyadic 整数化、模 BLAS、CRT 重构与明确舍入语义；按实测成本派发 |
| KKT 三次变一次，减算 50%–60% | 当前已经共享一次矩阵分解；三个 RHS 中 combined 依赖 affine | 首选一个分解、三个数学 RHS、两个提交波次；复用迭代内工作与原始 refinement |
| DAG 稠密 Cholesky 全面替换 QDLDL | 完整 KKT 不定，不能直接 Cholesky；全稠密化可能增内存/算力 | 泛型化已有 DenseBlockSolver，先识别正定连通块与负边框，再按结构选择 tile DAG；保留 LDL |
| MPI + 2D Cholesky，百千核扩展 | 可作为大模型目标，但仅替换分解不足以突破内存 | 输入、局部块、全局边框、RHS、验算、I/O 都具备分布式 ownership 与有界内存 |

建议执行顺序是 **Phase 0 → Phase 2 的低风险数据流 + Phase 1 算子原型 → Phase 3 → Phase 4**。Phase 1/2 逻辑独立，并不要求使用多个 agent 同时改代码。

## 1. 当前代码的实际起点

### 1.1 已经存在的东西，不要重复开发

当前有统一 predictor/corrector、augmented/condensed KKT、sampled operator、精确重复列复用、已有 Rayon pool、普通 CSC 与 sampled 操作的并行、Float64 的部分 dense-block Cholesky、MPFR 自有存储和所有已支持精度的直接 SVD。[R3]

HEAD 已修正 sampled ordinary CSC 串行路径、pooled forward/adjoint 缓冲未清零、condenser 内外层并行叠加和 GEMM/SYRK 的 limb-count fused 限制。不要把这些已经合入的改动再列为新贡献。[R1]

MPFR 使用 inline `MpFloat<N>`：值持有 limbs、kind 和 exponent，不持有长期 MPFR 指针；临时 descriptor 只在原生调用时存在。算术显式使用 round-to-nearest/ties-to-even。不能用浅拷贝原生 MPFR 对象替代当前独立 ownership。[R7]

BFLA、MFLA、CRT、MPI **并不是这个 Rust 项目已经实现的生产后端**。旧 Julia 项目中的能力和外部桥接试验不能算作 SDPX.rs 已有能力。[R3][R9]

### 1.2 当前证据已经否定“盯住 QDLDL 就能解决 64 核”

HEAD 提交记录的 Ising512、512-bit、单 64-core socket 结果如下。它们是该 fixture 的仓库记录，不是本次实测，也不能与不同提交、不同主机上的历史秒数直接比较。[R1]

| 配置宽度 | 原生时间（秒） |
|---:|---:|
| 1 | 60.52 |
| 4 | 20.24 |
| 8 | 13.35 |
| 16 | 12.47 |
| 64 | 14.60–15.50 |

同一记录给出 QDLDL 阶段约占 2.4%，消去树最宽层 11、深度 51。若该占比代表当前分析目标，即使该阶段完全免费，Amdahl 上限也仅为

`1 / (1 - 0.024) = 1.02459…`，即约 2.46% 全求解加速。

这是针对这条 profile 的推导，不否认另一类大问题可能由分解主导。Phase 3 需要分解占比高的另一组固定大模型证据，而不是用小 Ising 的 QDLDL 单线程形态推断所有模型。

同样，不要让“64 个线程都忙”成为目标函数。更快且更少同步的 16-worker 执行可以优于制造工作来占满 64 核。报告 `requested_capacity`、`effective_workers` 与测得的使用情况，不要通过把 64 改报 23 来掩盖真实配置。[R1][R3]

### 1.3 文件级审阅地图

| 现有位置 | 观察 | 建议改动边界 |
|---|---|---|
| `crates/arithmetic/src/lib.rs` | inline MPFR 值、固定精度、独立 storage | 精确 dyadic 导入导出、只读二进制 bridge；不改标量默认舍入 |
| `crates/solver/src/algebra/dense/blas/mpfr.rs` | ordered FMA GEMM/SYRK，按完整输出列并行 | provider 层接入精确 CRT；小矩阵/不适合数据保留原实现 |
| `crates/solver/src/solver/implementations/default/kktsystem.rs` | update 后常量 RHS，再 affine/combined；已经共用因子 | lazy constant + affine 多 RHS 批次，迭代内缓存 |
| `crates/solver/src/solver/core/solver.rs` | affine→步长/centering→combined 数据依赖 | 只调整合法计算排程，不删除 predictor/corrector |
| `crates/solver/src/solver/core/kktsolvers/mod.rs` | `setrhs`/`solve` 只支持单 RHS | 小范围增加 typed 多 RHS 接口，默认循环 fallback |
| `crates/solver/src/solver/core/kktsolvers/condensed.rs` | PSD/orthant condensation，保留原始 augmented refinement | 复用结构与精确 scatter；不能重复构造整个稠密 KKT |
| `…/direct/quasidef/ldlsolvers/dense_block.rs` | Float64 单正块、小负边框，Cholesky+TRSM+SYRK+fallback | 泛型化、连通分块、tile DAG，而非另起一套 KKT 框架 |
| `benchmark/profile/` 与 `benchmark/research/` | 已有 instrumentation、冻结与成对比较 | 扩充现有工具，不复制第二个不兼容评估器 |

以上观察基于 [R4]–[R8][R11][R12][R13]；代码路径的目录树不是未来新增模块已存在的声明。

## 2. SDPB：借鉴什么，不照搬什么

SDPB 3.1.0 的相关链路是：对局部 Schur 块 Cholesky，计算 `Y_j=L_j⁻¹B_j`，然后用 CRT/BLAS 构造全局 `Q=ΣY_jᵀY_j`，最后进行分布式求解。这里的 `Q` 是全局边框矩阵，不是 SDPX 目标函数中的 `P`；变量名不能机械对应。[S2][S3]

| SDPB 参考位置 | 借鉴内容 | SDPX 的对应落点 |
|---|---|---|
| `…/step/initialize_schur_complement_solver/compute_Q.cxx` | 局部分解、TRSM、全局 Gram 的分层 | component/border factorization，不改 HSD 外层 |
| `…/bigint_syrk/Readme.md` | 模整数 BLAS、prime×tile 并行、有界工作区 | MPFR dense provider 的 exact-CRT backend |
| `…/bigint_syrk/fmpz/Fmpz_Comb.hxx` | 位宽、符号与乘法结果边界，CRT 工作区 | exact input packer、modulus planner、worker-private scratch |
| `…/bigint_syrk/blas_jobs/` | 三角/矩形任务，按成本分派 | tile work estimate 与统一线程预算 |
| `…/BigInt_Shared_Memory_Syrk_Context/` | node-shared residues、first touch、restore/reduce | 节点内 scratch ownership、MPI 分布式归约 |
| SDPB 并行论文 | 局部块/全局矩阵两级分布、负载实测 | MPI 内存与进程组设计，完整求解而非单分解测速 |

本次读取了前述 README、compute_Q 与 Fmpz_Comb 等关键文件；子目录内其他函数是据 README 定位的 agent 后续阅读项，不表示已逐一审查。

**关键区别：** SDPB `compute_Q.cxx` 的规范化对角检查明确使用约 `2^(-N/2)` 的保守检查，并说明无法精确控制该过程的舍入误差。这个检查不是“输出逐位等于 p-bit ordered MPFR FMA”的证据。SDPX 的方案必须独立规定精确 dyadic 转换和舍入契约，不能照抄该归一化后宣称位精度完全不变。[S3]

采用已包含相关 output-split 修复的 3.1.0 参考，不以 3.0.0 的旧 restore/reduce 行为为蓝本。SDPB release 的整体加速属于 SDPB 特定版本/大问题比较，不能当成 SDPX 的预估。[S1]

## 3. Phase 0：正确性、热点与可复现基线

### 3.1 开始条件

在独立候选副本上工作。记录 HEAD、dirty diff、Cargo.lock、Julia manifest、输入与 evaluator hash、实际加载的共享库和原生依赖身份。分离 `baseline/` 与 `candidate/` 的 `CARGO_TARGET_DIR`，防止旧编译 metadata/动态库污染比较。[R9][R10][R11]

先跑相关数值回归。较大 Lambda11 的 sampled dual-consistency 问题在现有记录中未闭合，必须保留为失败并定位后才能赋予“大模型正确性/加速”资格。不得通过删除该题、切换更宽松容差或把已知失败默默排除来过关。历史被用户停止的完整 CSDR campaign 不在本计划默认重启范围。[R3][R9]

### 3.2 Profile 必须回答的具体问题

每个有代表性的实例，在 1/4/16/64 worker 与相同精度下分别归因：

- scaling/SVD、affine RHS/recovery、combined RHS/recovery、步长/特征分析、普通 CSC、sampled forward/adjoint、Schur assembly、numeric factorization、triangular solve、refinement、等待和分配。
- 记录 `factorization_count`、`rhs_application_count`、`solve_batch_count`、每 RHS refinement 次数、后端实际命中次数、fallback reason。
- 每个 GEMM/SYRK/TRSM 记录实际尺寸、LD、精度、transpose、alpha/beta 类别、调用点、输出可复用性、指数跨度及累计时间；捕获早/中/晚迭代矩阵。
- 记录实际可运行 cpuset、socket/NUMA 分布、requested pool capacity、任务数、关键路径/尾部等待。CPU 利用率不是 wall-time 的替代指标。

应优先扩展已有 `benchmark/profile/`；检查现有 span 的包含关系。exclusive timers 可以相加；SVD、GEMM 等 nested span 只能做归因，不得与外层再次相加。[R9][R12]

### 3.3 阶段完成标准

产生 `baseline_manifest.json`、`profile_summary.json`、完整数值结果和 `hot_shapes.jsonl`。报告每类实例的实际 dominant cost，并说明哪个 Phase 有多大的理论可改善占比。没有合格大输入不阻止进行单元/开发试验，但不得标记大模型或多节点已完成。

## 4. Phase 1：精确 CRT + 硬件 BLAS 算子后端

### 4.1 先定义两种不同的数值语义

当前实现对每个输出执行

`v₀=0; vₜ₊₁=RN_p(aₜ bₜ + vₜ)`，再按当前 `axpby` 顺序处理 alpha/beta。[R6]

拟新增后端首个版本定义为

`v=RN_p(Σ aₜ bₜ)`（内部和为精确 dyadic 值），然后执行**与当前相同的** alpha/beta 操作与舍入顺序。

二者通常不同。前者是 `OrderedMpfrFma`，后者建议命名 `ExactDyadicDotThenRound`。这不是把 512-bit 数字转换成普通 Float64 近似值求解：Float64 只承载可证明精确的小整数模运算。但它仍然是一次数值算法/舍入顺序变化，必须在设计记录中显式批准并通过旧外部门槛，而不是以“没降 precision_bits”替代正确性说明。

首先作为**显式选择的可选后端**接入。保留现有 ordered-FMA 路径作为参考与 fallback。不得篡改原 bitwise 测试：保留其原语义，另增 exact-once-rounded oracle 和完整 solver qualification。bitwise 一致只能在确实适用的测试中要求，不能要求整个 IPM 轨迹相同。

### 4.2 不丢位的 dyadic 整数化

有限输入都精确写成 `a_ik=m_ik·2^e_ik`、`b_kj=n_kj·2^f_kj`。指数与整数尾数通过二进制接口读取，不经 decimal 文本往返。

对一行 A 与一列 B，可选

`e_i=min_k e_ik`, `f_j=min_k f_kj`,

`Z_ik=m_ik << (e_ik-e_i)`, `W_kj=n_kj << (f_kj-f_j)`，

于是 `C_ij=(Σ_k Z_ik W_kj)·2^(e_i+f_j)`，等式是精确的。

零行/列单独处理。指数跨度大时整数位宽可能远大于 p；这时**分块、分指数层或明确退回 MPFR**，不能截断小元素。`max_integer_bits`、`max_scratch_bytes`、尺寸乘积和移位量必须预检查。

若按 k-panel 使用不同的 exponent offset，各 panel 的结果必须先在精确整数/dyadic 域对齐相加，最后才舍入。**每 panel 重构后先舍入到 p 位再求和，不符合全 dot 一次舍入契约。** 初版采用同一输出 tile 的共同 exponent plan，先正确再优化指数分层。

### 4.3 模 BLAS 的两个独立安全条件

设一个 BLAS panel 的收缩长度为 `k_c`，使用非负 residue `0≤r<p_ell`，要求

`k_c·(p_ell-1)^2 < 2^53`。

那么每个乘积与任意合法累加子和都是 binary64 可精确表示的整数。单 panel 使用 `alpha=1,beta=0`；不同 panel 之间在**整数模域**做有界加法/归约。不能给每个 panel 都开 `beta=1`，却仍用单 panel 边界保证整个结果不溢出精确整数范围。采用中心化 residues 时，也必须用绝对值和给出对应边界。[S2][S4]

另设整型精确结果满足 `|C_ij^Z|≤B_ij`，全局上界 `B=max B_ij`。signed CRT 唯一重构要求

`M=∏_ell p_ell > 2B`。

一种保守边界是 `B_ij=Σ_k |Z_ik||W_kj|`；也可以用 k 与操作数最大位宽生成严格较松上界。二者不可混为一谈：前一个条件防 BLAS 算错 residues，后一个条件防 CRT 得到错误的大整数。`num_primes` 从当前严格上界计算，不硬编码“512-bit 用若干 primes”。

对于支持的 BLAS provider，验证实际执行 FP64 运算且没有低精度近似路径。小整数 exact oracle 必须在每种 provider/build/CPU dispatch 上测试。没有证据时不得把所有硬件矩阵引擎都视为等价实现。

### 4.4 建议的窄接口与状态

新增模块名称是建议，可按现有结构合并，禁止为了模块数建立过度抽象：

```text
crates/arithmetic/src/dyadic.rs        # 精确只读值视图/二进制导入导出
crates/solver/src/algebra/dense/blas/crt/
  mod.rs                             # restricted exact kernel + checked dispatch
  plan.rs                            # bit/exponent/modulus/shape/memory plan
  pack.rs                            # 数据转 residue；无文本中转
  reconstruct.rs                     # signed CRT + 最后一次 RNDN
  workspace.rs                       # 有界且 worker-private scratch
  tests.rs                           # GMP/Fraction/MPFR oracle、padding、边界
native/crt/                          # 若复用 FLINT，用小型可选 C ABI bridge
```

`CrtPlan` 的缓存 key 至少包含 scalar precision、shape/layout、transpose、严格位宽/指数域边界、prime 集、provider identity 与 memory policy。形状相同但输入指数域变化时，旧的 prime/位宽计划必须失效。

`CrtWorkspace` 不能缓存可变输入值而漏掉 epoch；immutable matrices 的 residue 缓存只有在其内容 epoch 受控、总内存受控时才允许。FFI 不能返回指向临时 MPFR descriptor 的借用，也不能把 native allocation 的生命周期藏在 Copy 标量中。[R7]

原生库许可、原始版权及修改出处单独登记。SDPB 顶层 MIT 不自动代表它借用的 FLINT 片段或所有 native dependencies 都是 MIT。[S5]

### 4.5 任务分解、64 核与内存

采用 `prime × output tile` 任务，而非“每 prime 一个线程”。SYRK 只排一侧三角；对角 tile 使用 SYRK，非对角使用 GEMM。prime 数不足时有输出 tile 并行；tile 太小时可合并。调度成本首先用运算量与实测 kernel 延迟校准，避免纯按列数均分。[S2]

推荐默认：**一个现有/共享 Rayon pool 管外层 tile 与 prime；BLAS 每调用一个线程。** 若大矩阵实测支持“一个外层任务+多线程 BLAS”，作为互斥 policy，不能同时让 cone pool、tile pool、BLAS 各开 64 线程。不要在热循环创建新 pool 或反复修改 provider 全局线程配置。

内存设计以 output tile 为单位 stream primes/k-panels；界限包括 packed input、residue output、CRT exact accumulators、native comb 工作区、double buffering 与同时 in-flight task。SDPB 已提示 residues 可能显著扩大内存，因此预算从一开始就是 planner 约束，不是 OOM 后再加。[S2]

### 4.6 哪些算子先接入

从 Phase 0 实际命中的 GEMM/SYRK 开始，包括 PSD congruence、sampled transforms、Gram、未来 Cholesky trailing update。不要只测试 condensed 普通系数路径却宣称 sampled Ising 变快，也不要认为“Schur assembly”名称就意味着它是主热点。[R6][R9]

小 SVD/QL、三角 panel、稀疏 SpMV、短 dot 未必适合 CRT。SVD 可以借助成熟全精度实现另做 provider 对比，但不在这一步替换数学路径。POTRF panel 保留固定精度稳定实现，CRT 主要为大块乘法铺路。

### 4.7 测试与晋级

矩阵测试覆盖 f64 回归及 128/256/512/768/1024/2048 位；m/n/k 为零和非 tile 倍数；U/L、N/T/C、非紧密 leading dimension、未用三角/边界 padding 保留；alpha/beta 为 0/±1/一般值；正负、严重消去、极端 exponent span、零、非有限输入 fallback；重复调用、输入/缓存 epoch 改变、并发 scratch 不别名。

数学 oracle 用 GMP 整数或足够严格的 exact rational/dyadic；不能用另一次相同 CRT 运算相互证明。正确舍入需用 exact integer 位运算或夹逼区间验证，不能统一加固定“多 64 位”就认为足够。

计时包含 pack/mod/BLAS/reconstruction/unpack/分配与计划成本，并分别报告 cold/warm。只有微核加速不予全求解晋级。auto dispatch 的阈值使用结构/precision/实际值边界，不读取问题名称。相同准确性下，无 repeatable >=2% 完整收益的路径保持 optional，或以明确内存改善理由保留。[R2][R11]

## 5. Phase 2：KKT 数据流与多 RHS，而不是删除预测校正

### 5.1 当前到底做了几次

在一般成功、没有额外 retry 的迭代中：

1. cone scaling 更新 K_t，KKT update 完成数值分解，并求常量 RHS `v₀=K_t⁻¹[-q;b]`。
2. affine RHS 已知，求 `v_a=K_t⁻¹r_a` 并恢复 affine 全方向。
3. 根据 affine 方向、边界步长和 centering 组装 combined RHS，再求 `v_c=K_t⁻¹r_c`。

初始化、数值重分解、refinement 等另计。现行 `kktsystem.rs` 与主迭代循环确认了上述依赖。[R4][R12]

因此“3 次 solve”≠“3 次 factorization”。“一个常量 RHS”也不意味着其解跨迭代不变，因为 K_t 随 scaling 变。

### 5.2 第一目标：两个提交波次，三个数学 RHS

将 update 中立即做的常量 solve 延迟到 affine 需要时；在同一因子上求

`[v₀ v_a] = K_t⁻¹[[-q;b] r_a]`，

恢复 affine 后继续组装 combined，并求 `v_c=K_t⁻¹r_c`。

得到的结构是：

```text
scale → factor once → solve_many([constant, affine])
                    → affine recovery → step/centering/corrector assembly
                    → solve_many([combined]) → step acceptance
```

总 RHS 仍为 3，只是两个提交波次。多 RHS 对 dense TRSM、共享 transformations、批量 residual/recovery、调度开销可能有利；对 serial QDLDL 两列 RHS，收益可能很小。不能把批处理 API 次数变化当作运算量减少。

### 5.3 接口改造和数值失败传播

在现有 `KKTSolver`/direct factor 接口上增加小型 `solve_many` 或 prepared-RHS block，而不是重写整个 solver trait hierarchy。默认实现按列调用已有 solve，维持所有后端兼容。dense/component backend 才提供真正多列 TRSM。[R13]

每列单独记录 refinement residual、迭代与失败。refinement 可批量做 operator products，但不能让一列已收敛覆盖另一列失败。保留原 RHS，明确 in-place/out-of-place alias contract；失败不得让未初始化的另一列成为“成功解”。

若 lazy constant 修改改变了 update 的 bool 语义，必须相应调整调用方的错误时序：factor_failed 在 factor 阶段报；constant/affine_failed 在批次阶段报。不同失败路径仍按既有 accepted-iterate recovery 退出，不能留下 stale solution。

### 5.4 可优先检查的迭代内复用

在同一 accepted iterate、scaling 与矩阵 epoch 下，`ξ=x/τ`、`Pξ`、constant solve 结果以及 Δτ denominator 可考虑复用。第一版保持原算术表达式与操作顺序，只缓存相同表达式的结果；先不要把等价二次型改写成另一种浮点表达式。

当前 `workx` 在 recovery 中先承载 `ξ`、又作为 `ξ−x₂` 的别名被修改。新增 cache 必须独立拥有存储或在不可变阶段使用，不能把这个可变 scratch 当作长期 `ξ` 缓存。字段注释也必须依据实际 constant solve 写入 `x₂/z₂` 的数据流核对。[R4]

KKT key 建议包括：

`structure_epoch, matrix_values_epoch, cone_scaling_epoch, regularization_epoch, factor_epoch, precision, rhs_data_epoch, iterate_epoch`。

scalar ξ 等 iterate 缓存不必把所有字段都放一个巨型 key，但每个缓存要有明确依赖表。`q/b` prepared update、A/P value update、重分解、正则参数变化、scaling strategy retry、接受新步必须使相应缓存失效。

现有 prepared affine cone 逻辑已经做了部分复用，先读现行实现，避免创建第二份 cone state 或重复 cache。不要为“减少矩阵乘”直接用经过 regularized solve 的关系代替原算子回放，除非先证明误差传播并通过现有 refinement。[R12]

### 5.5 为什么不承诺真正三变一

corrector RHS 包含 affine 方向影响和 centering 参数；在这类 predictor/corrector 算法不变的前提下，不能在 affine 未知时把 combined RHS 作为已知列一次送入因子。

“把 τ/κ border 直接纳入较大矩阵”“用不同预测策略避免一个 RHS”等可作为另一个算法 ADR，但需要推导包括 QP、混合锥、不可行证书、regularization 和 recovery 的等价关系，并比较新增分解/填充成本。即使此法消除常量 RHS，也不自动消除 affine→corrector 的依赖。本路线不将未经证明的变换放进默认改造。

### 5.6 验收

计数器证明 factorization 数没有意外增加，常规迭代 logical RHS=3、batch=2；初始点/retry/refinement 单独解释。独立各列原始 KKT backward error 满足原门槛。prepared q/b 循环、多次 same-RHS、失败后恢复、non-symmetric cones、boundary/infeasible cases 均覆盖。测量 whole-solve、每阶段、内存和迭代数，不能预设 50%–60%。

## 6. Phase 3：泛型 component/border 求解与 tile DAG

### 6.1 已有实现就是起点

`DenseBlockSolver` 当前只接 `f64`，结构上要求前 n 个正号、后 m 个负号。admission 包含 n≥128、m≤256、m≤n/4、正块结构密度≥0.8 与存储上限；内部使用 POTRF/TRSM/SYRK，并保留 QDLDL/Faer fallback。[R5]

这些限制说明“Cholesky 尚未存在”是不准确的；真正缺口是高精度、多个正连通分量、可伸缩 tile 执行和更全面的 planner。

### 6.2 数学对象：只对可验证正定子块做 Cholesky

对经过正确结构置换的系统

`K = [H  B; Bᵀ  -C]`, `H = diag(H₁,…,H_J)`，

若 H_j 正定，分解 `H_j=L_jL_jᵀ`，定义

`Y_j=L_j⁻¹B_j`, `S=C+Σ_j Y_jᵀY_j`。

若 S 也正定，则对于 RHS `[r_x;r_z]`：

`u_j=L_j⁻¹r_x,j`,

`Δz=S⁻¹(Σ_j Y_jᵀu_j-r_z)`,

`Δx_j=L_j⁻ᵀ(u_j-Y_jΔz)`。

**完整 K 是不定矩阵；不能对 K 直接 POTRF。** H/S 正定条件不满足，或对应 mixed-cone structure 不受支持时，保留原 LDL 路径。C 半正定也不足以在 B 秩亏时单独保证 S 正定；不能跳过 pivot/rank/regularization 分析。

本式只定义 factor/solve，不改变整个 HSD 外层，也不建立第二套终止标准。包内 reference oracle 用精确有理数验证 Schur 的符号与 recovery。

### 6.3 结构识别与 planner

用现有 retained-system sparsity graph 识别正块连通分量，生成 permutation、component ranges、B scatter 和 upper/lower storage 映射。先采用保守结构图：存储的数值零不能随迭代变动而偷偷改变 symbolic structure。存在跨分量边则合并；不能为了并行剪掉小耦合。

planner 比较 sparse symbolic fill/cost 与 component dense storage、全局 border 成本和最大内存，而非只用原来的单一 0.8 密度。**不得先分配全稠密 K，再判断它太大。** byte estimate 需使用 scalar 实际大小/额外 native storage，而不是把 MPFR 当作 8-byte double。

多个小块可能已有 QDLDL 最优：新 component elimination 不是自动更快。旧 SOC3/PSD2 小块试验曾出现准确性/速度回退，须将那些诊断加入回归，而非重复宣布结构更小必然更快。[R9]

### 6.4 分三步实现，不一次重写

**P3a 泛型化：** 将 existing dense block factor 的矩阵操作通过现有 precision-aware provider 接口调用。先保持同样的单块结构与同步排程，验证 Float64 不退化、MPFR 正确、fallback values 与 shifts 一致。数据布局转换必须计入性能。

**P3b components/border：** 增加结构分量，将互不相交 H_j 独立分解/解 B_j，再归约 S。固定顺序累加 S 或使用明确的新 exact accumulation policy；不要因线程完成顺序引入不受控浮点求和。保留 sparse baseline 对照。

**P3c tile DAG：** 当一两个大块限制 block-level parallelism 时，再把 H_j 或 S 的大块分解转为 POTRF/TRSM/SYRK/GEMM DAG。Phase 1 在大 trailing updates 上作为可选 provider 使用；DAG 正确性不依赖 CRT 已经更快。

### 6.5 DAG 依赖与内存契约

对 lower Cholesky，每轮 k 的任务为：

```text
POTRF(k,k)
   └── TRSM(i,k), i>k
          ├── SYRK(i,i;k)
          └── GEMM(i,j;k), i>j>k
```

每个 tile 由唯一 writer 更新；同一 `(i,j)` tile 的 k 更新链有版本依赖。下一 panel 只等待它确实依赖的前驱更新，不等待整层所有远端 trailing tasks。优先推进关键 panel，支持受控 lookahead。

不能在每个 k 外围调用一次全 pool barrier 再称之为 task DAG；也不能把全部 O(t³) 逻辑任务一次展开为巨大的 heap graph。采用 lazy successors/windowed active tasks，限制 in-flight update buffers 与 metadata 数量。线程工作区在 worker 初始化/阶段进入时复用。

避免多套 pool：复用 solve-level execution context。当前“仅 dominant block 拿到 inner pool”的机制是已有基线，不是待修 bug；后续 DAG 可以取代其排程，但必须用真实更好结果证明，而不是仅换 abstraction。[R1][R6]

### 6.6 regularization、fallback 与诊断

从 current matrix snapshot 和 caller 原有 shifts 建立因子。失败 fallback 必须接收到所有当前 values，包括 dormant 后端期间的更新与临时 static shift；这是现有 dense-block 测试已经覆盖的语义。[R5]

不得在 dense backend 内无记录地加大正则、钳制 pivot、删 equality。若 default native 策略要求重试，更新 factor/regularization epoch。求解后继续使用原始 augmented/operator residual 做现有 refinement，而不是只验证已经 regularized 的小 S。

报告 `selected_backend`、SPD admission/rejection reason、component sizes、border order、symbolic estimated/actual factor bytes、factor/residual errors、fallback counts、active worker/BLAS widths。当前 `threads=1` 等信息不能被误用为实际原生库线程数证明。

### 6.7 scaling 验收

使用多小块、单大块、极不均匀块、稀疏不宜稠密化、dense+small-border、large-border、混合锥、rank-deficient 正则等结构族。512/1024/2048 位分别测 crossover。

强扩展固定输入、bits、tolerances、MPI rank=1，测 1/2/4/8/16/32/64 physical cores；报告 `S_p=T₁/T_p`、`E_p=S_p/p`、关键路径等待和 memory。相同核数下 BLAS1/外层池与多线程 BLAS policy 单独比较。

不设置“必须 64 倍才算成功”。只有足够大的固定模型实测具备近线性区间，才报告其区间；小模型可明确采用更低 effective_workers。若新 backend 全求解没有收益，保持 optional，不淘汰 QDLDL。

## 7. Phase 4：MPI + 2D 分布式，保持同一求解引擎

### 7.1 进入门槛

进入条件是：至少一个合格目标问题确实存在单节点内存/时间限制；局部 provider 和 component/border 的数值语义已有回归。不能用尚未通过 Lambda11 dual-consistency 的结果证明 scalability。[R3]

为了保持同一 MpFloat/精度语义，推荐主线为 **Rust 现有 IPM+tile 数学，MPI 只提供通信/ownership 的可选 native bridge**。Elemental/SDPB 用作分布式布局和算法参考，先做小型互操作评估；不要把 GMP `BigFloat` 的行为直接当成当前 MPFR scalar 的逐位替代。[R7][S2][S6]

若经阶段评估决定使用 Elemental 内部 factorization，必须先给出它的 scalar/rounding/ownership 适配证明与完整回归。该选择记录在 ADR，不能在同一 PR 中同时引入两个生产分布式系统。默认不为普通单机用户增加必需 MPI 依赖。

### 7.2 分布式数据归属

建议定义逻辑全局 grid `P_r×P_c`、块内 rank groups 与节点共享 communicator。对象至少包含：

- `DistributedMatrixPlan`：global shape、tile layout、owner、local strides、scalar bits、structural epoch。
- `DistributedFactor`：分解版本、local tiles、通信计划、regularization 与失败状态。
- `DistributedRhs`：列数、owner 分布、root/replicated boundary policy。
- `DistributedInput/Result`：stream/shard I/O、原坐标映射和可选 root gather。

使用 2D block-cyclic tile ownership 是本方案的实现选项，不等于声明 Elemental 所有 `MC,MR` 数据结构都采用同一种 block size。

局部 PSD/sampled blocks 尽量与其 rank group 保持 locality；全局 S/H 大块用 2D distribution。大的 residues 在 node-local shared 或线程共享工作区中存放，不能每 rank 完整复制 Q/S。

### 7.3 “打破内存限制”要覆盖完整链路

不要先由 rank0/Julia materialize 全部巨大 A、PSD Hessian、Schur，再 scatter。新增 shard/stream 输入和执行入口，保留普通 bulk ABI 单机路径。分布式输入仍要保留 authoritative sampled factors/rounded CSC 语义；不能转换成近似因素来省内存。

全局 assembly 直接写 owner tiles，通过有界 reduction 得到最终 S；factor、RHS solve、原始 residual、termination 必须能在分布式布局执行。结果仅在明确允许且满足预算时 gather 全向量/矩阵；否则写 distributed result shards，并提供显式的小规模读取接口。

把 local H、B/Y、global S tiles、CRT buffers、MPI windows、reduction scratch、checkpoint 和 root frontend 常驻量全部纳入 max-node memory。不能只报告 `n²/ranks` 的因子内存。

### 7.4 MPI 线程与 collective 错误协议

第一版选择通信由初始化 MPI 的主线程执行，明确使用/检查满足 `MPI_THREAD_FUNNELED` 的支持级别。Rayon workers 不直接调用 MPI。通信与计算通过 ready queues 协调；后续需要 MULTIPLE 时单独性能/正确性资格审查。

所有 rank 共享同一 iteration/factor epoch、precision、prime plan 和 collective sequence。一个 rank factor/CRT 失败时先传播状态，再统一 fallback 或退出；禁止只有失败 rank 走 fallback、其他 rank 继续等待 collective。

MPI counts、displacements、序列化字节数检查整数溢出；大消息 chunk 化。测试空 local tiles、不均匀 grid、非整除 split、某 rank 无本地块以及多种 P/Q split factor。SDPB 3.1.0 相关修复说明这些边界不是附带的小问题。[S1]

### 7.5 高精度 wire format

不直接 `memcpy(MpFloat)` 作为跨进程/跨版本协议：它虽无长期指针，但 Rust 默认布局、padding、endianness、exponent type 不是稳定 wire ABI。[R7]

建议 canonical binary format 包含 version、precision_bits、kind/sign、signed exponent、limb count、规定字节序的 limbs 与 checksum。浮点特殊值明确编码；拒绝精度不一致、未知版本或无效长度。

若所有 rank 同一机器镜像，可以额外提供经验证的 fast pack，但 canonical format 作为保存与不兼容检查的基础。不能以十进制文本批量传输作为生产数据通道。

### 7.6 分布式 factor/solve 的实现顺序

先 1 rank 走 MPI backend，与单机 provider 对照；再 2/4 ranks 的分布式 GEMM/SYRK/TRSM；再分布式 POTRF 与 border solve；最后嵌入整个同一 IPM 循环。

panel broadcast、trailing update、local contribution reduction 的 dependencies 按 tile 版本维护。设置有界 double-buffering 和通信进度驱动。直接克隆节点内所有 residues 到每个 rank 或每个 node 都持有全部 S 不能通过内存验收。

先采用固定 reduction tree 形成可重复的同配置输出；跨 rank 数可以产生不同舍入，不声称 serial bitwise identity。无论 reduction tree 怎样，工作精度与原始残差/锥/目标门槛不变。若采用 exact CRT aggregate，可对那一算子单独提供更强的确定性说明。

### 7.7 I/O 与恢复

checkpoint 保存 accepted iterate、原坐标映射、模型/input hash、精度/settings、layout/schema 与依赖身份。checkpoint resume 是同一计算的恢复，不顺带引入不同精度 ladder warm start。

写入使用临时 manifest+完成标记的提交方式；中断 checkpoint 不可被当作有效 checkpoint。rank0 恢复不得要求将完整大矩阵加载到本地后再分发。

### 7.8 qualification

按有资源授权的规模做 1/2/4/8 节点，再到 128/256/512/1024 核，不是预先承诺一定运行到最大值。每个点保留原始数据、实际物理核/SMT、rank/thread/grid、真实 backend 命中与收敛失败。

强扩展固定模型，弱扩展固定每 rank/节点工作与内存。报告 input+setup+solve+output、单次迭代与分解、MPI wait/bytes、最大节点内存和通信占比。shared pages 多 rank RSS 重复计数时使用 node/PSS 或明确的共享内存账本，不能把 Σrank RSS 当成实际物理消耗。

真正“超出单节点内存”的验收必须选择单机输入链路无法在既定预算下容纳、分布式全流程成功的模型，且没有某个 rank/root 偷偷超过该单节点预算。

## 8. 推荐模块结构：一个核心，不是四套 solver

```text
existing Julia/MOI + bulk C ABI
              |
         same Rust IPM
              |
       same KKTSystem
              |
  augmented / condensed formulation
              |
  Factor backend (existing interface extended narrowly)
     ├── sparse LDL (retain)
     ├── component + border factor (generic T)
     └── distributed tile factor (optional)
              |
  Dense operations / arithmetic contracts
     ├── Float64 native BLAS
     ├── ordered MPFR kernels (retain)
     └── exact dyadic CRT kernels (optional → qualified auto)
              |
   shared execution budget + owned scratch
```

新增 `ExecutionContext` 是否需要新模块，先查现有 pool ownership；优先把重复预算与调度 policy 合并，而不是把全工程重命名。只把 precision/provider/tiles/communication 等真正跨模块契约做成小型类型。

新增 feature 名如 `crt-blas`、`mpi` 在本方案中均是建议，不能在它们实现以前把 `cargo --features …` 当作现有可执行命令。需要扩充 C ABI 时显式版本协商，不偷偷改变现有 ABI 3 布局或让旧 Julia frontend 误加载。[R3]

## 9. Agent 工作包与依赖

详细字段见 `task_graph.json`。每项都需要“现状核查→最小修改→focused tests→固定比较→决定→回退记录”，且不得同时把新算法、新 provider、新调度混成一个无法归因的提交。

| ID | 工作包 | 必需前驱 | 主要验收物 |
|---|---|---|---|
| A00 | 固定 source/input/runtime；确认现行 AGENTS 与差异 | — | baseline manifest、dirty/source/dependency identities |
| A01 | 正确性与缺口清单，较大模型 gate | A00 | 支持精度回归、已知失败及最小 reproducer |
| A02 | 扩充真实热点、计数与形状采集 | A00 | exclusive profile、hot_shapes、实际命中 |
| A03 | pool/预算 ownership 审计与最小调度接口 | A01,A02 | 1/4/16/64 预算无 oversubscription、配置与实际区分 |
| A10 | dyadic bridge 与 rounding ADR | A01,A02 | exact roundtrip、指数/位宽/舍入测试 |
| A11 | 整数模乘与 signed CRT 参考实现 | A10 | 全范围 exact integer oracle、一致的重构上界 |
| A12 | BLAS、prime×tile、有界 stream | A03,A11 | provider exactness、padding/splits、总成本与内存 |
| A13 | dense provider 接入与成本派发 | A12 | 实际热点调用、原门槛全求解、fallback receipts |
| A20 | RHS/dataflow/epoch proof 与计数 | A01,A02 | 1 factor/3 RHS 解释，允许缓存依赖图 |
| A21 | 单 RHS compatible 的 solve_many | A20 | 每列原始残差/refinement 与故障传播 |
| A22 | 常量+affine 批次、迭代内 cache | A21 | 常规 2 waves、3 RHS，q/b/update/retry 不 stale |
| A30 | 泛型化已有 dense block，保持其结构 | A01,A20 | 全精度 factor/solve、现有 fallback 语义 |
| A31 | 正连通分块+border planner | A02,A30 | 原方程 proof、symbolic/memory gate、稀疏回退 |
| A32 | tile DAG + 多 RHS factor 路径 | A03,A21,A31 | 依赖/唯一 writer/关键路径/有界 task storage |
| A33 | 本机组合验证 | A13,A22,A32 | 消融+完整回归+可支持的大模型扩展区间 |
| A40 | MPI ownership、wire、threading、错误协议 | A03,A30 | 1/2/4 rank 小矩阵与错误注入 |
| A41 | distributed input/assembly/factor/solve | A32,A40 | 全链路不 root/每 rank 复制、全精度原始残差 |
| A42 | checkpoint/output/scaling/memory 验收 | A01,A41 | 合格大实例、分布式高水位、strong/weak scaling |
| A90 | 集成、文档、回滚与保留集资格 | A33,A42 | 已实现/optional/拒绝/未测边界及 reproducible artifacts |

A40 的有限小规模协议开发可以提前，不代表 A42 可绕过大模型正确性门槛。A33 依赖 A13 表示“已完成后端评估决策”，允许 A13 的结论是 optional/no-promotion；不要求失败的优化必须开启。A90 可形成单机 release milestone，MPI 未完成明确标为 deferred，不能伪造所有 tasks done。

### 9.1 每个工作包统一交付模板

```text
Task ID / baseline SHA / candidate SHA or diff hash
Hypothesis and measured target hot path
Files changed and why current implementation was insufficient
Arithmetic / factor / ownership / failure-semantics changes
Focused tests and all failures
Benchmark identity + raw samples + input/oracle hash
Native time / API time / cold-warm / iterations / RSS scope
AB and BA comparisons + regression coverage
Decision: keep_candidate | reject | correctness_only | incomplete
Default enabled? Why? Exact rollback boundary
Remaining uncertainty / next dependency
```

不要把“代码能编译”“入口已接通”“test-only 参考实现变快”填成 performance qualified。

## 10. 测试与测量矩阵

### 10.1 正确性层次

| 层次 | 内容 | oracle |
|---|---|---|
| 标量/bridge | mantissa/exponent、kind、RNDN、别名、移动/重复使用 | GMP/精确 dyadic 与独立舍入 |
| kernel | GEMM/SYRK/TRSM/POTRF、布局、极端指数、padding | 精确矩阵乘/因子残差；不以同一路径自证 |
| factor | KKT signs、Schur、regularization、fallback、nrhs | 原始 K 的逐 RHS backward error |
| solver | feasible/infeasible、边界、混合锥、prepared updates | 现有 original-coordinate point/ray gates |
| parallel | widths、重复运行、DAG race、scratch reuse | 固定配置确定性与相同精度准确性 |
| MPI | 空 rank、非整除布局、传播失败、中断恢复 | 分布式原始残差 + 小模型单机等价 |

新数学路径允许结果与旧浮点轨迹不同，但**不允许**放宽问题的外部准确性要求。测试中对非唯一 primal/dual 最优解不能要求某一个指定解；验证应使用原始最优性和锥条件，同时保留错误 oracle 的历史失败记录。[R9]

### 10.2 不得混用的性能口径

现有研究工具默认主指标是 Julia API fresh setup+solve+结果提取/清理，原生 Rust 时间另报；OS RSS 包括 Julia/JIT/输入/验算，不等于 native per-solve allocation。[R11]

沿用已有 AB/BA 两方向比较；默认类内/类间权重、2% 类退化、10% 单例/峰值 RSS 退化和不删失败规则，以冻结 campaign 的现行配置为准，不在看到结果后改。至少接近晋级线的结果需要独立时段复验。它们是工程筛选，不是统计显著性证明。[R11]

独立 provider 实验当然允许 provider 不同，但除了被检验变量之外的 source/input/precision/runtime 应固定；不能把这种 provider 对比偷送到要求同 provider 的源码比较晋级器。必要时新增明确的 campaign 类型，而非取消身份检查。

现有 research pair driver 的线程范围为 1/2/4/8；**不能假定给它直接传 64 或 MPI 就已经正确支持**。先扩展已有 parallel/Ising driver 或实现经审查的大规模 profile driver，并保留原有限范围流程。[R11]

### 10.3 成功与失败的决策

- 精度或 status 回归：立即停止性能晋级，保存最小例与 raw logs；不得因某些算子变快而放过。
- 小 kernel 快、whole solve 不快：保留参考/optional 或拒绝；不要写“全求解 3 倍”。
- 新路径只减内存但不加速：可用明确 memory benefit 合入 optional/default policy，单独说明成本。
- 无法取得依赖/资源/输入：状态为 incomplete/blocked，不是 pass 或零秒。
- 集成后收益消失：用消融定位 P1/P2/P3 交互，保留 baseline backend；不要把各单项 speedup 相乘。

## 11. 可以直接使用的构建/检查命令

在目标环境已准备依赖、源码固定且获得执行授权后，从独立工作区执行。以下基础 features 是本次读取文档确认已有的；新增 CRT/MPI flags 要等实现后再加入。[R3][R10]

Linux 示例：

```sh
export CARGO_TARGET_DIR=/absolute/external-build/sdpx-candidate
cargo build --locked --release -p sdpx-ffi --features sdp-openblas,faer-sparse
cargo test --locked --release --workspace \
  --features sdpx-ffi/sdp-openblas,sdpx-ffi/faer-sparse -- --test-threads=1
julia --startup-file=no --project=julia/SDPX.jl -t1 --gcthreads=1 \
  julia/SDPX.jl/test/runtests.jl
```

macOS 使用 `sdp-accelerate` 替换 `sdp-openblas`。不要复制文档里个人机器绝对路径到其他环境。绑定 `SDPX_LIBRARY` 为本轮冻结 library，并检查 Julia 实际加载当前包而不是 sibling。离线缓存存在时可以使用 `--offline`，缓存缺失不允许偷偷改锁版本。[R10]

现有研究命令使用其 `--help` 确认参数：

```sh
python3 benchmark/research/catalog.py list --suite development
python3 benchmark/research/run.py --help
python3 benchmark/research/run.py pair --help
```

具体 campaign 配置、资源预算和 oracle 在运行前冻结，不能把本计划中的示例路径直接当有效输入。holdout 只用于最终资格，不要在日常优化循环反复读取。[R11]

包内独立工具：

```sh
python3 tools/reference_oracles.py --self-test
python3 tools/preflight.py /absolute/path/to/SDPX.rs \
  --output /absolute/external-evidence/preflight
python3 tools/task_status.py task_graph.json
```

preflight 不构建、不下载、不修改仓库、不提交作业；它只记录可见身份和环境，不等于验证二进制来源或性能合格。

## 12. 最终发布应该说什么

应发布“具体提交、输入、精度、provider、核数、通过/失败数量、全求解速度与内存变化、可用与不可用后端”，而不是“已经击穿 MPFR 上限”。

理想结束形态：普通稀疏 LP/SOCP/混合锥仍可靠使用现有后端；高精度大矩阵乘法按真实成本选择 exact CRT；有结构的大 KKT 使用 component/border 与 tile DAG；超出单机预算的问题使用可选 MPI 全链路。每条路径共享同一 IPM 状态机、原始数值门槛和明确回退，不是四个各自维护的求解器。
