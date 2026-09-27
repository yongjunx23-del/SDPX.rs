# SDPX plan

更新：2026-09-27。本文件只保存**当前状态、已知失败、优先级和里程碑**，就地更新。
工作规则与数值契约见 [AGENTS.md](AGENTS.md)；已完成实验的过程和数据见
[docs/JOURNAL.md](docs/JOURNAL.md)；每次运行的原始记录在 `$SDPX_E2E_HOME/journal.jsonl`。

## 1. 目标

把 SDPX 做成成熟的高性能锥求解器：

- **Float64**：同题接近 MOSEK——medium ≤ 1.3× MOSEK，large ≤ 2×（large 需先重测基线）。
- **MPFR**：Ising 上时间—精度超越 SDPB（同精度、内部 1e-42、外部原问题 1e-30）。
- **规模**：单核、单节点多线程、跨节点 MPI 三档都有可复现证据。
- **代码**：生产代码 ≤ 30k 行，靠去重而不是压缩格式。

## 2. 当前状态

| 项 | 当前 | 目标 | 证据 |
|---|---|---|---|
| medium（Float64，`e2e.py run medium`） | 2.77 s api（fast，1 线程），19 it，`AlmostSolved`，**外部审计失败** | ≤ 1.3× MOSEK（1.89 s） | 2026-09-23 journal |
| large（Float64） | 旧记录 129 s / 26 it，未在当前代码复测 | ≤ 2× MOSEK（25.4 s） | 旧记录 |
| ising11（MPFR 512，`e2e.py run ising11`） | 16.5 s api（fast，1 线程）；release 记录 15.77 s；50 it，`Solved`，审计通过 | 优于 SDPB 同题 | 2026-09-23 journal |
| Λ19（MPFR 768，集群） | 空闲节点、固定 NUMA 域、审计通过（rns60/rns63，2026-09-26，节点间 ±5–10%）：**32 核 111.6–116.7 s**（SDPB 204.2 s）；**64 核 109.9–123.3 s**（SDPB 152.8 s）。119 it（SDPB 243 it） | 已达成 | 2026-09-26 journal |
| 更大题 Λ19 spins 0–50（52 PSD 块，768） | **已修复**：精度相关的均衡上下界 + 精确内层残差后 768 bit 177 it `Solved`，审计通过（此前 345 it `InsufficientProgress`）。1 节点 52 线程 **257.5 s**（rns63；rns60 291.2 s，rns48 370.1 s），1 节点 4 进程×16 线程 434 s（rns46），**2 节点 16×8 230.4 s 求解 / 4:04 墙钟**（rns63；rns48 8×16 309.0 s）；SDPB 1 节点 64 进程 328 s（265 it），32 进程 557 s | 单节点快于 SDPB 1 节点 64 进程（257 vs 328 s） | 2026-09-26 journal |
| MPI | owner 分区路径：大题（52 块）单节点 4×16 比 1×64 快 8%，2 节点 8×16 比单进程 64 线程快 25%（363 s，审计通过，结果与单进程逐位一致）；Λ19（28 块）多进程不划算。SDPB 2 节点在本集群挂起（docker0/网桥同址，TCP 与 ib0 均未解决）。普通路径仍逐 site gather | 单一 owner 路径，多节点快于单节点 | 2026-09-24 journal |
| 代码规模 | 生产约 54.7k 物理行（旧口径“有效代码”约 41.5k） | ≤ 30k | M7 |

热点（详见 journal）：medium 的 Schur 装配约占 60%（dot_scatter 为主）；ising11 的时间集中在
锥内核（SVD/eig/W 乘积）和 RHS/残差中的块同构，而不在 Schur 分解。

## 3. 已知失败（保留为失败，不放宽门槛，不阻断无关改动）

| 失败 | 现象 | 状态 |
|---|---|---|
| medium 外部审计 | 2026-09-27 binary64 精度修复后求解器报告 `Solved` 18 it，dual residual 2.85e-6→1.92e-6，仍 > 1.75e-6（此前 `AlmostSolved` 19 it，gap 1.12e-6） | 用户暂缓；证据指向末端 poor-progress 轨迹，而非 Schur 装配 |
| `SDP_control3` | 29 it，primal residual ≈ 4.98e-4；输入 SHA256 `d3a7cc27…6e84` | 用户 2026-09-21 暂缓；保留在完整题库 |
| `condensed_graded` 测试（MPFR） | `--ignored` 扩展测试失败；f64 版已修复并启用（2026-09-27） | 保留为失败 |

新失败、新状态或残差恶化必须检查，不能归入上表。

## 4. 优先级与里程碑

按“收益 × 风险”排序；每项的验收都是对应 case 的一次完整 E2E（见 AGENTS.md）。

| # | 里程碑 | 下一步 | 验收 case |
|---|---|---|---|
| 1 | **M4 MPFR 锥内核** | 找 SVD/eig 的矩阵级冗余，再评估 WY/GEMM、结构化 PSD Schur/W 乘积；停止单一除法级别的小替换 | `ab ising11` |
| 2 | **M5 RHS/精化 + MPI 收敛** | 合并真正可共享的 RHS、复用当前迭代缩放乘积；owner 路径补非对称锥回退与 augmented KKT，覆盖 CLI 默认路由与 C ABI，在有 MPI 的机器上完成 E2E 后删除普通路径的逐 site gather | `ab ising11`；MPI E2E |
| 3 | **M2 Float64 Schur 装配** | 保留四路 FMA 与分块；减少大块矩阵级工作，不加别名查找。稠密块的平方根 Hessian/SYRK 仅在真实稠密输入出现时再评估 | `ab medium`（审计已知失败，只比较时间且点不变坏） |
| 4 | **M7 精简** | 按生产调用证据合并 `default` 与 `distributed` 的重复实现；拆分 `sampled`、`condensed`、`compositecone` 大文件 | 两个 case 输出逐位不变 |
| 5 | 线程预算 | BLAS 线程纳入求解器线程预算，避免过量订阅 | 多线程 `run --threads N` |
| 6 | **M6 规模化与比较（超越 SDPB）** | 现状（rns63）：Λ19 32 核 ~112–117 s（SDPB 204.2），64 核 ~110–123 s（SDPB 152.8）；大题 1 节点 52 线程 257.5 s、2 节点 16×8 230.4 s（SDPB 1 节点 64 进程 328 s）。已完成：精度下限修复、内核 GEMM 线程本地打包、SVD 旋转回放、残数核按块拆分与绑核、abs(A)ᵀabs(z) 精确核、按实测收缩率跳过停滞修正、残差跳过与持久停滞下限（Λ19 pass 692→484）。2026-09-26 新增：owner 布局按三次方工作量均衡 + LPT 后交换细化、arrow 叶子在线程富余时按列并行（L⁻¹B、面板解、叶 LDLᵀ）、融合路径线性部分乘积入池、主线程长向量运算入池（串行时间减半）。剩余：平均忙线程仍仅约 30/64（块相位内每块可拆分度有限）；64 核满载时降频；2 节点等待约 21–27%（site 102/320，单叶与双叶 rank 的并行效率不同，需实测成本布局，类似 SDPB block timings）。rns62–67：精确内残差按条目数划分、信息范数改为精确平方和（MPFR）、初始点锥 margin 入池（默认起点 5.3→1.8 s）。已排除：SVD 的 U/V 并行（锥相位吞吐受限）、2 节点均匀结构布局（等待来自每轮抖动，非布局）。rns70–74：owner 路径约化精化融合为一轮 all-gather；步长 λmin 用可证 Float64（纯 Rust 三对角+Sturm，误差界不满足时回退 768 位），eigmin 15 ms→0.14 ms；Λ19 32 核 101.9 s、大题 52 线程 246.9 s（审计通过）。**2 节点必须用 TCP（`--mca btl self,vader,tcp`，ib0）**：OpenMPI 4.1.4 openib BTL 在本集群会丢/损坏消息（挂起、wire 解码失败、RIP=0 段错误），TCP 下 2/2 完成。rns75/77：SVD 旋转回放行分给空闲线程（大题锥相位 11.1→8.2 s/30 it）；owner 集体通信握手合并为一次向量 allreduce（TCP 下 allreduce −85%，2 节点 −7%）。当前（rns77，审计）：Λ19 32 核 105.6 s、64 核 101.6 s；大题 1 节点 52 线程 237.6 s、2 节点 TCP 238.7 s。瓶颈：每块相位在每块 >2 线程后几乎不再加速（残数核按素数拆分对单块关键路径收益低），所以 2 节点并不比 1 节点快。下一步：块内关键路径（单块 congruence/recover 的列拆分、SVD 回放）以提升每块多线程效率；无效节点 node70 需避开。通用锥（rns80–82）：EXP/POW 回溯步长用 binary64 筛选（逐位不变）、非对称锥步长并行 + 验证、障碍函数并行求值；256 位 POW 7.40→3.80 s、EXP 2.25→1.92 s。binary64：用 `faer-sparse` 构建时大填充 KKT 走 faer 超节点多线程 LDL（阈值 ≥1e8 flops），LP 24.2→3.7–4.8 s。剩余：高精度 LP/SOCP 的串行稀疏 LDL | Λ19 与 Λ19 spins 0–50 集群完整求解 + 审计，与 SDPB 同节点、空闲节点对照 |

已完成：M0 fast profile；M1 f64 BLAS 化 `pooled_gemm_sym` 与单次 G·X·G；M3 exactdot 接入
`dot_fma`；2026-09-23 目录重组与死代码删除。

## 5a. 超大题超越 SDPB 的路线（2026-09-25，依据 SDPB 剖析与主元跨度）

两种规模区间，瓶颈不同：
- **A 区：块多、N 中等（加自旋、单关联函数）**。SDPB 每迭代 81% 是逐块工作（方向 36%、步长 Cholesky 24%、双线性配对 21%），Q 仅 9%。
  SDPX 每迭代逐块工作约为 SDPB 的 2.6 倍：每迭代约 5 次求解 pass（HSD 三个右端项 + 修正）对 SDPB 2 次；NT 缩放需逐块 SVD。
  迭代次数 SDPX 少 1.5–2 倍，但尚不足以抵消。要点：(1) 减少 pass——同一分解下的右端项共用块阶段（多列同构、一次屏障），
  常数右端项不做修正，停滞预测跨分解保留并周期性复测；(2) 逐块 SVD 提速（多轮 Givens 旋转聚合成块正交因子后用残数 GEMM 应用，
  分块二对角化）；(3) 单节点多进程 + 多节点按块分配（已测 −8%/−25%）。
- **B 区：N 大（高 Λ、混合关联函数，N ~ Λ²/8 至数千）**。代价由 ΣP_j·N²、N³ 主导，SDPB 用 bigint_syrk + Elemental 分布式 Cholesky。
  SDPX 的 arrow（叶 LDLᵀ、L⁻¹B、BᵀS⁻¹B、边界分解）全是标量 MPFR、边界串行——N 增大后会成为最大瓶颈。要点：(1) arrow 全部上残数 BLAS
  （分块 LDLᵀ、分块 TRSM、精确 SYRK 在所有叶子上以残数累加、一次 CRT，边界分块并行分解，多右端项批量求解）；SDPB 仅 syrk 用残数核，
  因此 SDPX 的分解每迭代可快于 SDPB，再乘以更少的迭代数；(2) 分解远贵于求解时，按实测“分解/求解耗时比”自适应加入多重中心校正
  （Gondzio），以一次分解换更少迭代——SDPB 结构上做不到；(3) 多节点 owner arrow：边界贡献以残数精确归约（确定性），N≲2000 复制分解，更大时分布式。
- **精度**：两者在 Λ19 都需 ≥768 bit（512 bit 均失败；叶主元跨度 Λ19 1e121→6e234，大题 1e134→1.4e250），且随规模增长；
  无法靠降精度取胜，但所有加速都按精度放大。
- **前提**：需要正确的大 N 自举测试题（当前 Λ27 生成有结构问题），验证 B 区。

## 5. 并行架构目标（参照 SDPB）

1. **同节点：线程做块内切分。** 块数少于核数时，把单块同构 GEMM、SVD 反射子更新、Schur
   贡献列 tile 切给同一 rayon 池；`scaling_dispatch` 的 LPT 成本模型扩展到锥计算与 Schur 贡献；
   一个进程一个池，BLAS 线程由同一预算分配。
2. **跨节点：MPI 块 → 进程网格。** owner-partitioned 路径是唯一 MPI 实现：块状态、锥缩放、
   Schur 贡献、RHS 块部分常驻属主 rank；按实测成本（cost history v2）LPT 分配；块组不跨节点。
3. **耦合系统：复制小系统 + 一次有序归约。** 每次 KKT 更新一次 Schur 贡献归约（按 rank 序
   折叠，MPFR 可用精确整数累加），各 rank 复制分解耦合系统；每次求解一次 n 维 RHS 归约。
4. **确定性**：分区数变化不改变结果（MPFR 精确归约；f64 记录与串行差异上界）。

5. **超越 SDPB 的内核路线（依据 SDPB `bigint_syrk` 与 Λ19 分解，2026-09-23）**：
   (a) 残数 BLAS GEMM：定点对齐 → 小素数残数 → 每素数一次 double `gemm`/`syrk` → CRT，
   一次舍入，精度与现有 exactdot 契约一致。先用于 PSD 块同构（每迭代 24 次，1 核约 13 s/it），
   再用于 W 乘积和 Gram。R/Rinv/G/Ginv 在一个迭代内不变，可缓存其残数编码。
   (b) 并行轴改为“块 × 素数 × 列 tile”，不再是 28 个块，填满 32+ 线程。
   (c) SVD/eig 改为分块算法（先约化再用 GEMM 更新），套用 (a)。
   (d) setup 剩余的串行部分（materialize、build_wdiag）在池建立后按块并行。
   状态（2026-09-23）：(a)(b)(d) 已完成。(c) 暂缓：n≈45 时分块尾更新低于内核盈亏点，
   SVD 的主要开销是作用于 V 的 Givens 旋转。
   已否决：降低正则化以减少精化遍数（ising11 实测精化反而增加）。

SDPB 源码 `~/projects/sdpb-build/sources/sdpb`；集群说明见 `ucas-hpc` 的 `references/sdpb.md`。

## 6. 最终验收

- **Float64 / MOSEK**：固定 medium、large，对照 MOSEK 11.1.3（参考适配器仅单线程），相同外部精度；
  记录总时间、状态、objective、原坐标残差和 gap。
- **MPFR / SDPB**：代表性 Ising 规模，相同精度、内部 1e-42、外部 1e-30；核对 SDPB 实际精度，
  记录完整求解时间与并行配置。
- 每个数据点记录输入与产物身份、精度、容差、线程、状态、时间、RSS 和原问题残差；失败如实保留。

## 7. 已关闭方向（无新证据不重开）

跨节点普通 MPI 路径扩展（负扩展，改走 M5）；MPFR Ising 用 Float64 分解（Λ15 768-bit 条件数
5.8e18→3.0e54）；NaN 主元钳制；MPFR Jacobi SVD 热启动；特征值热启动；以 eig 代 SVD（κ²）；
无 SIMD 的残数域 Gram/Schur/LDL；小稀疏结构强上稠密分解（3× 慢）；旋转复用、成对尾部调度、
整块列分组；在未修 chordal 索引前扩大 presolve（已修）；MPFR 精化容差挂钩收敛门（会改变轨迹，
需用户单独授权）；2026-09-24 实测放宽精化容差：Λ19 在 1e-145 下为 AlmostSolved、审计失败，1e-116 下为 InsufficientProgress，保持 eps^(3/4)。跨秩拆分单块（通信代价高于收益）。
2026-09-24（固定 NUMA 域的同节点 A/B）：精化 pass 融合为单一块任务（逐位一致，但端到端慢 3–4%，块同质时去掉屏障不缩短关键路径）；
glibc malloc 参数（缺页 258k→7k，时间不变）；ways 过分解 f=4/8（32 线程 −3.5%/+3%，64 线程 0%，内存 +140 MiB）；
32 线程锥内并行（−1.6%，噪声内）。

Float64 候选（均已撤回，详见 journal）：禁用/增加曲线步、`max_step_fraction` 0.995/0.98、
transform→Schur 面板融合、等 nnz 列结构重排、稀疏左列等价复用、跳过 fused recovery 的 GEMV、
PSD `apply` 改回 R/Rinv 四次 GEMM。MPFR 候选（已撤回）：SVD 2 的幂归一化、eig 两项 `fmma`、
SVD 常量缓存、Householder 分量除法倒数化、整块缩放共用倒数。
