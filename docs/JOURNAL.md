# SDPX experiment journal

Human-readable record of completed experiments and their conclusions, newest
first. Raw per-run rows (status, times, audit, RSS, binary hash) are appended
automatically by `benchmark/e2e/e2e.py` to `$SDPX_E2E_HOME/journal.jsonl`.
Add a dated entry here when an experiment is **kept or reverted**, in the
format: hypothesis → change → E2E result (case, arm, api s, audit) → decision.
Do not rewrite old entries; the plan (`REVIEW_AND_PLAN.md`) holds only current
status and next actions.

## 2026-09-23 — module restructure and dev harness

- Solver source moved to `solver/{core,default,cones,kkt,sampled,distributed,chordal}`
  with per-module `tests/`; dead items removed; `distributed` compiled only
  with `sdp`. medium and ising11 outputs bitwise identical to `pre-restructure`.
- `benchmark/e2e/` added: pinned medium/ising11 inputs, committed Julia audit
  environment, `build`/`run`/`ab` commands. Replaces ad-hoc `/tmp` setups.

## 2026-09-22/23 — performance round (Apple M4, 4P+6E, single process)

Frozen binaries for this round were in `/tmp/sdpx-perf-20260922/` (not kept).
The text below is carried over unchanged from the plan's former §3–§4.


输入：`大规模矩阵/scripts/export_conic_json.py` 把 MOSEK export 转成原生 conic JSON
（medium n=1887、m=13739、nnz=138192，与 MOSEK metadata 一致；large n=7054、
m=42023）。设置 `settings1e6.json`（1e-6，与 MOSEK 记录同门槛），全部
`OPENBLAS/OMP/VECLIB/RAYON_NUM_THREADS=1`、`--threads 1`。

#### Float64 medium（MOSEK 1.89 s / 16 it）

| 臂 | native s（3 次中位） | it | 说明 |
|---|---:|---:|---|
| base（本轮工作树） | 5.44 | 18 | 旧 Julia 前端时代记录 7.98 s 不再可比 |
| + `pooled_gemm_sym` f64 走 BLAS | 4.23（−22%） | 18 | objective 逐位不变 |
| + PSD scaling 单次同构 G·X·G | 3.79（−3.5%，对前一臂同批 3.93） | 18 | objective、残差逐位不变 |

当前 medium ≈ 2.0× MOSEK。剩余相位（symblas 臂，solve 3.68 s）：assemble 2.20 s
（dot_scatter 1.12、transform 0.91、sparse 0.14）、cones_schur 0.52、refactor 0.46、
eigmin 0.15、ir+trsv 0.24。**Schur 装配占 60%**，其中 dot_scatter 实测约
1.4 GFLOP/s；当前实现按分段连续切片、8 项分组累加，transform GEMM 部分已接近 BLAS 峰值。

9 题开发集比较（LP/SOCP/SDP）两臂状态、迭代数全同；`SDP_arch0` objective 差 2e-10
（f64 BLAS 与标量 dot 舍入差），其余逐位一致。这是旧 development 结果，没有逐题附带
原问题审计，不能视作本计划的外部精度验收；large 未在本工作树复测（旧记录
129 s / 26 it，MOSEK 25.4 s）。

2026-09-23 当前 medium 快照（1887 变量、13725 约束，1 thread，1e-6）需与上述历史值
区分：未改动 release 基线与 Schur 四路 FMA 交错候选分别为 2.951 s 与 2.424 s
`api_seconds`（单次 −17.9%），外部墙钟却为 3.03 s 与 3.69 s，计时信号不一致。两臂
均 `AlmostSolved`、20 次迭代，x/s/z 逐位相同；原问题审计也相同但不通过：dual residual
2.85e-6、相对 gap 1.12e-6，超过 1e-6 门槛。此前 18 次 `Solved` 的历史输出也未通过
同一严格审计（dual residual 1.92e-6）。因此这项改动没有引入精度/状态变化，性能收益
暂记候选信号；当前 medium 的外部精度验收仍未通过，不能沿用旧记录称为已验收。

2026-09-23 补测：对同一 medium 分别运行 COSMO 0.8.11 与 Hypatia 0.10.2，求解器容差
均设为 1e-6，并使用相同的原坐标独立审计。COSMO 达到 5,000 次迭代上限（35.63 s；
外部原始 primal residual 2.05e-4、dual residual 8.49e-6、gap 2.71e-5）。Hypatia 在
31 次迭代后报告 `OPTIMAL`（130.85 s）；primal residual 7.18e-8、gap 8.36e-7 通过，
但 dual residual 2.154e-6 略超共同审计的 1.75e-6 门槛。两者都未通过该题的共同审计。
Hypatia 的 combined-stepper 可借鉴其共用 KKT 分解和多 RHS；SDPX 的 affine 与 combined
方向已经复用同一分解。它默认的稠密 QR-Cholesky，以及 COSMO 面向 ADMM 的 adaptive-rho /
Anderson 加速，不适合直接替换 SDPX 的稀疏 NT-IPM。

额外 Float64 诊断保留原曲线步和 0.99 最大步长：禁用曲线步得到更差的 18 次迭代结果
（dual 5.83e-6、gap 2.29e-6）；增加第三个更小曲线比例后结果明显变差（primal 3.05e-4）；
把 `max_step_fraction` 改成 0.995 或 0.98 均未通过（分别 dual 4.99e-6、gap 1.96e-6，
以及 primal 2.72e-3）。四个候选均已撤回。
基线日志显示第 19 次出现进度退化并恢复到前一可接受迭代，因此通用步长调参暂不作为下一项
优化。Schur 四路内核也尚未取得资格：有一次内部计时变快而墙钟变慢，暂不宣称有性能收益。

单 lane 的 transform→Schur dot 面板融合已按 E2E 结果撤回：候选保持 19 次迭代和与基线
相同的点及原坐标误差，native/API 为 3.123 s（基线 3.206 s），但 CLI 墙钟为 4.95 s
（基线 4.53 s）。没有可信的整题提速，且实现增加了专用代码，故恢复原有分块路径。
回退后的 fast-profile medium E2E 复现 `AlmostSolved`、19 次迭代、原坐标 dual residual
2.8486e-6 与 gap 1.1209e-6（审计门分别为 1.75e-6、1e-6）；状态和残差与基线一致，仍
未通过既有外部门槛，fast 与 release 的时间不作比较。

随后隔离两个可能改变 Schur 数值路径的实现。临时关闭结构零列压缩、改走完整 GEMM 后，
medium 仍是同一个 `AlmostSolved` 点和相同原坐标误差，故该分支不是当前精度偏差来源；
结构压缩路径已恢复。临时把 PSD `apply` 从合并后的 `G/Ginv` 两次 GEMM 改回因子 `R/Rinv`
的四次 GEMM，结果变为 18 次 `Solved`，gap 降至 7.54e-7，dual residual 降至 1.916e-6，
但仍超过 1.75e-6 外部门槛；API/native 增至 9.354 s，约为同配置单次变换的 3.27 倍。
因此恢复更快的合并变换。当前证据将优先级指向末端方向/poor-progress 轨迹，而不是 Schur
面板融合、结构零压缩或 componentwise 判据；后者只增加 full `Solved` 的额外条件，无法改变
当前 gap 未达 full tolerance 后的 poor-progress 回退。

#### MPFR Ising Λ11，512-bit，内部 1e-42（50 it，接受）

| 线程 | native s | kkt update | kkt solve | scale cones | step len |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.6 | 14.2 | 7.3 | 11.2 | 7.6 |
| 8 | 8.3（5.4×） | 2.7 | 1.5 | 1.75 | 1.2 |

注：上表在低电量降频下测得；同臂接电复测 t1 = 22.33 s（约 2× 差距，跨电源状态
的绝对时间不可比，相位占比仍有效）。接电同批 A/B：

| 臂 | native s | it | 说明 |
|---|---:|---:|---|
| base（`mpfr_fma` 链 dot） | 22.33 | 50 | objective、receipt 残差逐位一致 |
| `exactdot` 定点精确累加 | 17.47（−21.8%） | 50 | 末端一次舍入；单批 E2E 初测，非稳定性能结论 |

在此基础上，MPFR SVD Givens 两分量改用两项 `exactdot` 后，512-bit Ising 同配置
release E2E 为 API/native 17.219/17.219 s，对照原旋转路径 17.730/17.728 s，单次约
−2.9%；两者均 `Solved`、50 次迭代，独立原问题审计 `accepted=true`。外部墙钟计时
方向相反，故只记为初步信号，不外推稳定收益。

2026-09-23，PSD 最小特征值路径把 `svec_to_mat` 后的对称矩阵缩放改为只算上三角并
镜像结果，避免重复 MPFR 乘法。Λ11 512-bit、单线程、内部 `1e-42` 的 release E2E
为 15.953 s、50 次迭代；独立原问题审计 `accepted=true`，原坐标 primal/dual/gap 均
小于 `1e-30`，峰值 RSS 104.8 MiB。相较此前 17.219 s 记录约快 7.4%；这是不同单次
计时之间的初步信号，尚不能排除机器状态波动。

改后采样归属：exactdot 31% inclusive，leaf 以 `__gmpn_addmul_1` 36% 为主（已受
原始乘法吞吐约束）；其次 `svd_rotate` 16%、`eig` 三对角化 7%、逐元素 mul_add 14%。

t1 相位累计：cone_svd 10.0 s（22%）、residual 7.8（scale 4.3、fwd 1.9、adj 1.6）、
cone_eigmin 4.5、recover_rhs 4.0、prepare_rhs 3.6、cone_wprod 3.3、sync 1.7、
cones_schur 1.4、refactor 1.4、ir 1.0、assemble 0.27。**MPFR 时间集中在锥（SVD/eig/W 乘积）
和 RHS/残差中的块同构，而不是 Schur 分解。** G·X·G 改动对 Ising 无变化
（43.6 s vs 43.6 s；residual.scale 未降，原因待查：sampled 块走 fused Condense/Recover，
普通 apply 路径不是瓶颈的假设尚未验证）。

#### MPFR 标量内核微基准（单核，每项 ns，k=8/32/128 基本不变）

| 位数 | `mpfr_fma` 链 | `mpn_mul_n`+移位+`mpn_add` 精确累加 | 比值 |
|---:|---:|---:|---:|
| 256 | 64–74 | 23–26 | 2.5–3.1× |
| 512 | 99–114 | 54–55 | 1.8–2.1× |
| 768 | 154–172 | 102–103 | 1.5–1.7× |
| 1024 | 223–241 | 172–174 | 1.3–1.4× |

上界估计未含真实指数对齐分支与末端一次舍入。内核已按 `mpn_mul_n` 全积、定点累加、
一次 `mpfr_set_z_2exp` 实现（`exactdot.rs`）并接入 `MpFloat::dot_fma`；Ising Λ11
单批 E2E 初测为 −21.8%，后续性能结论以固定精度完整求解和原问题残差为准。

### 当时的问题清单（收益 × 风险排序，含证据）

| # | 问题 | 证据 | 优化方向 | 风险 |
|---|---|---|---|---|
| 1 | f64 Schur dot_scatter 仍是 medium 主要热点 | 冻结 E2E profile：1.12 s / 3.7 s；按结构模式重排等 nnz 列的候选由 2.389 s 增至 2.519 s；新增稀疏左列等价复用的 A–B–B–A 为 2.338 s 对 2.324 s；两项均未提速，已撤回 | 保留现有四路 FMA 与分块实现；下一步应减少大块矩阵级工作，而非增加别名查找 | 中：medium 当前外部 1e-6 审计未通过，任何结果均标记为未验收 |
| 2 | 真正稠密的 f64 PSD 块可考虑平方根 Hessian 与 SYRK | Hypatia 的变换后 Gram 恒等式适用；当前 medium 五块的系数密度只有 0.386–0.710%，全量 SYRK 运算量远高于现有稀疏点积 | 仅在后续真实稠密 SDP 输入中评估全稠密块分支；不向当前 medium 强行引入 Gram 路径 | 中：额外内存及求和顺序变化需由完整求解评估 |
| 3 | MPFR SVD 的 Givens rotate 仍逐元素执行多次标量运算 | Ising Λ11 profile 中 svd_rotate 约 16% inclusive；两项 exactdot 候选已通过 Ising E2E 与外部精度审计，单次约 −2.9% | 保留候选；若继续做 MPFR E2E 优化，再试 WY/GEMM 或特征值内核 | 低到中：当前收益仍是单次测量 |
| 4 | MPFR SVD/eig/W 乘积仍占高精度主耗时 | 当前 Ising Λ11 完整求解阶段记录：SVD 4.85 s、eigmin 2.28 s、W 乘积 0.95 s / 总 16.19 s；Jordan 对称化与二项 `fmma` 的组合在同机 A–B–B–A 中为 16.218 s 对 16.459 s，50 次迭代、外部审计均通过；eig Householder 对称更新在新的配对中为 15.933 s 对 16.197 s，原问题审计通过。SVD 2 的幂归一化、eig 两项 `fmma`、SVD 常量缓存、Householder 分量除法倒数化均无收益并已撤回。整块缩放共用倒数初版在 t1/t4 快 0.60%/0.85%，补齐极端指数回退后 t4 反慢 0.66%，亦撤回。随后把 eig 三对角化的行内积改用现有精确累加接口，Λ11 512-bit A–B–B–A 在 t1 为 15.765 对 15.911 s（快 0.92%），t4 为 5.203 对 5.222 s（快 0.37%）；全部 50 次迭代、`Solved`，独立原问题审计通过 | 保留行内积精确累加；停止针对单一除法做小幅替换，优先找 SVD/eig 矩阵级冗余，再评估 WY/GEMM 或结构化 PSD Schur/W 乘积 | 中：配对收益小于 1%，尚不能外推跨机幅度 |
| 5 | RHS/残差路径重复做块同构与多波次求解 | 当前 Ising Λ11 阶段记录 prepare/recover/residual 分别为 1.06/1.21/1.73 s；solve_many 已存在；跳过 fused sampled recovery 的普通线性 GEMV 在同题配对 E2E 无可信收益，已撤回（其 CSC 本来不含采样 PSD 行） | 合并真正可共享的 RHS，并复用当前迭代的缩放乘积；优先审阅原算子剩余计算量 | 中：依赖关系需保持，按完整求解耗时评估 |
| 6 | condensed 外层和 Schur 内层存在精化工作重叠 | 静态调用审阅确认外层 correction 再调用带内层 IR 的 reduced solve；Λ11 512-bit 的 163 次 reduced solve 中仅 10 次来自外层 correction，内层累计 87 次修正（0.73 s），所以只在外层 correction 跳过内层 IR 的上限很小，且两层分别检查不同算子 | 暂不取消内层 IR；若 Float64 可验收工作负载显示其占比显著，再用独立候选检验 | 中：不能仅凭调用嵌套认定精化冗余，必须保留原问题精度 |
| 7 | BLAS 线程不完全纳入求解器线程预算 | settings.threads 主要控制锥池 | 明确线程预算，避免求解器池与 BLAS 过量订阅 | 中：仅在多线程 E2E 对比中验收 |
| 8 | 两条 MPI 路径并存，普通路径有逐 site gather | 既有跨节点记录有负扩展；owner-partitioned 已覆盖对称 LP/SOCP/SDP，但 CLI 无 `--partitions` 时和 C ABI 仍走普通路径，owner 路径尚不支持非对称锥或 augmented KKT，真实 MPI E2E 尚未合格 | 先补实际 MPI E2E 与默认路由/回退，再收敛普通 gather；不能仅凭 Ising mock 测试删掉通用路径。2026-09-23 复核：owner 路径仍拒绝非对称锥与 augmented KKT，本机无 MPI 运行时，故普通路径保留 | 高：受 runner 和集群资源支持限制 |
| 9 | dispatch 成本阈值散落且与机器相关 | rate、填充、Arrow、内存等常数来自启发式 | 先维持当前值；仅在目标完整求解明确受其影响时调整 | 低 |
| 10 | 代码规模与大文件职责重叠 | 当前约 41.5k 有效代码行；condensed_psd、sampled、compositecone 较大；内部未使用的 LU 与特征向量包装累计净减 211 个物理行 | 继续按生产调用证据去重，目标生产约 30k 有效代码行。2026-09-23 已完成目录重组（`solver/{core,default,cones,kkt,sampled,distributed,chordal}`，测试移入各模块 `tests/`，`distributed` 仅在 `sdp` 下编译）并删除库与测试均未使用的项；medium 与 Ising Λ11 E2E 输出逐位不变。行数基本未变，真正减量仍需去重 | 中 |
| 11 | release 构建较慢、debug 产物占用大 | 本轮记录：release 构建约 6 min，target/debug 约 59 GB | 迭代使用 fast profile；清理产物不作为性能里程碑验收。2026-09-23 已清理 target/debug（target 83 GB → 7 GB） | 低 |

