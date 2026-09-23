# Benchmark library and research loop

固定数据、评估器和数值契约。当前性能计划以一次代表性完整 E2E 求解作为每项改动的
验收：检查状态、原坐标残差/gap 和总时间。保存候选结果，不自动覆盖工作区源码；
多题、重复 A/B、微基准和全套测试不作为逐项改动的门槛。

流程借鉴 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 的小步实验、固定评估、保留/丢弃和实验记录。求解器需要额外保留精度、失败覆盖率、各类问题退化及内存约束，不能只优化一个时间数字。项目执行规则见 [AGENTS.md](../../AGENTS.md)。

## 执行范围

每项 E2E 执行有明确的假设、输入和结束条件；完成求解、原问题审计与候选决策后报告结果。
默认由主 agent 完成，不自动启动后台研究或集群任务。

`tree.py` 保留候选谱系和 `record/cut/merge/adopt` 状态管理功能；
`hpc.py` 保留 PBS 提交、查询和结果获取功能。需要集群实验时，按用户指定
范围使用这些工具：先检查现有作业和预算，冻结候选，再提交并收集结果。
这些工具的容量上限不是必须消耗的资源配额。命令参数以 `--help` 为准。
清理候选前先保留结果，并确认没有活动作业使用它。

## 日常开发与回归

每项改动运行一个与目标路径相符的完整 solve。一次 E2E 同时记录终止状态、原坐标
残差/gap 和求解时间；不要求逐项改动另跑 unit/focused tests、screen、重复 A/B、
microbench、线程矩阵或全套测试。`pair`、`qualify` 和参考适配器保留给最终外部比较或
用户明确要求的批量研究，不是日常推进门槛。计时受噪声影响时如实标成初步结果，不因此
自动增加重跑次数。

数值通过标准是现有状态语义、原问题残差/gap/目标误差门槛。迭代数和耗时用于诊断或性能
分析；有效算法调整可以改变它们，不要求逐题精确复现旧的迭代数或目标值文本。既有
`SDP_control3` 精度失败保留为失败，不放宽门槛，也不阻断无关局部改动。

用户于 2026-09-21 明确暂缓 `SDP_control3` 的既有精度问题：输入 SHA256
`d3a7cc2724fbf39b819f34e8c191bda4a7c3f3669772fc0b5552267d26286e84`，
更早基线与候选同为 29 次迭代、原坐标 primal residual 约 `4.98e-4`。
该题保留在完整题库，原失败收据保留，**不得标成通过或放宽容差**；
这个已复现的既有问题暂不阻断局部性能候选保留。新失败、新状态或残差恶化仍需检查。
完整评估器的严格判定不改写，局部验收不得冒充整库通过。

## 数据库

`catalog.json` 固定输入身份、来源、结构、角色与精度；`catalog.py` 仅校验和复制/解压输入，不调用求解器。运行数据、源码副本、构建产物和日志放在仓库外。

| Suite | 内容 | 用途 |
|---|---|---|
| `smoke` | 3 个有解析最优解的 LP/SOCP/SDP | 检查完整接口与外部验算，不作为性能排名 |
| `development` | 9 个历史实例，每类按规模选小/中/大各一个 | 较完整的集成/候选评估；不是每次编辑的门槛 |
| `regression` | 52 个公开实例和 3 个固定合成实例 | 保留所有已暴露题目，包括原来的 holdout |
| `holdout` | LP_ship04s、SDP_copo14、混合锥 SDP_filter48_socp、SOCP_strictmin_2D_43_dual，均未求解 | ship、共正性、PAM 滤波器和几何 ARAP 家族；large SOCP 设 180 秒上限 |
| `mpfr-dev` | 一个 orthant LP、两个 sampled SDP 配方 | 256/512 位及 1/2/4/8 线程的定向诊断 |

原目录中的 46 个历史 JSON 已从保留的集群快照恢复到 `benchmark/float64/data`，逐个匹配 catalog 原 SHA256；只迁移存储路径，未改变角色或输入。development 9 项、regression 55 项、mpfr-dev 3 项均通过 materialize 前置校验。MOSEK 驱动和依赖指纹工具位于 `benchmark/float64/adapters`；实际参考运行仍需安装包及有效许可证。

原六项及 ship 来源为固定的 ClarabelBenchmarks revision `3679912c6bbd3f64c5c962f9d1c09d524561c412`；源 URL、原始/转换后 SHA256、转换信息和归属保留在目录中。六个压缩输入约 23 KB，原始来源与许可见 [data/holdout](data/holdout)。

9 月 16 日对比 MOSEK 后，原 6 个保留实例已转入回归，不再作为独立验收。
另外加入两个固定种子的 SOCP 和一个 80 阶 PSD 合成问题，全部使用预先构造的
可行、互补 primal/dual 点；它们是结构诊断，不能代表公开问题的总体表现。
`selection_policy.refresh_20260916` 固定这次的 9 个成员。三个新输入可以运行
`generate_planted.py --output /absolute/new-directory` 重建，目录中的压缩输入和
catalog 哈希可逐字节核对。没有依据求解成功或速度删除、替换成员。
新预留 `LP_ship04s`（1506 变量、1908 行、5906 非零元）仅做来源、转换和格式检查，
没有运行求解器。原始/转换哈希与 180 秒、4096 MiB 的计划预算记录在 catalog；
`run.py` 自动取题目 reservation、命令行和剩余总预算中的更小上限，并在每次 process 记录中写明实际时间/内存限制；更短的用户预算不会被延长。
另预留 DIMACS 官方归档的 `copo14` 与 `filter48_socp`，原始/转换哈希及来源记录在 catalog 和 attribution/DIMACS.json。后者含 PSD48 和 SOC49，不能算纯 SOCP 保留集。
`import_sedumi.py`（可选依赖 NumPy/SciPy）要求原始 MAT 文件 SHA256，将对称 PSD 变量映射为上三角 svec，保留全部等式和目标符号；不启动求解器。两题均只完成格式和独立算子检查，计划预算各 180 秒/4096 MiB。可见历史记录未发现同族暴露，但不据此声称覆盖已删除历史。纯 SOCP 几何 ARAP 家族已由 `strictmin_2D_43_dual` 补齐。`ss30` 因仍属已暴露的
桁架家族未纳入；`db_shear_wall` 仅下载检查，较大规模需要独立验收预算。

```sh
python3 benchmark/research/catalog.py list --suite development
python3 benchmark/research/catalog.py materialize \
  --workspace /absolute/SDPX-workspace --suite development \
  --output /absolute/external-cache/development-v1
```

输入校验失败、缺失或重复身份冲突会明确失败。失败实例不删除、不换成容易的题。MPFR 配方在目标精度直接构造输入，由现有驱动记录实际数值哈希；不把 Float64 JSON 升格为原生高精度数据。

## 单轮执行

Python 3.11+，macOS/Linux；native `sdpx` executable and any independent oracle
environment must be prepared in advance. Measurements on one machine run
serially. All commands share the `--state` file lock; coordinate other numerical
tools as well.

每个 arm 用一个 JSON 配置，路径均指向已冻结的来源：

```json
{
  "source": "/absolute/frozen/candidate",
  "cli": "/absolute/frozen/candidate/target/release/sdpx",
  "blas": "accelerate",
  "env": {"RAYON_NUM_THREADS": "1"},
  "provider_files": ["/absolute/extra-provider-library-if-needed"],
  "oracle": {"julia": "/absolute/julia/bin/julia", "project": "/absolute/oracle-env"},
  "qualification": "/absolute/qualification/qualification.json"
}
```

`cli` must be an absolute executable. `blas` and `provider_files` describe the
external runtime when applicable. `oracle` is optional and is used only for an
independent audit/input generator; it is never a solver frontend. A historical
`library` field may be retained for identity records, but it is not a native
binding and a missing file fails clearly. Source and CLI are hashed separately;
the hashes do not claim an inferred build relationship.

`fingerprint` records production Rust files, Cargo.lock, the CLI, optional
provider/oracle files and environment hashes. The complete mapping includes
uncommitted changes, so a Git checkout is not required. Use it for final
published comparisons when source/artifact provenance matters; it is not a
prerequisite for each E2E optimization iteration.

每行保留 CLI 路径、产物哈希、版本与求解状态；这些不混入两臂必须相同的
`settings_sha256`。该哈希比较实际数值设置、精度、方向、KKT/线性后端与线程预算；
产物身份在各 arm 内独立验证，状态与原问题残差仍单独验收。

```sh
python3 benchmark/research/run.py fingerprint --config /absolute/candidate.json
python3 benchmark/research/run.py qualify \
  --config /absolute/candidate.json --commands /absolute/frozen-checks.json \
  --output /absolute/experiments/qualification-candidate
```

`frozen-checks.json` 仅供需要严格来源追溯的最终实验使用，不定义日常优化的必经门槛。

```sh
python3 benchmark/research/run.py pair \
  --baseline /absolute/baseline.json --candidate /absolute/candidate.json \
  --data /absolute/external-cache/development-v1 --stage development \
  --profile full --threads 1 --output /absolute/experiments/iteration-001 \
  --state /absolute/experiments/state --budget-seconds 3600
```

`pair` 是需要外部相对性能报告时使用的可选成对工具。它会按 A/B 顺序运行
fresh native CLI processes，分别记录 `api_seconds`（native setup+solve）、
`native_seconds`（solver timer）、`load_seconds`、CLI wall time 和外部 oracle。
日常优化只需跑一组匹配配置的完整求解，不要求执行 `pair` 或其多轮矩阵。
Float64 对照保持既定内部/外部容差以及默认 Ruiz/presolve/chordal。

full 默认每进程上限 900 秒、4096 MiB；单轮总预算 3600 秒，可在实验开始前固定其他预算。超时、OOM、异常及未执行的 case 都保留，不能从分母排除。进程组清理不能确认时停止后续启动。前后源码、依赖、库、评估器或输入身份变化会使结果失效。

### 结果与计量

`baseline.json` / `candidate.json` 保存规范化记录，子目录保存 stdout/stderr、原始设置、
残差、命令和内存记录。日常改动以同一输入、精度、容差及执行配置下的一次完整 E2E
和原问题审计判断：状态与原问题误差达标才算通过；失败如实保留，不得放宽门槛。
`pair` 的多样本聚合规则只用于可选的最终相对性能报告，不阻塞普通开发。

单次计时代表该次机器状态，不是统计显著性结论；如计时受干扰，标注为初步测量。时间、
内存、迭代数及失败状态分别记录，微核收益不能代替端到端收益。

当前主指标是 native `api_seconds`（native setup+solve）。`native_seconds`、
`load_seconds`、CLI process wall time 和 external audit time retain separate
scopes; a warm row always means another fresh CLI process. RSS is the OS
high-water mark of that native child, including input/result serialization. Keep
Rust-core comparisons and any oracle process memory separately labelled.

`--state/results.jsonl` 追加每次候选决策并保留失败；完成预算内的实验后报告结果，不自动修改主目录或 Git 状态。

## 保留集与参考求解器

`pair --stage holdout` 需要 `--development-result /absolute/kept-development/comparison.json`，并重新检查同一候选的开发阶段决策。manifest 角色和完整成员必须匹配固定 catalog，不能通过改 `--stage` 绕过。读取保留集会将原始/转换后输入身份追加到同一个 state 的访问记录；改名称或重打包 manifest 不能重用已经暴露的输入。保留集一旦用于反馈，即转为后续回归资料，下一轮需要预留新的版本。

参考比较复用相邻稳定 benchmark 的 Clarabel.rs/MOSEK 驱动与 oracle：

```sh
python3 benchmark/research/references.py \
  --data /absolute/external-cache/development-v1 --engine clarabel \
  --workspace /absolute/SDPX-workspace \
  --binary /absolute/frozen/clarabel-benchmark --source /absolute/frozen/clarabel-harness \
  --output /absolute/experiments/reference-clarabel

python3 benchmark/research/references.py \
  --data /absolute/external-cache/development-v1 --engine mosek \
  --workspace /absolute/SDPX-workspace --python /absolute/mosek-env/bin/python \
  --output /absolute/experiments/reference-mosek
```

本版参考适配器仅支持 **1 线程 Float64**，拒绝保留集，强制 Clarabel 默认预处理；MOSEK 保持产品默认内部容差，报告相同外部门槛及真实设置差异。不可把 API/运行时不同的结果直接送入候选晋级器；参考摘要单独保留。缺少二进制、许可证或依赖是失败/不完整，不能自动跳过。Clarabel 的原生依赖可用 `--provider-file` 显式固定；源码和二进制的独立哈希不构成构建证明。

MPFR 高精度优化以对应 Ising 完整 E2E 路径验收。单节点线程和 MPI 比较留到并行里程碑；
每个数据点均为完整求解，不把独立内核计时或批处理吞吐算作单次求解加速。

真实高精度 SDP 的阶段验收继续使用 [Ising controller](../ising/README.md)，复用同一输入、同一精度/容差和同一集群分配比较 SDPX/SDPB。它是独立里程碑，**本模块尚未自动归一化 SDPB/MPI 结果**；不自动提交集群任务。更大的 Ising 和新的 bootstrap 问题先固定数据与门槛，再运行；synthetic sampled 提速不代表 Ising 提速。

## 验证本工具

工具测试和 `smoke` A/A 可按需运行；它们不替代完整求解，也不是每项求解器性能改动的验收门。

`SOCP_strictmin_2D_43_dual` 来源为 [CBLIB](https://cblib.zib.de/download/all/strictmin_2D_43_dual.cbf.gz)，保留全部连续变量、等式与二阶锥；101676 变量、111757 行。转换仅使用 MOI 文件读取及仿射导出，独立核对原始 CBF 系数与锥嵌入，未求解。来源、哈希、许可见 attribution/strictmin.json 和 CBLIB 许可文件 `data/holdout/attribution/CBLIB-README.md`。最终验收由 runner 自动封顶 180 秒、4096 MiB；超时作为结果保留，不用于开发调参。
