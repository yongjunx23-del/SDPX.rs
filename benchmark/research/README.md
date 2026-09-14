# Benchmark library and research loop

固定数据、评估器和数值契约，在独立源码副本中逐项尝试优化。每轮先检查正确性，再比较时间和内存；输出候选决策，不自动覆盖工作区源码。

流程借鉴 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 的小步实验、固定评估、保留/丢弃和实验记录。求解器需要额外保留精度、失败覆盖率、各类问题退化及内存约束，不能只优化一个时间数字。供 coding agent 执行的说明在 [program.md](program.md)。

## 数据库

`catalog.json` 固定输入身份、来源、结构、角色与精度；`catalog.py` 仅校验和复制/解压输入，不调用求解器。运行数据、源码副本、构建产物和日志放在仓库外。

| Suite | 内容 | 用途 |
|---|---|---|
| `smoke` | 3 个有解析最优解的 LP/SOCP/SDP | 检查完整接口与外部验算，不作为性能排名 |
| `development` | 9 个历史实例，每类按规模选小/中/大各一个 | 日常反馈；选择依据是尺寸，不是求解结果 |
| `regression` | 46 个去重后的历史公开实例 | 保留所有已暴露题目，包括原来的 holdout |
| `holdout` | 新预留的 4 LP、2 SDP | 候选里程碑验收；本版尚无新 SOCP |
| `mpfr-dev` | 一个 orthant LP、两个 sampled SDP 配方 | 256/512 位及 1/2/4/8 线程的定向诊断 |

历史题库复用相邻 `SDPX.jl/benchmark/data`，不再依赖某次 `/tmp` 历史任务。新保留集来自固定的 ClarabelBenchmarks revision `3679912c6bbd3f64c5c962f9d1c09d524561c412`；源 URL、原始/转换后 SHA256、转换信息和归属保留在目录中。六个压缩输入约 23 KB，原始来源与许可见 [data/holdout](data/holdout)。

新实例与已盘点的历史名称/输入哈希不重合；这不是从未被任何人或任何 agent 看过的证明。选集时没有调用求解器或查看排名。已完成结构检查，**尚无这些新实例的求解准确性或速度结论**。它也不能单独证明 SOCP 泛化。后续应预先固定新版 CBLIB/Hypatia SOCP 与更大 SDP，并增加不同结构家族；同一个参数化家族应一起分组，不能把近似副本分进开发和保留两侧。

```sh
python3 benchmark/research/catalog.py list --suite development
python3 benchmark/research/catalog.py materialize \
  --workspace /absolute/SDPX-workspace --suite development \
  --output /absolute/external-cache/development-v1
```

输入校验失败、缺失或重复身份冲突会明确失败。失败实例不删除、不换成容易的题。MPFR 配方在目标精度直接构造输入，由现有驱动记录实际数值哈希；不把 Float64 JSON 升格为原生高精度数据。

## 单轮执行

Python 3.11+，macOS/Linux；Julia、数值依赖与共享库必须事先准备好。相同机器上的测量串行执行。所有本流程命令共享 `--state` 的文件锁；其他工具的计时也需要协调。

每个 arm 用一个 JSON 配置，路径均指向已冻结的来源：

```json
{
  "source": "/absolute/frozen/candidate",
  "library": "/absolute/frozen/candidate/lib/libsdpx.dylib",
  "julia": "/absolute/julia/bin/julia",
  "julia_project": "/absolute/frozen/candidate-benchmark-env",
  "blas": "accelerate",
  "env": {"JULIA_DEPOT_PATH": "/absolute/frozen-depot:/absolute/fallback-depot"},
  "provider_files": ["/absolute/extra-provider-library-if-needed"],
  "qualification": "/absolute/qualification/qualification.json"
}
```

Linux 库后缀用 `.so`，`blas` 通常为 `default`；删除不需要的 `provider_files` 示例项。benchmark 环境需有 SDPX、JSON；Accelerate 分支需要 AppleAccelerate，sampled MPFR 还需要 GenericLinearAlgebra。环境中的 SDPX 必须解析到 `source/julia/SDPX.jl`，驱动会检查。两个 arm 的依赖版本、Julia、BLAS 和有效运行环境应一致；仅源码/产物身份可以不同。

`fingerprint` 记录生产文件、Cargo.lock、库、Julia、环境与本地 path 依赖的哈希。完整项目文件映射包括未提交改动，因此不要求项目已经是 Git 仓库。源码和产物之间的构建关系仍需真实的锁定构建记录，独立哈希本身不证明二者对应。

```sh
python3 benchmark/research/run.py fingerprint --config /absolute/candidate.json
python3 benchmark/research/run.py qualify \
  --config /absolute/candidate.json --commands /absolute/frozen-checks.json \
  --output /absolute/experiments/qualification-candidate
```

`frozen-checks.json` 是事先审阅过的 argv 数组列表，命令使用绝对路径，构建/测试目标指向该源码副本。可包含锁定的 Rust provider/solver 测试、Julia 前端检查和此次改动相关的数值回归。命令不经过 shell。命令和验算标准属于固定评估器，优化 agent 不得把它们改成无效检查。缺失、失败或身份不匹配的 qualification 都不能带来速度晋级。

```sh
python3 benchmark/research/run.py pair \
  --baseline /absolute/baseline.json --candidate /absolute/candidate.json \
  --data /absolute/external-cache/development-v1 --stage development \
  --threads 1 --output /absolute/experiments/iteration-001 \
  --state /absolute/experiments/state --budget-seconds 3600
```

每个 case 先 A→B，再反向 B→A；第二块还反转 case 顺序。每个进程执行一次冷调用和三次 warmed **fresh solves**，每个 arm/case 共八次调用。复用已有原坐标外部 oracle；没有 prepared handle 重用。Float64 数值设置保持现有完整容差 1e-8、外部门槛 1e-6，以及默认 Ruiz/presolve/chordal。线程预算只取 1/2/4/8；Julia/BLAS 各为 1，锥和 Faer 使用记录的预算。

默认每进程上限 900 秒、4096 MiB；单轮总预算 3600 秒，可在实验开始前固定其他预算。超时、OOM、异常及未执行的 case 都保留，不能从分母排除。进程组清理不能确认时停止后续启动。前后源码、依赖、库、评估器或输入身份变化会使结果失效。

### 决策与计量

`baseline.json` / `candidate.json` 保存规范化记录，子目录保存完整 stdout/stderr、原始设置、残差、命令和 wait4 内存记录。`comparison.json` 的决策规则：

- 必须完整匹配输入、精度、数值设置、运行时/provider、线程、计时范围及所有要求的样本；完整准确状态和外部验算都通过。
- 每个 case 使用 warmed 中位数；类内按 case 等权，LP/SOCP/SDP 类之间等权。AB 与 BA 两块都要求至少 1.02× 提速。
- 预设每类退化上限 2%，单 case 时间和峰值 RSS 退化上限各 10%。这是工程筛选政策，可在新研究周期开始前收紧；不能见到结果后调整。
- `keep` 是此阶段的候选保留；`discard` 丢弃实验；`correctness_only` 只记录正确性覆盖改善；`incomplete` 表示证据不完整。失败后不能只比较共同通过的子集来晋级。

这些是描述性的成对比较，不是统计显著性证明。接近门槛的候选应在独立时段重复相同协议。时间、内存、迭代数、每轮耗时及失败覆盖率分别保留；微核收益不能当作全求解收益。

当前主指标是 Julia API 的 fresh setup+solve+结果提取/清理时间；解析与外部检查在计时外。冷调用单列。RSS 是整个 Julia 子进程的 OS high-water，包含 JIT、输入和验算；不是单轮原生分配。原生 Rust 的比较使用已有 [build_kernel.py](../float64/build_kernel.py)，必须另外标注计时与内存范围。

`--state/results.jsonl` 追加每次候选决策，保留失败；不自动 commit/reset、修改主目录或启动无限后台任务。

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

MPFR 的日常循环使用 `mpfr-dev`、`--precision-bits 256|512 --stage development`。依次执行 1/2/4/8；不同精度分别评分。线程扩展记录 `S(p)=T(1)/T(p)`、效率 `S(p)/p`，与相同宽度的基线比较，不能把多进程批处理吞吐算作单次求解加速。

真实高精度 SDP 的阶段验收继续使用 [Ising controller](../ising/README.md)，复用同一输入、同一精度/容差和同一集群分配比较 SDPX/SDPB。它是独立里程碑，**本模块尚未自动归一化 SDPB/MPI 结果**；不自动提交集群任务。更大的 Ising 和新的 bootstrap 问题先固定数据与门槛，再运行；synthetic sampled 提速不代表 Ising 提速。

## 验证本工具

```sh
python3 -m unittest discover -s benchmark/research/tests -v
```

测试覆盖身份/角色校验、重复或丢失样本、错误状态、精度变化、NaN/无效计时、AB/BA 顺序依赖、内存退化、保留集重命名、命令绑定、生成器变更和进程清理失败。`smoke` 可对同一个已冻结产物做 A/A：应得到准确结果，但没有候选提速信用。

## Fast iteration

The default per-experiment development loop is focused tests, then `run.py pair
--profile screen` with the usual baseline/candidate/data/output arguments, then
full `cap` only at milestones. `pair` itself retains `--profile full` by default.
Screen uses the pinned development cases LP_afiro, SOCP_sambal and SDP_truss1
in catalog order, one thread, one cold plus one warmed fresh solve, and AB only.
Only the development stage is allowed. Process timeout defaults to 120 seconds
and the campaign budget to 600 seconds (approximately <= 10 minutes per iteration,
plus orchestration/cleanup); larger explicit values are clamped and recorded.
Screen verdicts are `screen_pass` or `screen_fail` and never grant speed credit.
All required samples must pass and each warmed case median must respect the
existing 1.02 regression ratio. Full evidence rules remain unchanged.

Per iteration, run only affected Rust/Julia test binaries relevant to the change:
for example, a filtered `cargo test -p sdpx-solver --features sdp-accelerate --lib`
and the affected integration test file. Reserve the full workspace suite,
Julia 1.12/1.13 runs, and regression/holdout/MPFR/Ising protocols for milestone
acceptance. This scheduling policy changes neither tolerances nor required
milestone coverage.
