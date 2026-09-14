# SDPX autonomous research protocol

目标：在固定准确性下改善 LP/SOCP/SDP 的时间、内存和单次求解多核效率，保持一个简洁的求解器实现。执行工作区 AGENTS.md；本文件不授予发布、提交、远程任务或无限运行权限。

本流程参考 [autoresearch 的 program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)，采用独立实验与固定评估；不复用它的训练代码、Git reset 或无限循环策略。

## 启动前固定

1. 研究周期配置：目标精度和硬件；最多实验数和墙钟时间；数据/catalog 版本；回归/qualification 命令；内存/时间上限；baseline 构建身份；同一个 `--state`。未另指定时，计划最多 10 个实验、2 小时；这是执行预算，不是新的无限目标。
2. 先读 README 和源码，不依据排行榜挑题。现有 holdout 已经暴露的就作为回归资料；新保留集只能在候选里程碑读取结果。本版新保留集缺 SOCP，不宣称完整三类泛化。
3. 保持 benchmark、oracle、参数、停止条件、精度、tolerance 和晋级规则固定。改变评估器应单独审阅并开始新周期，不能同时修改求解器和裁判。
4. baseline/candidate 各用仓库外独立源码副本；记录锁文件、编译器、build argv、features、动态库和本地依赖。测量阶段冻结所有来源。没有 Git 也使用完整文件哈希，不把 uncommitted 改动漏掉。

## 第一轮先处理数值阻塞

当前审阅暴露了以下必须有针对性回归的行为。先建立测试并修正，不能因既有测试通过就跳过：

- 带主元三对角求解：T=[[0,2,0],[2,3,1],[0,1,4]]，rhs=[1,0,0]，正确解首项 -11/16；全部 MPFR 档位检查精确残差。
- Sturm：d=[0,0]、e=[1]、x=0 应计数 1；补充恰好命中特征值端点的约定。
- NT Gram 与因子重排：病态、非对易 SPD、梯度谱/重复谱，验证固定精度的分解/方向运算；高位数不能代替稳定性论证。
- λmin 索引路径：直接测试原语并区分快路径与完整 QL 回退，避免自比较隐藏错误。

这些测试属于外部测试层。不要恢复独立生产证书或 SDPX 特有的收敛/方向验证阶段；保留 Clarabel 的数值判据与必要的 ABI/内存/精度检查。

## 每个实验

1. 从最后验收的 baseline 开一个外部候选副本。写清一个假设、影响的输入结构、预期减少的工作、需要验证的数值风险及最小测试。不要按实例名称分支；不要同时改迭代算法、provider 和调度。
2. 独立的实现任务可以由多个 agent 并行完成，每个拥有不同文件/副本；只保留一个数值执行器。依赖变更顺序集成，禁止竞争性 benchmark。
3. 做最小实现并让另一位审阅者检查。优先复用 Clarabel、SDPB、Hypatia、COSMO 与成熟 provider 的设计；保留许可和归属。没有证据时不增加通用框架或额外运行时检查。
4. 冻结候选，运行事先固定的 qualification。所有要求的测试通过并绑定实际产物才具备速度评分资格。修复正确性而仍未通过全套时，记录改善和未完成项，不算速度胜利。
5. 运行 `smoke` 检查适配器；再按目标精度运行 `development` 的 AB/BA。每个进程的超时、OOM、外部验算失败、非完整准确状态都保留；不能删除困难实例、扩大门槛或降低精度。
6. 阅读 `comparison.json`，同时看逐类/逐题分布、RSS、冷/热时间与迭代数。`keep` 只把候选加入本阶段短名单；`correctness_only` 只承认数值覆盖进步；其余丢弃或修复明确的实现错误。用明确新实验记录重试，不覆盖日志。
7. 对短名单执行全部相关回归、目标 1/2/4/8 线程，核对真实 provider/线程设置。不同精度单独判断，不能用 Float64 的大收益掩盖高精度退化；native kernel 与 Julia API 时间分别报告。
8. 在独立时段复验接近 2% 门槛的候选。最后才读取尚未暴露的保留集；通过后仍明确其结构覆盖限制。一般算法改动要补新的 SOCP/大 SDP；高精度结构改动按已授权范围安排 Ising/SDPB 同机验收。
9. 所有相关验收完成后，输出可审阅的补丁、源码/产物身份、准确率与性能分布，才更新下一轮 baseline。保持主工作区用户修改；不要自动 git reset、提交、推送或删除他人的候选。

## 搜索顺序

1. 已知数值错误与测试盲点；失败覆盖率改善。
2. 无需改变数学的重复工作：多余 Gram/cache、空 P 遍历、重复 FFI 转换与不可用输出恢复。
3. 实际热点的 MPFR GEMM/SYRK 面板、分块分解与多 RHS；用微核解释机制，用完整求解决定是否保留。
4. 分阶段负载调度、尾部块和内层并行；维护单一线程预算与确定的归约顺序，避免过度订阅。
5. 有相应理论和回归证据后才做校正步、KKT 换型等算法实验。MPI 和多节点排在可测的单机瓶颈之后。

不要重复实现已有的共享池、排序复用、因子复用和 sampled 算子；先检查当前代码。

## 停止与输出

达到预设实验数/时间预算、用户停止、清理无法确认、缺少必要输入或验收失败时，保留记录并结束当前周期。只报告实测事实、未完成项和下一项最高价值实验；不把“代码已写完”当作数值或性能验收，也不把共同比较子集的速度当作总体领先 MOSEK/SDPB。

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
