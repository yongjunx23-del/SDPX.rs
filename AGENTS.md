# SDPX

## Working agreement

- **Unless the user explicitly requests subagents for the current task, do not
  use subagents.** Implement and review changes in the main agent.
- Complete requested work and affected checks. Broad regression belongs at
  integration milestones; documentation-only changes need document checks.
- Preserve unrelated edits, sibling/reference repositories and licenses.
  Commit, push and remote work must remain within the user's authorized scope.

## Solver contracts

- Keep one Julia frontend / Rust solver / native-library architecture. Reuse
  Clarabel.rs algorithms and mature libraries; preserve upstream attribution.
- Preserve precision, original-coordinate outputs, accepted-iterate recovery,
  convergence/infeasibility criteria and regularization/refinement. No looser
  accuracy gates, mixed precision, precision-ladder warm starts, approximate rank
  reduction or benchmark-name branches.
- Runtime numerical checks follow Clarabel.rs. Do not restore the retired
  independent certificate stage, status promotion or SDPX-only five-equation
  correction gates. Tests audit returned points/rays outside solver timing.
- Direct solves default to Ruiz, presolve and chordal decomposition. Prepared
  handles retain Ruiz and disable structural preprocessing for q/b updates.
  Preserve upstream reduced tolerances; AlmostSolved earns no full-accuracy credit.
- Mutable MPFR values must own storage. Sampled factors define their operator;
  do not replace authoritative rounded CSC data with approximate factors.
- Prefer small typed interfaces. Check consumers before retiring code. Aim for
  20–30k production lines, counting adapted code and implementations behind wrappers.

## Evidence and task guidance

Freeze source, dependencies and inputs before timing. Run measurements sequentially
on each host. Record identities, precision, settings, providers, threads and failures.
Separate native/API, cold/warm time and process memory. Retain performance changes
with repeatable ≥2% median improvement or a justified correctness/memory benefit.

Read only the relevant entry:
- [README](README.md): installation, API and architecture.
- [Review and plan](REVIEW_AND_PLAN.md): current priorities and unqualified work.
- [sdpx-development](.agents/skills/sdpx-development/SKILL.md): build/test setup,
  precision ownership and benchmark routing.
- [Benchmark protocol](benchmark/research/README.md): research experiments.
  Use the available `ucas-hpc` skill for actual cluster work.

Guidance follows OpenAI's [skills and prompts advice](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra):
keep project-specific constraints here and load operational detail on demand.
