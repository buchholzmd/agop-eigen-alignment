# Eigenspace Alignment of AGOP for Learning Multi-Index Models

This repository serves as a template for experiments.

__Contents__:
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Running on Fir](#running-on-fir)
- [Specifying the target](#specifying-the-target)
- [W&B metrics](#wb-metrics)
- [Repository structure](#repository-structure)
- [Running experiments](#running-experiments)
- [Contact & attribution](#contact--attribution)


## Introduction

<!-- placeholder for an optional banner image -->
<!-- <div align="center">
  <img width="600" src="banner.png">
</div> -->

One paragraph motivating the project. This can be, for example, the paper abstract.

## Dependencies

### Python
All experiments were performed using Python 3.11. We use `micromamba` to install and manage dependencies:


```bash
micromamba create -f jax_env.yml -y
```

To remove the current environment and restart fresh use:

```bash
micromamba env list
micromamba env remove -n agop-eigen-alignment -y
micromamba clean --all -y
```

If you need to reset the environment for any reason use:
```bash
micromamba env update -n agop-eigen-alignment -f jax_env.yml --prune
```

*Note: The current `jax_env.yml` contains all used packages. Remember to run `micromamba env export > jax_env.yml` when making changes.*

### Fir (Alliance cluster)

On Fir, use a virtual environment built from the Alliance wheelhouse instead of micromamba. Create it once, on a login node:

```bash
module load python/3.12.4
python -m venv ~/jax_env
source ~/jax_env/bin/activate
python -m pip install --upgrade --no-index pip
python -m pip install --no-index jax flax pillow matplotlib
python -m pip install --no-index "jax[cuda]"        # installs jax-cuda12-pjrt, jax-cuda12-plugin
python -m pip install --no-index optax pyyaml tqdm wandb ipython ipywidgets
# python -m pip install --no-index pandas           # only needed for scripts/aggregate.py
# avail_wheels --name "*pytorch*" --all_versions    # check out the available packages
# python -m pip install --no-index gpytorch==1.13   # if needed, install a specific version
wandb login                                         # once: saves your W&B API key to ~/.netrc
deactivate
```

The SLURM scripts activate the environment at `~/jax_env`, so keep that path. Jobs log to W&B online, which works because Fir compute nodes have internet access.

## Running on Fir

Run everything from the repository root. `<SWEEP-NAME>.yaml` is a file in `configs/` (for example `configs/full_sweep.yaml`), and every sweep config extends `configs/base.yaml`.

### From the command line

For quick tests, run the sweep runner directly on a login node. Login nodes have no GPUs, so force JAX onto the CPU:

```bash
source ~/jax_env/bin/activate
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python scripts/run_sweep.py --config full_sweep.yaml --chunk 0 --nchunks 438
```

The same run without W&B logging:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python scripts/run_sweep.py --config full_sweep.yaml --chunk 0 --nchunks 438 --no-wandb
```

> **Note:** Setting `WANDB_MODE=offline` has no effect in this repo. The W&B mode comes from `wandb.mode` in the config, which is passed directly to `wandb.init` and takes precedence over the environment variable, for command-line runs and SLURM jobs alike. To log offline, set `wandb.mode: offline` in the config and upload the runs later with `wandb sync --sync-all` from the repository root.

A *cell* is one point of the sweep grid with every axis except the learning rate fixed; each cell scans its learning rates in ascending order. `--nchunks N` splits the feasible cells into `N` strided chunks and `--chunk i` runs chunk `i`. `full_sweep.yaml` has 438 feasible cells, so `--nchunks 438` runs exactly one cell. Other flags:

- `--dry-run` prints the number of feasible cells and lists the cells in the chunk, without running anything.
- `--outdir DIR` writes results under `DIR` instead of `output/`.
- `--no-wandb` turns off W&B logging.

Results go to `output/<sweep name>/<cell>/lr=<lr>/done.json`. Finished learning rates and cells are skipped on reruns, including by the SLURM sweep. A test run into `output/` therefore means the sweep will skip that cell, so use `--outdir output/test` for throwaway runs.

### Sweeps with SLURM

Submit a whole sweep as a SLURM array job:

```bash
./scripts/submit.sh <SWEEP-NAME>.yaml      # e.g. ./scripts/submit.sh full_sweep.yaml
```

`scripts/submit.sh` reads the `slurm:` block of the config and submits `scripts/sweep.sbatch` with those settings. The defaults (account, GPU type, CPUs, memory, time limit, array) live in `configs/base.yaml`, and a sweep config can override any of them. Each array task gets its own GPU (an H100 MIG slice by default) and runs

```bash
python scripts/run_sweep.py --config <SWEEP-NAME>.yaml --chunk $SLURM_ARRAY_TASK_ID --nchunks $SLURM_ARRAY_TASK_COUNT
```

so the array size is the number of chunks. For one cell per task, set `slurm.array` to `"0-<N-1>%<K>"`, where `N` is the feasible-cell count from `--dry-run` and `K` caps how many tasks run at once. `full_sweep.yaml` uses `"0-437%16"`.

> **Don't export the CPU flags before submitting.** `submit.sh` calls `sbatch --export=ALL`, so any variable exported in your shell (for example `export JAX_PLATFORMS=cpu`) is copied into every job and silently runs it on the CPU. The inline `VAR=value command` form used above is not exported, so it is safe.

Monitor and manage the jobs:

```bash
squeue -u $USER                                                # TRES_PER_N shows the GPU type; N/A means no GPU
grep -E '^jax|FAILED' output/slurm/agop-sweep-<jobid>_0.out   # should show CudaDevice and no FAILED lines
sacct -j <jobid> -X --format=JobID,State,Elapsed               # find tasks that hit TIMEOUT
scancel <jobid>                                                # cancel the whole array
```

Each task logs to `output/slurm/agop-sweep-<jobid>_<task>.out`. To finish tasks that hit the time limit, run the same `submit.sh` command again: completed cells and learning rates are skipped. Resubmit the whole array rather than single task IDs, because the chunk count comes from the array size.

## Specifying the target

The target is a multi-index model $f^*(x) = g(Ux)$ with $U \in \mathbb{R}^{r \times d}$ orthonormal
rows, and $g$ a general multivariate polynomial in the orthonormal Hermite basis
$h_k = He_k/\sqrt{k!}$:

$$g(z) \;=\; \sum_{m} c_m \prod_{j=1}^{r} h_{\alpha^{(m)}_j}(z_j), \qquad \alpha^{(m)} \in \mathbb{N}_0^{\,r}$$

`src/poly.py` evaluates this for any multi-index set $\Lambda = \{\alpha^{(m)}\}$. The config does
not list $\Lambda$ directly, because `data.r` is a sweep axis and a literal matrix has a fixed
number of columns; instead `target.terms` gives a **rule** that regenerates $\Lambda$ for whatever
`r` is. The rule is applied in `build_data` ([src/experiment.py](src/experiment.py), the
`target.terms` block), which emits the `(alpha, coeffs)` pair `poly` consumes.

### Grammar

A **term** maps a *relative* direction offset to a Hermite degree. Three constructs:

| construct | meaning |
|---|---|
| `{0: 3, 1: 2}` | one term, a **product**: $h_3(z_k)\,h_2(z_{k+1})$ |
| `{0: 3}, {0: 2}` as siblings | two terms, a **sum**: $h_3(z_k) + h_2(z_k)$ |
| flat list `[A, B]` | **broadcast**: every direction gets $A + B$ |
| nested list `[[A], [B]]` | **cycle**: direction $k$ gets group $k \bmod n_{\text{groups}}$ |

Offsets wrap, so a term with a nonzero offset couples direction $r-1$ back to $0$.
Coefficients are $c_k \propto k^{-\texttt{target.alpha}}$ normalized to $\|c\|_2 = 1$, with each
group additionally divided by $\sqrt{|\text{group}|}$, so $\mathbb{E}[f^{*2}] = 1$ exactly at every
`r`. `target.alpha = 0` gives constant coefficients.

### Examples (shown at $r = 4$)

**`terms: [{0: 2}]`** — one quadratic link per direction, the canonical single-index-per-direction
target (Ben Arous et al. [arXiv:2508.03688](https://arxiv.org/abs/2508.03688) use exactly this
shape with $\sigma = He_2$):

$$g(z) = c_0 h_2(z_0) + c_1 h_2(z_1) + c_2 h_2(z_2) + c_3 h_2(z_3)$$

**`terms: [[{0: 4}], [{0: 3}], [{0: 2}]]`** — nested, so degrees **differ by direction**, cycling
with period 3:

$$g(z) = c_0 h_4(z_0) + c_1 h_3(z_1) + c_2 h_2(z_2) + c_3 h_4(z_3)
\qquad \alpha = \begin{pmatrix}4&0&0&0\\0&3&0&0\\0&0&2&0\\0&0&0&4\end{pmatrix}$$

**`terms: [[{0: 4}], [{0: 3}, {0: 2}]]`** — nested groups, and group 1 holds **two siblings**, so
odd directions get a *sum* of two degrees on the same coordinate:

$$g(z) = c_0 h_4(z_0) + \tfrac{c_1}{\sqrt2}\big[h_3(z_1) + h_2(z_1)\big]
       + c_2 h_4(z_2) + \tfrac{c_3}{\sqrt2}\big[h_3(z_3) + h_2(z_3)\big]$$

**`terms: [[{0: 4}], [{0: 3, 1: 2}]]`** — group 1 is a **single term with two offsets**, so it is a
*product* across two directions. Note the wraparound in the last row:

$$g(z) = c_0 h_4(z_0) + c_1 h_3(z_1)h_2(z_2) + c_2 h_4(z_2) + c_3 h_3(z_3)h_2(z_0)
\qquad \alpha = \begin{pmatrix}4&0&0&0\\0&3&2&0\\0&0&4&0\\2&0&0&3\end{pmatrix}$$

### Choosing a target

Subspace recovery is governed by the information exponent $p=\min\{|\alpha|:c_\alpha\neq0\}$ --
the *lowest* degree present, not the highest. At $d=128$, $r=4$, `alpha: 1`:

| terms | $p$ | $\max\lvert\alpha\rvert$ | $n \gtrsim d^{p/2}/c_{\min}^2$ |
|---|---|---|---|
| `[{0: 1}, {0: 2}]` | 1 | 2 | 258 |
| `[{0: 1}, {0: 1, 1: 1}]` | 1 | 2 | 258 |
| `[{0: 2}]` | 2 | 2 | 2,916 |
| `[{0: 2}, {0: 4}]` | 2 | 4 | 2,916 |
| `[{0: 4}]` | 4 | 4 | 373,191 |

Adding a high-degree term alongside a low-degree one does not raise $p$: `[{0: 2}, {0: 4}]`
costs the same as `[{0: 2}]`, because the degree-2 part already exposes every direction. Only
the degree-4 term *alone* raises the exponent.

> **Rough guide only.** $d^{p/2}$ is a CSQ *lower* bound (Damian et al.
> [arXiv:2305.10633](https://arxiv.org/abs/2305.10633)) for correlational queries, not a
> guarantee for full-batch GD on a ReLU network. One-pass SGD's upper bound is $d^{p-1}$
> (Ben Arous et al. [arXiv:2003.10409](https://arxiv.org/abs/2003.10409)) and data reuse can
> reach $d\,\mathrm{polylog}\,d$ (Lee et al. [arXiv:2406.01581](https://arxiv.org/abs/2406.01581)) --
> three orders of magnitude apart at $p=4$. Which applies here is an open question, and the
> $1/c_{\min}^2$ factor is a heuristic for per-direction signal strength, not from any theorem.
> Sweep `data.n` rather than trusting these numbers.

## W&B metrics

W&B puts each metric in the section named by its prefix before `/`: it ["organizes panels into sections based on the first prefix of the metric name"](https://docs.coreweave.com/models/track/workspaces). Metrics below follow the order that `log_run_to_wandb` ([src/experiment.py](src/experiment.py)) logs them in, because by default W&B ["does not sort panels by plot title alphabetically"](https://docs.coreweave.com/models/track/workspaces).

**Notation.** $G=\frac1n\sum_i\nabla f(x_i)\nabla f(x_i)^\top$ is the network's learned AGOP and $G^*$ the target's, both $d\times d$ on the training inputs. $M=W^\top W$ is the NFM of the first-layer weights $W\in\mathbb{R}^{m\times d}$. $v_X$, $v^*_X$, $m_X$ are the eigenvectors of $G$, $G^*$, $M$ with the X-th largest eigenvalue ($\lambda_X$, $\lambda^*_X$), and $Q_M\in\mathbb{R}^{d\times r}$ holds the top $r$ eigenvectors of $M$ as columns. $H$ is the training-loss Hessian, and $\eta$ is the absolute learning rate (`lr_absolute` in the run config).

**What X is.** X is an index $0,\dots,r-1$, and W&B draws one panel for each value. In `rayleighX`, `rayleigh_ratioX` and `align_dirX` it indexes the target direction $u_X$ (row X of $U$). In `align_trueX` it indexes the target-AGOP eigenvector $v^*_X$, and in `rank_trueX` the learned-AGOP eigenvector $v_X$.

### `geometry/`

- **`ntk_max_eigval_x2`**: $2\lambda_{\max}(K)$, where $K=\frac1n JJ^\top$ is the empirical NTK. The loss is MSE, so this is exactly the top eigenvalue of the Gauss–Newton Hessian $\frac2n J^\top J$. Its gap to `hessian_max_eigval` is the part of the sharpness that the Gauss–Newton term does not explain.
- **`sharpness_ratio`**: $\eta\,\lambda_{\max}(H)/2$. It crosses 1 exactly at the edge of stability (EOS), $\lambda_{\max}(H)=2/\eta$, and values above 1 mean GD has passed the classical stability limit.
- **`eos_gap`**: $\lambda_{\max}(H)-2/\eta$. It marks the same crossing as `sharpness_ratio`, in eigenvalue units, so a positive value means above EOS.

### `neuron/`

- **`sparsity_layer1` vs `frac_dead`**: both use the ReLU activations $a_k(x_i)$ of the first hidden layer, the only layer logged (hence "layer1"). `sparsity_layer1` $=\frac{1}{mn}\sum_{k,i}\mathbf{1}[a_k(x_i)\le 10^{-8}]$ is the fraction of (neuron, input) pairs that are off. `frac_dead` $=\frac1m\sum_k\mathbf{1}[a_k(x_i)\le 10^{-8}\ \forall i]$ is the fraction of neurons that are off on every training input (dead ReLUs), so `frac_dead` ≤ `sparsity_layer1`.

### `nfm/`

- **`cosine_to_agop` vs `mean_alignment`**: both test the Neural Feature Ansatz $M\propto G$. `cosine_to_agop` $=\langle M,G\rangle_F/(\|M\|_F\|G\|_F)$ compares the whole matrices, weighted by their eigenvalues. `mean_alignment` $=\frac1r\sum_X\langle m_{\pi(X)},v_X\rangle^2$, where $\pi$ is the Hungarian matching of the two top-$r$ eigenbases, compares only eigenvectors and ignores eigenvalues; it is a different metric from `subspace/nfm_mean_alignment`, which compares $M$ to $U$.

### `spectrum/`

- **`agop_cosine`**: $\langle G,G^*\rangle_F/(\|G\|_F\|G^*\|_F)$, which equals 1 iff $G\propto G^*$. `nfm_cosine` is the same quantity with $M$ in place of $G$.
- **`eigvec_score`**: $\frac1r\sum_X\langle v^*_X,v_{\sigma(X)}\rangle^2$, where $\sigma$ is the Hungarian matching of the top-$r$ eigenvectors of $G$ and $G^*$; it is the mean of `nfa/align_trueX`. When target eigenvalues are tied (e.g. `target.alpha: 0` with the same degree in every direction), sampling noise picks $v^*_X$ within span($U$), so this score is noise. Use the `subspace/` metrics for those targets instead.
- **`w1_agop_target` vs `w1_nfm_agop`**: the 1-Wasserstein distance between two top-$r$ spectra, each normalized to a distribution $p_X=\lambda_X/\sum_{X'}\lambda_{X'}$ over the indices, so $W_1(p,q)=\sum_X\big|\sum_{X'\le X}(p_{X'}-q_{X'})\big|$. `w1_agop_target` compares $G$ to $G^*$ and `w1_nfm_agop` compares $M$ to $G$. Both ignore overall scale and are 0 iff the spectral shapes match.
- **`agop_participation_ratio` vs `participation_ratio_ratio`**: `agop_participation_ratio` $=(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$ over all $d$ eigenvalues of $G$, the effective number of directions $G$ uses (exactly $k$ for a flat rank-$k$ spectrum). The `_ratio` suffix, which `stable_rank_ratio` and `intrinsic_dim_ratio` also carry, divides the learned statistic by the same statistic of $G^*$. So `participation_ratio_ratio` is 1 when $G$ has the target's effective rank and above 1 when $G$ spreads over more directions.
- **`rayleighX` vs `rayleigh_ratioX`**: `rayleighX` $=u_X^\top G\,u_X$ is the learned AGOP's energy along target direction X. `rayleigh_ratioX` divides it by $\lambda^*_X$, the X-th largest target eigenvalue. That denominator equals $u_X^\top G^*u_X$, making 1 mean "matched", only when $G^*$ is diagonal in the $U$ basis with eigenvalues decreasing in X; product terms or cycled degrees can break this.

### `subspace/`

- **`nfm_cos_max`**: the largest cosine among the principal angles between the top-$r$ eigenspace of $M$ and span($U$), i.e. how well the best-captured direction is recovered. It has no AGOP counterpart only because `log_run_to_wandb` doesn't log one. `subspace_recovery` ([src/metrics.py](src/metrics.py)) computes `agop_cos_max` and per-direction AGOP alignments as well.
- **`nfm_subspace_dist` vs `agop_subspace_dist`**: the chordal distance $\|\sin\Theta\|_F=\sqrt{r\,(1-a)}$ between span($U$) and the top-$r$ eigenspace of $M$ (`nfm_`) or $G$ (`agop_`), where $a$ is the matching `*_mean_alignment`. It runs from 0 (subspace recovered) to $\sqrt r$ (orthogonal). Comparing the two shows whether the weights or the AGOP find the subspace first.
- **`align_dirX`**: $\|Q_M^\top u_X\|^2\in[0,1]$, the fraction of target direction X that lies in the NFM's top-$r$ eigenspace (it is logged for the NFM only). `nfm_mean_alignment` is its mean over X.

### `nfa/`

- **`align_trueX` vs `rank_trueX`**: both come from the matching $\sigma$ behind `spectrum/eigvec_score`, and "true" means an eigenvector of $G^*$, not a row of $U$. `align_trueX` $=\langle v^*_X,v_{\sigma(X)}\rangle^2$ is keyed by target eigenvector X. Despite its name, `rank_trueX` $=\sigma^{-1}(X)$ is keyed by learned eigenvector X: it is the index of the target eigenvector matched to $v_X$, and it agrees with the intended $\sigma(X)$ unless the matching contains a cycle of length ≥ 3.

## Repository structure

```
├── configs/
│   └── experiment.json
├── data/
│   ├── original/
│   └── processed/
├── figures/
├── notebooks/
├── output/
├── scripts/
│   ├── run.sh
│   └── submit_slurm.sh
└── src/
    ├── executor.py
    ├── experiment.py
    ├── experiment.jl
    └── utils.py
```

- `configs/` — JSON files specifying experiment parameters
- `data/` — Raw data (`original/`) and their polished/organized versions (`processed/`)
- `figures/` — Generated figures
- `notebooks/` — Jupyter notebooks for analysis and visualization
- `output/` — Experiment results (JSON files)
- `scripts/` — Shell scripts for running experiments locally or on SLURM
- `src/` — Source code:
  - `executor.py` — Reads config, selects parameters, and runs the experiment script
  - `experiment.py` / `experiment.jl` — Main experiment logic (Python/Julia)
  - `utils.py` — Plotting utilities for LaTeX-compatible figures

## Running experiments

### Configuration

Experiments are configured via JSON files in `configs/`. Each config file has three sections:
- **Experiment parameters** (top-level): Fixed values (e.g., `"iterations": 10`) or arrays for variables (e.g., `"N": [10, 20]`) that are the main parameters passed to the Python/Julia script
- **Executor settings** (`executor`): Values that specify the name of the Python/Julia script to run, the output directory, and the logging directory
- **SLURM settings**: Resource requirements for execution on compute clusters (see `_comments` in the `slurm` section for parameter descriptions) --- these are simply ignored if the experiment is executed locally.

### How it works

The executor creates a grid of all parameter combinations. Array values in the config
(e.g., `"N": [10, 20]`, `"prob": [0.1, 0.5]`) are combined into 4 experiments (2×2).
Each combination runs as a separate experiment, either sequentially (local) or in
parallel (SLURM array job).

### Local execution

To run all parameter combinations locally:

```bash
./scripts/run.sh -n experiment
```

By default, this runs `configs/experiment.json` using Python.

**Options:**
- `-n, --name NAME` — Experiment name (required). It tells the executor to look for the config `configs/NAME.json` and the python/julia script uses that name as a prefix for output files.
- `-l, --lang LANG` — Language: `python` or `julia` (default: `python`)
- `-h, --help` — Show help message

**Examples:**
```bash
# Use Julia instead of Python
./scripts/run.sh -n experiment -l julia

# Use a different experiment
./scripts/run.sh -n my_other_experiment

# Combine options
./scripts/run.sh -n my_other_experiment -l julia
```

### SLURM execution

To submit experiments as a SLURM array job:

1. Configure SLURM settings in your config file (e.g., `configs/experiment.json`)
2. Submit the job:

```bash
./scripts/submit_slurm.sh
```

**Options** are the same as `run.sh`, plus:
- `-d, --dry-run` — Preview the sbatch command without executing

**Examples:**
```bash
# Preview what would be submitted
./scripts/submit_slurm.sh -n experiment --dry-run
# 💡 This is particularly useful if you do not have permission to execute bash scripts on the login node of your SLURM cluster. Simply run this command locally and copy-paste the resulting sbatch command.

# Submit Julia experiment
./scripts/submit_slurm.sh -n experiment -l julia

# Submit a different experiment
./scripts/submit_slurm.sh -n my_other_experiment
```

Monitor jobs with `squeue -u $USER` and check logs in the `slurm.log_dir` specified in your config (default: `output/slurm_logs`).

### Output

Results are saved to the `output_dir` specified in the config (default: `output/`). Executor logs are saved to `log_dir` (default: `output/logs`). Output filenames are auto-generated from variable parameters:

```
{exp_name}__{param1}={value1}__{param2}={value2}.json
```

For example: `test__N=10__prob=0.5.json`

## Contact & attribution

If you use this code in your research, please cite:

```bibtex
<!-- Insert your paper's BibTeX here -->
```
---