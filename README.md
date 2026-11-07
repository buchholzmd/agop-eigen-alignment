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

**Notation.** $f$ is the network and $f^*(x)=g(Ux)$ the target, with $u_X$ the X-th row of $U\in\mathbb{R}^{r\times d}$ (see [Specifying the target](#specifying-the-target)). $G=\frac1n\sum_i\nabla f(x_i)\nabla f(x_i)^\top$ is the network's learned AGOP and $G^*$ the target's, both $d\times d$ on the training inputs. $M=W^\top W$ is the NFM of the first-layer weights $W\in\mathbb{R}^{m\times d}$. $v_X$, $v^*_X$, $m_X$ are the eigenvectors of $G$, $G^*$, $M$ with the X-th largest eigenvalue ($\lambda_X$, $\lambda^*_X$), and $Q_G,Q_M\in\mathbb{R}^{d\times r}$ hold the top $r$ eigenvectors of $G$ and $M$ as columns. $H$ is the training-loss Hessian, and $\eta$ is the absolute learning rate (`lr_absolute` in the run config).

The loss responds to the square. This is my own derivation, not from the literature. Suppose the network has the right linear link $g(z)=a^\top z$ but the wrong subspace. Then the excess loss is $\mathbb{E}[(a^\top U(I-P_Q)x)^2]=\|(I-P_Q)U^\top a\|^2$, which is quadratic in the subspace error, and the same holds to leading order for smooth links. So if loss $\propto$ dist², fitting the unsquared distance gives half the loss exponent. aggregate.py fits both loss_min and final_nfm_subspace_dist against width, so that's where the choice shows up.

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

Every sweep runs as a [W&B Sweep](https://docs.coreweave.com/models/sweeps/sweep-config-keys) through one script, `scripts/run_sweep.py`. A *cell* is every sweep axis except the learning rate, and a *point* is one (cell, lr) pair; each point becomes one W&B run. W&B "creates a parallel coordinates plot, a parameter importance plot, and a scatter plot when you start a sweep job" ([Visualize sweep results](https://docs.coreweave.com/models/sweeps/visualize-sweep-results)), and they rank runs by `wandb.sweep_metric`, which defaults to minimizing `val_loss`.

```bash
python scripts/run_sweep.py --config <SWEEP-NAME>.yaml --dry-run       # show the W&B sweep config and its size
python scripts/run_sweep.py --config <SWEEP-NAME>.yaml --create        # create the sweep; prints its ID last
python scripts/run_sweep.py --config <SWEEP-NAME>.yaml --sweep-id ID   # run points until the grid is done
```

With `--sweep-id`, `--count N` stops after `N` points. The sweep has one parameter per `sweep:` axis plus `optim.lr` from `lr_scan`, and W&B's grid serves every point:

- **No early exit over learning rates.** Every lr of every cell runs, including those past divergence, so `full_sweep.yaml` is 39,780 points.
- **Infeasible cells still get points.** A point that fails the `limits:` checks closes at once without training, tagged `infeasible`, with the reason in its `infeasible` summary field. That is 17,442 of `full_sweep.yaml`'s points; filter them out by the tag.
- **List-valued axes are swept by index.** `target.terms` appears in the sweep as `target.terms_index`, and the run config still holds the real value under `target.terms`.
- **Order and retries.** W&B's [open-source grid code](https://github.com/wandb/sweeps/blob/master/src/sweeps/grid_search.py) takes the product over parameter names in sorted order, with the last varying fastest. It counts every run it is given as done, whatever its state, so expect a point killed by the time limit not to be served again.
- **`job_type` is `sweep`.** A point's run is created before W&B hands over the point, and `job_type` cannot change afterwards, so it is the config's `wandb.job_type` rather than the parameterization. The parameterization is in the run's tags and config.

> **Note:** Setting `WANDB_MODE=offline` has no effect in this repo. The W&B mode comes from `wandb.mode` in the config, which is passed directly to `wandb.init` and takes precedence over the environment variable. To log offline, set `wandb.mode: offline` in the config and upload the runs later with `wandb sync --sync-all` from the repository root. The sweep itself still needs W&B's server to hand out points.

### From the command line

For a quick test, run one point on a login node. Login nodes have no GPUs, so force JAX onto the CPU:

```bash
source ~/jax_env/bin/activate
python scripts/run_sweep.py --config dataset_sweep.yaml --create      # note the ID it prints
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python scripts/run_sweep.py --config dataset_sweep.yaml --sweep-id <ID> --count 1
```

The test run joins the config's W&B group like any other, so delete the test sweep and its runs afterwards.

### Sweeps with SLURM

```bash
./scripts/submit.sh <SWEEP-NAME>.yaml              # create a sweep and start SLURM tasks on it
./scripts/submit.sh <SWEEP-NAME>.yaml <SWEEP-ID>   # more tasks on an existing sweep
```

`scripts/submit.sh` runs `run_sweep.py --create` unless it is given a sweep ID, then submits `scripts/sweep.sbatch` with the config's `slurm:` settings. The defaults (account, GPU type, CPUs, memory, time limit, array) live in `configs/base.yaml`, and a sweep config can override any of them. Each array task gets its own GPU (an H100 MIG slice by default) and runs `run_sweep.py --sweep-id <SWEEP-ID>`, taking points until the grid is exhausted or the time limit ends the task. `slurm.array` is `"0-<N-1>%<K>"`: `N` tasks in total, at most `K` at once.

> **Don't export the CPU flags before submitting.** `submit.sh` calls `sbatch --export=ALL`, so any variable exported in your shell (for example `export JAX_PLATFORMS=cpu`) is copied into every job and silently runs it on the CPU. The inline `VAR=value command` form used above is not exported, so it is safe.

Monitor and manage the jobs:

```bash
squeue -u $USER                                                  # TRES_PER_N shows the GPU type; N/A means no GPU
grep -E '^jax|Traceback' output/slurm/agop-sweep-<jobid>_0.out   # should show CudaDevice and no Traceback
sacct -j <jobid> -X --format=JobID,State,Elapsed                 # find tasks that hit TIMEOUT
scancel <jobid>                                                  # cancel the whole array
```

Each task logs to `output/slurm/agop-sweep-<jobid>_<task>.out`. To keep going after tasks hit the time limit, pass the sweep ID that `submit.sh` printed: `./scripts/submit.sh <SWEEP-NAME>.yaml <SWEEP-ID>`.

## Specifying the target

The target is a multi-index model $f^*(x) = g(Ux)$ with $U \in \mathbb{R}^{r \times d}$ orthonormal
rows, and $g$ a general multivariate polynomial in the orthonormal Hermite basis
$h_k = He_k/\sqrt{k!}$:

$$g(z) \;=\; \sum_{m} c_m \prod_{j=1}^{r} h_{\alpha^{(m)}_j}(z_j), \qquad \alpha^{(m)} \in \mathbb{N}_0^{\,r}$$

`src/poly.py` evaluates this for any multi-index set $\Lambda = \{\alpha^{(m)}\}$. Because `data.r` is a sweep axis and the matrix form of $\Lambda$ would therefore change. `target.terms` gives a **rule** that regenerates $\Lambda$ for whatever `r` is. The rule is applied in `build_data` of [src/experiment.py](src/experiment.py).

### Configuring degree by term and coordinate 

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

**`terms: [{0: 2}]`** — one quadratic link per direction i.e.  $\sigma = He_2$:

$$g(z) = c_0 h_2(z_0) + c_1 h_2(z_1) + c_2 h_2(z_2) + c_3 h_2(z_3)$$

**`terms: [[{0: 4}], [{0: 3}], [{0: 2}]]`** — degrees **differ by direction**, cycling
with period 3:

$$g(z) = c_0 h_4(z_0) + c_1 h_3(z_1) + c_2 h_2(z_2) + c_3 h_4(z_3)
\qquad \alpha = \begin{pmatrix}4&0&0&0\\0&3&0&0\\0&0&2&0\\0&0&0&4\end{pmatrix}$$

**`terms: [[{0: 4}], [{0: 3}, {0: 2}]]`** — odd directions get a *sum* of two degrees on the same coordinate:

$$g(z) = c_0 h_4(z_0) + \tfrac{c_1}{\sqrt2}\big[h_3(z_1) + h_2(z_1)\big]
       + c_2 h_4(z_2) + \tfrac{c_3}{\sqrt2}\big[h_3(z_3) + h_2(z_3)\big]$$

**`terms: [[{0: 4}], [{0: 3, 1: 2}]]`** — group 1 is a **single term with two offsets**, a
*product* across two directions with a wraparound in the last row:

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

A low-degree term masks a high-degree one: `[{0:2},{0:4}]` costs the same as `[{0:2}]`.

> **Rough guide only.** $d^{p/2}$ is a CSQ lower bound (Damian et al.), not a guarantee for GD on a ReLU net — SGD's upper bound is $d^{p-1}$ (Ben Arous et al.) and reuse can reach $d\,\mathrm{polylog}\,d$ (Lee et al.). Sweep `data.n` instead.

## W&B metrics

W&B organizes panels into sections based on the first prefix of the metric name and metrics below follow the order that `log_run_to_wandb` of [src/experiment.py](src/experiment.py) logs them in. Sections are split so that none logs a function of another section's metrics: `nfa/` compares whole matrices, `spectrum/` eigenvalues, and `subspace/` eigenvectors and eigenspaces.

**Definitions.** For $d\times d$ PSD matrices $A,B$ with eigenvalues $\lambda_1(A)\ge\lambda_2(A)\ge\dots$:

- **Cosine**: $\cos(A,B)=\langle A,B\rangle_F/(\|A\|_F\|B\|_F)$, which is 1 iff $A\propto B$ and weights directions by their eigenvalues.
- **Effective ranks**: the stable rank $\operatorname{srank}(A)=\sum_i\lambda_i^2/\lambda_1^2$, intrinsic dimension $\operatorname{idim}(A)=\sum_i\lambda_i/\lambda_1$ and participation ratio $\operatorname{PR}(A)=(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$. Always $\operatorname{srank}\le\operatorname{idim}\le\operatorname{PR}$, with all three equal to $k$ on a flat rank-$k$ spectrum: srank is dominated by the top eigenvalue, idim gives the tail more weight, and PR counts the directions with comparable eigenvalues.
- **Wasserstein-1**: $W_1(A,B)=\sum_X\big|\sum_{X'\le X}(p_{X'}-q_{X'})\big|$, where $p_X=\lambda_X(A)/\sum_{X'}\lambda_{X'}(A)$ normalizes the top-$r$ spectrum of $A$ and $q$ does the same for $B$. It ignores overall scale and is 0 iff the spectral shapes match.
- **Matching**: for orthonormal $r$-frames $(a_X)$ and $(b_X)$, $\sigma_{a,b}$ is the permutation maximizing $\sum_X\langle a_{\sigma(X)},b_X\rangle^2$ (Hungarian algorithm), pairing each $b_X$ with $a_{\sigma(X)}$.
- **Principal angles**: $\theta_1\le\dots\le\theta_r$ between a top-$r$ eigenspace and span($U$), with $\cos\theta_X$ the singular values of $Q^\top U^\top$. The prefix `nfm_` or `agop_` says whether $Q$ is $Q_M$ or $Q_G$.
- **`_ratio` suffix**: the learned statistic divided by the same statistic of the target AGOP $G^*$.

Metrics without a prefix:

- **`loss`**: the training MSE $\frac1n\sum_i(f(x_i)-y_i)^2$, evaluated at the same parameters as every other metric at that step.
- **`val_loss`**: the same MSE on `data.val_n` held-out inputs with noiseless labels $f^*(x)$, so it reaches 0 only if $f$ recovers $f^*$.

### `geometry/`

- **`hessian_max_eigval`**: $\lambda_{\max}(H)$, the sharpness, from `metrics.num_power_iters` steps of power iteration.
- **`ntk_max_eigval_x2`**: $2\lambda_{\max}(K)$, where $K=\frac1n JJ^\top$ is the empirical NTK. The loss is MSE, so this is exactly the top eigenvalue of the Gauss–Newton Hessian $\frac2n J^\top J$. Its gap to `hessian_max_eigval` is the part of the sharpness that the Gauss–Newton term does not explain.
- **`sharpness_ratio`**: $\eta\,\lambda_{\max}(H)/2$, which crosses 1 exactly at the edge of stability (EOS), $\lambda_{\max}(H)=2/\eta$; above 1, GD has passed the classical stability limit. Unlike `eos_gap`, it has no units, so runs with different learning rates share one scale.
- **`eos_threshold`**: $2/\eta$, constant within a run: the EOS line that `hessian_max_eigval` crosses.
- **`eos_gap`**: $\lambda_{\max}(H)-2/\eta$, positive above EOS. It marks the same crossing as `sharpness_ratio`, but in eigenvalue units.

### `neuron/`

All six use the activations $A\in\mathbb{R}^{n\times m}$, $A_{ik}=a_k(x_i)$, of the first hidden layer (the only one logged). The ranks are those of the (uncentered) activation covariance $C=\frac1n A^\top A$, exact when $\min(n,m)\le2048$ and Hutchinson-estimated otherwise.

- **`cov_max_eigval`**: $\lambda_{\max}(C)$.
- **`intrinsic_dim`**: $\operatorname{idim}(C)$.
- **`stable_rank`**: $\operatorname{srank}(C)$.
- **`participation_ratio`**: $\operatorname{PR}(C)$.
- **`sparsity_layer1`**: $\frac{1}{mn}\sum_{k,i}\mathbf{1}[a_k(x_i)\le10^{-8}]$, the fraction of (neuron, input) pairs that are off. Unlike `frac_dead`, it also counts neurons that are off on only some inputs, so `frac_dead` ≤ `sparsity_layer1`.
- **`frac_dead`**: $\frac1m\sum_k\mathbf{1}[a_k(x_i)\le10^{-8}\ \forall i]$, the fraction of neurons off on every training input (dead ReLUs).

### `nfa/`

Whole-matrix comparisons of $M$, $G$ and $G^*$. The ranks of $G$ use all $d$ of its eigenvalues ($G^*$ has rank at most $r$).

- **`nfa_cosine`**: $\cos(M,G)$, the Neural Feature Ansatz $M\propto G$. Unlike `subspace/mean_matched_alignment`, it weights directions by their eigenvalues.
- **`agop_cosine`**: $\cos(G,G^*)$, which is 1 iff the learned AGOP is proportional to the target's.
- **`nfm_cosine`**: $\cos(M,G^*)$, comparing the NFM directly to the target AGOP.
- **`agop_nuclear`**: $\|G\|_*=\operatorname{tr}G=\frac1n\sum_i\|\nabla f(x_i)\|^2$, the network's mean squared input-gradient norm.
- **`nuclear_ratio`**: $\operatorname{tr}G/\operatorname{tr}G^*$, which is 1 when the network's gradient energy matches the target's.
- **`agop_stable_rank`**: $\operatorname{srank}(G)$.
- **`agop_intrinsic_dim`**: $\operatorname{idim}(G)$.
- **`agop_participation_ratio`**: $\operatorname{PR}(G)$, the effective number of directions $G$ uses. Unlike `participation_ratio_ratio`, it is an absolute count.
- **`stable_rank_ratio`**: $\operatorname{srank}(G)/\operatorname{srank}(G^*)$.
- **`intrinsic_dim_ratio`**: $\operatorname{idim}(G)/\operatorname{idim}(G^*)$.
- **`participation_ratio_ratio`**: $\operatorname{PR}(G)/\operatorname{PR}(G^*)$, which is 1 when $G$ has the target's effective rank and above 1 when $G$ spreads over more directions.

### `spectrum/`

- **`wasserstein1_agop_target`**: $W_1(G,G^*)$. Unlike `wasserstein1_nfm_agop`, it measures progress toward the target.
- **`wasserstein1_nfm_agop`**: $W_1(M,G)$, a spectral NFA test that needs no target.
- **`rayleighX`**: $u_X^\top G\,u_X$, the learned AGOP's energy along target direction X; X indexes the rows $u_X$ of $U$. Unlike `rayleigh_ratioX`, it is not normalized by the target.
- **`rayleigh_ratioX`**: `rayleighX` divided by $\lambda^*_X$; X indexes the target direction $u_X$ in the numerator and the X-th largest target eigenvalue in the denominator. That denominator equals $u_X^\top G^*u_X$, making 1 mean "matched", only when $G^*$ is diagonal in the $U$ basis with eigenvalues decreasing in X; product terms or cycled degrees can break this.

### `subspace/`

- **`nfm_mean_alignment`**: $\frac1r\sum_X\cos^2\theta_X$, the mean of `align_dirX`, which is 1 iff the subspace is recovered.
- **`nfm_cos_max`**: $\cos\theta_1$, i.e. how well the best-captured direction is recovered. It has no AGOP counterpart only because `log_run_to_wandb` doesn't log one: `subspace_recovery` ([src/metrics.py](src/metrics.py)) computes `agop_cos_max` and per-direction AGOP alignments as well.
- **`nfm_sin_max`**: $\sin\theta_r$, i.e. how badly the worst-captured direction is missed, which is 0 iff the subspace is fully recovered.
- **`nfm_subspace_dist_sq`**: the squared chordal distance $\|\sin\Theta\|_F^2=r(1-a)$, with $a$ = `nfm_mean_alignment`, from 0 (recovered) to $r$ (orthogonal). It is squared so that its power-law exponent compares directly with the loss's. Compared with `agop_subspace_dist_sq`, it shows whether the weights or the AGOP find the subspace first.
- **`agop_mean_alignment`**: `nfm_mean_alignment` for $Q_G$.
- **`agop_sin_max`**: `nfm_sin_max` for $Q_G$.
- **`agop_subspace_dist_sq`**: `nfm_subspace_dist_sq` for $Q_G$. Compared with `nfm_subspace_dist_sq`, it shows whether the AGOP or the weights find the subspace first.
- **`mean_matched_alignment`**: $\frac1r\sum_X\langle m_{\sigma(X)},v_X\rangle^2$ with $\sigma=\sigma_{m,v}$, the Neural Feature Ansatz at the level of eigenvectors. Unlike `nfa/nfa_cosine`, it ignores eigenvalues, and unlike `nfm_mean_alignment`, it compares $M$ to $G$ rather than to $U$.
- **`mean_align_true`**: the mean of `align_trueX`. When target eigenvalues are tied (e.g. `target.alpha: 0` with the same degree in every direction), sampling noise picks $v^*_X$ within span($U$), so this is noise; use the principal-angle metrics for those targets instead.
- **`align_dirX`**: $\|Q_M^\top u_X\|^2\in[0,1]$, the fraction of target direction X inside the NFM's top-$r$ eigenspace; X indexes the rows $u_X$ of $U$. It is logged for the NFM only.
- **`align_trueX`**: $\langle v_{\sigma(X)},v^*_X\rangle^2$ with $\sigma=\sigma_{v,v^*}$; X indexes the target-AGOP eigenvectors $v^*_X$ ("true" means an eigenvector of $G^*$, not a row of $U$). Unlike `rank_trueX`, it says how well $v^*_X$ is matched, not by which learned eigenvector.
- **`rank_trueX`**: $\sigma(X)$, the index of the learned-AGOP eigenvector matched to $v^*_X$; X indexes $v^*_X$. It reads X when $G$ orders the directions as $G^*$ does. Runs logged under the old key `nfa/rank_trueX` hold the inverse permutation instead.

### `slices/`

- **`directional`**: an image logged once after training, with one panel per target direction $u_X$ showing $f^*(t\,u_X)$ and $f(t\,u_X)$ for $t\in[-3,3]$.

### Run summary

These are single numbers per run, set in `wandb.summary` by `run_summary` ([src/metrics.py](src/metrics.py)) rather than logged at every step.

- **`regime`**: `monotonic`, `catapult` (the loss rises, then ends below its value before the rise) or `diverged` (the loss is non-finite, or its final value exceeds 100× its minimum).
- **`loss_final`**: `loss` at the last step.
- **`loss_min`**: the minimum of `loss` over training.
- **`loss_exponent`**: $\beta$ in $\text{loss}\approx C\,t^\beta$, fit over the longest log-log window of at least 5 steps with $R^2\ge0.98$; NaN if no window qualifies.
- **`loss_exponent_r2`**: the $R^2$ of that fit.
- **`loss_fit_lo`**: the first step of the fit window, counting from 1 (W&B step + 1).
- **`loss_fit_hi`**: the last step of the fit window, counting from 1.
- **`sharpness_ratio_max`**: the maximum of `geometry/sharpness_ratio` over training.
- **`eos_first_step`**: the first step with `sharpness_ratio` ≥ 1, or −1 if the run never reaches EOS.
- **`eos_fraction`**: the fraction of steps with `sharpness_ratio` ≥ 1.
- **`infeasible`**: on sweep runs tagged `infeasible` only, the `limits:` check the cell failed. These runs have no other metrics.
- **`final_<metric>`**: the last-step value of `subspace/{nfm,agop}_{mean_alignment,subspace_dist_sq}`, `subspace/mean_align_true` and `nfa/{agop_cosine,nuclear_ratio,intrinsic_dim_ratio,stable_rank_ratio}`.

### Aggregate run

`scripts/aggregate.py` logs a separate run, `<sweep name>_scaling` with job type `aggregate`, that fits $y\approx C\,\texttt{width}^\beta$ per parameterization. Each $y$ is first reduced over the learning-rate grid (min, or max for `final_nuclear_ratio`), with diverged runs dropped.

- **`scaling/exponents`**: a table of every fit: $\beta$, $R^2$ and the width range it covers.
- **`scaling/<y>_vs_model.width`**: the log-log plot of each fit, where $y$ is `loss_min`, `final_nfm_subspace_dist_sq` or `final_nuclear_ratio`.
- **`beta__<y>__<parameterization>`**: each fitted $\beta$ as a summary scalar.

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