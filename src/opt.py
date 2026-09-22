import sys
import jax
import time

import jax.numpy as jnp
import jax.random as jr

from flax import struct

from functools import partial
from typing import Any, Callable

from metrics import TrainingBar, compute_metrics

class TrainState(struct.PyTreeNode):
  """Simple train state for custom optimizers.

  Args:
    step: Counter starts at 0 and is incremented by every call to
      ``.apply_gradients()``.
    apply_fn: Usually set to ``model.apply()``. Kept in this dataclass for
      convenience to have a shorter params list for the ``train_step()`` function
      in your training loop.
    params: The parameters to be updated and used by ``apply_fn``.
    opt_step: Optimizer step.
  """
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  params: Any
  update_fn: Callable = struct.field(pytree_node=False)

  def apply_gradients(self, *, grads, **kwargs):
      """
      Applies externally-computed gradients using user-provided update_fn.

      update_fn must have signature:
      new_params = update_fn(params, grads, **kwargs)
      """
      new_params = self.update_fn(self.params, grads, **kwargs)

      return self.replace(
          step=self.step + 1,
          params=new_params,
      )
  
  @classmethod
  def create(cls, *, apply_fn, params, update_fn):
      return cls(
          step=0,
          apply_fn=apply_fn,
          params=params,
          update_fn=update_fn,
      )

def train(state, loss_fn, X, y, lr, steps, metrics_config):
    metrics = {}
    traj = []

    pbar = TrainingBar(steps)

    for t in pbar:
        traj.append(state.params)

        t0 = time.perf_counter()
        state, metric_dict = train_step(state, loss_fn, (X, y), lr)
        loss_val = metric_dict['loss']
        metrics.setdefault('loss', []).append(loss_val)
        t_update = time.perf_counter() - t0

        step_metrics = compute_metrics(
            state=state, 
            metrics_config=metrics_config,
            activations=metric_dict['activations']
        )

        for name, result in step_metrics.items():
            for k, v in result.items():
                metrics.setdefault(name, {}).setdefault(k, []).append(v)

        pbar.update(
            tqdm_info={"loss": float(loss_val), "update": f"{t_update:.3f}s"},
            aux_metrics={k: f"{v['time']:.3f}s" for k, v in step_metrics.items()}
        )
        
    traj.append(state.params)
    return state, traj, metrics

# def gd_update(params, grads, lr):
#     return jax.tree_util.tree_map(
#         lambda p, g: p - lr * g,
#         params,
#         grads
#     )

def gd_update(params, grads, lr, parameterization="standard"):
    new_params = []
    L = len(params)

    for layer_idx, (p, g) in enumerate(zip(params, grads)):
        W, b = p["weights"], p["bias"]
        gW, gb = g["weights"], g["bias"]

        fan_in, fan_out = W.shape

        if parameterization == "standard":
            lr_scale = 1.0

        elif parameterization == "ntk":
            lr_scale = 1.0 / fan_in

        elif parameterization == "mup":
            if layer_idx == 0:
                # input layer: d -> m
                lr_scale = fan_out / fan_in
            elif layer_idx == L - 1:
                # output layer: m -> 1
                lr_scale = 1.0 / fan_in
            else:
                # hidden layer: m -> m
                lr_scale = 1.0

        elif parameterization == "spectral":
            lr_scale = fan_out / fan_in

        else:
            raise ValueError(
                f"Unknown parameterization: {parameterization}"
            )
        
        new_params.append({
            "weights": W - lr * lr_scale * gW,
            "bias": b - lr * gb
        })

    return new_params

def batch_loss(apply_fn, loss_fn, batch):
    inputs, targets = batch

    def compute_loss(params):
        outputs, acts = jax.vmap(
            lambda x: apply_fn(params, x, return_activations=True)
        )(inputs)
        loss = loss_fn(outputs, targets).mean()
        return loss, acts  # acts as aux
    return compute_loss

@partial(jax.jit, static_argnums=(1,))
def train_step(state, loss_fn, batch, lr):
    (loss, acts), grads = jax.value_and_grad(
        batch_loss(state.apply_fn, loss_fn, batch),
        has_aux=True
    )(state.params)

    state = state.apply_gradients(grads=grads, lr=lr)
    return state, {'loss': loss, 'activations': acts}