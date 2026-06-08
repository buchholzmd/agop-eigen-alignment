import jax.nn as nnx
import jax.numpy as jnp
import jax.random as jr

from jax import vmap
from utils import RNGKey

# def relu(x):
#     return jnp.maximum(0, x)

class MLP:
    def __init__(self, sizes, params=None, activation=nnx.relu, parameterization='standard', rng=RNGKey(33)):
        self.rng = rng
        self.parameterization = parameterization
        self.activation = activation

        if params is None:
            self._init_params(sizes)
        else:
            self.params = params

    def _init_params(self, sizes):
        self.params = []
        keys = jr.split(self.rng.next(), len(sizes)-1)

        L = len(sizes) - 1

        for layer_idx, (k, (fan_in, fan_out)) in enumerate(
            zip(keys, zip(sizes[:-1], sizes[1:]))
        ):

            w_key, b_key = jr.split(k)

            if self.parameterization in ['standard', 'ntk']:
                W = jr.normal(w_key, (fan_out, fan_in)) / jnp.sqrt(fan_in)
                b = jr.normal(b_key, (fan_out,)) / jnp.sqrt(fan_in)
            
            elif self.parameterization == 'mup':
                # input layer
                if layer_idx == 0:
                    W = jr.normal(w_key, (fan_out, fan_in)) / jnp.sqrt(fan_in)
                    b = jr.normal(b_key, (fan_out,)) / jnp.sqrt(fan_in)
        
                # output layer
                elif layer_idx == L - 1:
                    W = jr.normal(w_key, (fan_out, fan_in)) / fan_in
                    b = jr.normal(b_key, (fan_out,)) / fan_in

                # hidden layers
                else:
                    W = jr.normal(w_key, (fan_out, fan_in)) / jnp.sqrt(fan_in)
                    b = jr.normal(b_key, (fan_out,)) / jnp.sqrt(fan_in)
            
            else:
                raise ValueError(f"Unknown parameterization: {self.parameterization}")
            
            self.params.append({
                "weights": W,
                "bias": b
            })

        return self.params

    def _set_params(self, params):
        self.params = params

    def apply_fn(self, params, x, init_layer=0, return_activations=False):
        activations = [x]
        
        for i, layer in enumerate(params[init_layer:-1], init_layer):
            w, b = layer["weights"], layer["bias"]
            x = self.activation(x @ w.T + b)
            # if self.parameterization == 'ntk':
            #     x = x / jnp.sqrt(w.shape[0])
            # elif self.parameterization == 'mup':
            #     x = x / jnp.sqrt(w.shape[1])
            
            if return_activations:
                activations.append(x)
 
        w, b = params[-1]["weights"], params[-1]["bias"]
        
        if return_activations:
            return (x @ w.T + b).squeeze(), tuple(activations)
        return (x @ w.T + b).squeeze()

    def predict(self, X):
        return vmap(self.apply_fn, in_axes=(None, 0))(self.params, X)