import jax.random as jr

class RNGKey:
    def __init__(self, seed):
        self.key = jr.PRNGKey(seed)
    
    def next(self):
        self.key, subkey = jr.split(self.key)
        return subkey