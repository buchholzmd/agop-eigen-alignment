import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_directional_slices(
    fns,
    fn_labels,
    directions,
    dir_labels,
    t_min,
    t_max,
    steps,
    start=0,
    num_dirs=None,
    figsize=None
):  
    # Get input time steps
    t = jnp.linspace(t_min, t_max, steps)

    if num_dirs is None:
        num_dirs = min([dirs.shape[1] for dirs in directions])

    handles = []

    if figsize is None:
        figsize = (3*num_dirs, 7*len(directions))

    fig, axs = plt.subplots(num_dirs, len(directions), squeeze=False, figsize=figsize)
    for i in range(num_dirs):
        idx = start + i

        for j, (dir_label, dirs) in enumerate(zip(dir_labels, directions)):
            ax = axs[i][j]

            # pick direction
            v = dirs[:, idx]

            X = t[:, None] * v[None, :]

            for fn_label, fn in zip(fn_labels, fns):
                y = fn(X)
                line, = ax.plot(t, y)

                if len(handles) != len(fn_labels):
                    handles.append(line)

            ax.set_title(f"{dir_label}[:,{idx}]")
            ax.legend()

    if handles is not None:
        fig.legend(handles, fn_labels, loc="upper right", ncol=len(fn_labels))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axs
