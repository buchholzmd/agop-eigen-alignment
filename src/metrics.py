import time
import ipywidgets as widgets

from tqdm import tqdm
from IPython.display import display, clear_output, HTML

class TrainingBar:
    def __init__(self, steps):
        self.tqdm_bar = tqdm(range(steps))
        display(HTML("""<style>
            .dark-output { background: #1e1e1e !important; }
            .dark-output .widget-box { background: #1e1e1e !important; }
            .dark-output .jp-OutputArea-output { background: #1e1e1e !important; }
        </style>"""))
        self.html_widget = widgets.HTML(value=self._render({}))
        box = widgets.Box([self.html_widget], layout=widgets.Layout(
            background='#1e1e1e',
            padding='8px',
            width='100%',
        ))
        box.add_class('dark-output')
        display(box)

    def __iter__(self):
        return iter(self.tqdm_bar)

    def _render(self, info: dict):
        rows = "".join(f"<tr><td style='color:#d4d4d4; font-family:monospace; padding:2px 12px 2px 4px'>{k}</td><td style='color:#d4d4d4; font-family:monospace; padding:2px 4px'>{v}</td></tr>" for k, v in info.items())
        return f"<table style='background:#1e1e1e; width:100%; border-collapse:collapse'>{rows}</table>"

    def update(self, tqdm_info: dict, aux_metrics: dict):
        self.tqdm_bar.set_postfix(tqdm_info)
        self.html_widget.value = self._render(aux_metrics)

def compute_metrics(*, state, metrics_config, **kwargs):
    metrics = {}
    results = {}

    for name, cfg in metrics_config.items():
        fn = cfg["fn"]
        args = cfg.get("args", ())

        t0 = time.perf_counter()
        out = fn(state, *args, **results, **kwargs)
        dt = time.perf_counter() - t0

        # capture output to be used by other metrics
        results[name] = out

        if isinstance(out, dict):
            metrics[name] = {**out, "time": dt}
        else:
            metrics[name] = {"value": out, "time": dt}

    return metrics