#!/usr/bin/env python
"""Build an interactive dashboard and videos from finished W&B sweeps. Read-only.

configs/dashboard.yaml says what to show. This script only READS runs that already exist:
sweep and run listings, and each run's config, logged keys and history. It caches them per
run and builds everything locally. It never starts a sweep, an agent or any training. The
one thing it can write to W&B is --upload, which logs the finished pages and videos to a
single run named sweep-dashboard.

    python scripts/dashboard.py                      # pull (cached), build pages + videos
    python scripts/dashboard.py --no-videos          # pages only
    python scripts/dashboard.py --refresh            # re-download every run
    python scripts/dashboard.py --list [TEXT]        # list the project's sweeps and groups, then exit
    python scripts/dashboard.py --project ENT/PROJ   # read another project than the spec's
    python scripts/dashboard.py --upload             # also log the pages and videos to W&B

Output, in output/dashboard/:
    dashboard.html               every section, with its videos
    dashboard_<section>.html     one section per page, for W&B's HTML panel
    videos/<name>.mp4            rows = metrics, cols = one sweep axis, frames = another

Every page is self-contained: its data, libraries (uPlot, marked, MathJax) and videos are
inlined. A section's control bar is the run set for all of its panels: one axis colors the
curves, and every other axis is filtered to one value or to "all".
"""
import argparse, base64, hashlib, json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

from config import load_config, get_in

# Pinned by content: a truncated or altered download is never inlined. The full MathJax
# bundle, because an inlined MathJax cannot autoload TeX extensions.
VENDOR = {
    "uplot.js":   ("https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.iife.min.js",
                   "7f29cd14e36d84141ab49c5e0a806d88e4d855596a9bc0e10799bc2c114b4c4f"),
    "uplot.css":  ("https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.min.css",
                   "df630c6a8d6f8eeaff264b50f73ce5b114f646ffd9a0bb74f049b0a00135fa04"),
    "marked.js":  ("https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js",
                   "15fabce5b65898b32b03f5ed25e9f891a729ad4c0d6d877110a7744aa847a894"),
    "mathjax.js": ("https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg-full.js",
                   "a4354ff94fd868aea0cc6eaaa79a57fda0588646fc46ee3700a349ee0a11cbe6"),
}
# Pages keep, for each curve, the min and max of each of BUCKETS step ranges, each at the
# step where it occurs (plus the first and last steps and the edges of non-finite stretches).
BUCKETS = 64
TERMINAL = {"finished", "crashed", "failed", "killed"}
REGIME_COLORS = {"monotonic": "#5aa9e6", "catapult": "#f5a623", "diverged": "#e5534b"}


# ----------------------------------------------------------------------------- W&B

def project_path(arg, spec, base):
    """ENTITY, PROJECT from --project (ENTITY/PROJECT or PROJECT), else the spec's wandb:
    block, else configs/base.yaml."""
    w = spec.get("wandb") or {}
    entity = w.get("entity", get_in(base, "wandb.entity"))
    project = w.get("project", get_in(base, "wandb.project"))
    if arg:
        entity, project = arg.split("/", 1) if "/" in arg else (entity, arg)
    return entity, project


def project_sweeps(api, entity, project):
    """-> {sweep id: Sweep} for the project, or a clear error when the project does not exist."""
    try:
        return {s.id: s for s in api.project(project, entity=entity).sweeps()}
    except ValueError as e:
        raise SystemExit(f"W&B project {entity}/{project}: {e} (set --project or the spec's wandb: block)")


def sweep_index(api, entity, project, filters=None):
    """-> {sweep id or None: {runs, groups: {group: runs}, latest}} from one lightweight pass
    over the matching runs; None collects the runs that belong to no sweep."""
    out = {}
    for r in api.runs(f"{entity}/{project}", filters=filters or {}, per_page=1000, include_sweeps=False):
        e = out.setdefault(r.sweep_name, {"runs": 0, "groups": {}, "latest": ""})
        e["runs"] += 1
        e["groups"][r.group] = e["groups"].get(r.group, 0) + 1
        e["latest"] = max(e["latest"], r.created_at or "")
    return out


def list_sweeps(api, entity, project, text):
    """Print the project's sweeps, and its runs outside any sweep, with state, runs and groups."""
    sweeps = project_sweeps(api, entity, project)
    index = sweep_index(api, entity, project)
    rows = []
    for i in set(index) | set(sweeps):
        e = index.get(i, {"runs": 0, "groups": {}, "latest": ""})
        s = sweeps.get(i)
        name = s.name if s else ("(no sweep)" if i is None else "(deleted sweep)")
        groups = ", ".join(f"{g or '(no group)'} ({n})"
                           for g, n in sorted(e["groups"].items(), key=lambda gn: -gn[1]))
        row = [name, i or "-", s.state if s else "-", str(e["runs"]),
               e["latest"][:16].replace("T", " "), groups]
        if text.lower() in " ".join(row).lower():
            rows.append(row)
    rows.sort(key=lambda r: r[4], reverse=True)
    header = ["sweep", "id", "state", "runs", "newest run", "groups (runs)"]
    widths = [max(len(r[c]) for r in rows + [header]) for c in range(5)]
    for r in [header] + rows:
        print("  ".join(r[c].ljust(widths[c]) for c in range(5)) + "  " + r[5])
    print(f"{len(rows)} rows in {entity}/{project}" + (f" matching {text!r}" if text else ""))


def resolve_sweeps(api, entity, project, source):
    """The sweeps behind a section or video: its `sweep` (a name or an id), its `group`
    (every sweep with runs in that W&B group), or both. Several matches use the sweep with
    the newest run, unless the spec says `sweeps: all`."""
    want, group = source.get("sweep"), source.get("group")
    label = " and ".join(f"{k} {source[k]!r}" for k in ("sweep", "group") if source.get(k))
    sweeps = project_sweeps(api, entity, project)
    index = sweep_index(api, entity, project, {"group": group}) if group else {}
    ids = [i for i in index if i in sweeps] if group else list(sweeps)
    if want:
        ids = [i for i in ids if i == want] or [i for i in ids if sweeps[i].name == want]
    if not ids:
        raise SystemExit(f"no sweep matches {label} in {entity}/{project}; see --list")
    if len(ids) > 1 and source.get("sweeps", "latest") != "all":
        if not index:
            index = sweep_index(api, entity, project, {"sweep": {"$in": ids}})
        ids.sort(key=lambda i: index.get(i, {}).get("latest", ""))
        print(f"{label}: {len(ids)} sweeps match ({', '.join(ids)}); using {ids[-1]}, the one "
              f"with the newest run. Set `sweeps: all` to combine them.", flush=True)
        ids = ids[-1:]
    if group and index.get(None):
        print(f"group {group!r}: skipping {index[None]['runs']} runs that belong to no sweep", flush=True)
    return [sweeps[i] for i in ids]


def resolve(key, logged, aliases):
    """The W&B key a run logged `key` under: the key itself, else its alias, else for an
    indexed key such as agop/rayleigh_ratio2 the alias of its stem plus the index."""
    if key in logged:
        return key
    if aliases.get(key) in logged:
        return aliases[key]
    m = re.fullmatch(r"(.*?)(\d+)", key)
    if m and m.group(1) in aliases and aliases[m.group(1)] + m.group(2) in logged:
        return aliases[m.group(1)] + m.group(2)
    return None


def fetch_run(run, axes, metrics, aliases, cache_dir, refresh):
    """-> {id, name, state, created, axes, curves}, cached as one .npz per run.

    Each metric is read under whichever name `resolve` finds the run logged. The cache is
    reused only while the run's state is unchanged and terminal, it has every axis, and every
    metric still resolves to the same W&B key; so an unfinished run, or an edited alias, is
    fetched again.
    """
    path = os.path.join(cache_dir, f"{run.id}.npz")
    if os.path.exists(path) and not refresh:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        logged, cached = set(meta.get("logged", [])), meta.get("resolved", {})
        if (meta.get("state") == run.state and run.state in TERMINAL
                and all(a in meta["axes"] for a in axes)
                and all(k in cached and cached[k] == resolve(k, logged, aliases) for k in metrics)):
            return {**meta, "curves": {k: z[f"m_{i}"] for i, k in enumerate(meta["metrics"]) if k in metrics}}

    logged = set(run.summary.keys())
    resolved = {k: resolve(k, logged, aliases) for k in metrics}
    keys = sorted({v for v in resolved.values() if v})
    rows = run.history(keys=keys, samples=100_000, pandas=False) if keys else []
    if len(rows) >= 100_000:
        print(f"warning: run {run.name}: history was sampled to {len(rows)} rows", flush=True)
    steps = 1 + max([r.get("_step", 0) for r in rows], default=-1)
    curves = {}
    for k, key in resolved.items():
        if key is None:
            continue
        y = np.full(steps, np.nan, dtype=np.float32)
        for r in rows:
            v = r.get(key)
            if v is not None:
                y[r["_step"]] = float(v)          # W&B hands back "NaN" / "Infinity" as strings
        curves[k] = y
    meta = {"id": run.id, "name": run.name, "state": run.state, "created": run.created_at,
            "axes": {a: run.config.get(a) for a in axes}, "metrics": list(curves),
            "resolved": resolved, "logged": sorted(logged)}
    np.savez(path, meta=json.dumps(meta), **{f"m_{i}": curves[k] for i, k in enumerate(meta["metrics"])})
    return {**meta, "curves": curves}


def pull(api, entity, project, source, metrics, aliases, cache_root, refresh, workers):
    """-> (axes, runs) for one source: the union of its sweeps' axes, and one run per point
    (when sweeps are combined, the newest run of a point wins)."""
    from metrics import classify_regime

    sweeps = resolve_sweeps(api, entity, project, source)
    axes = {}
    for s in sweeps:
        for k, v in s.config["parameters"].items():
            if "values" in v:
                axes.setdefault(k, [])
                axes[k] += [x for x in v["values"] if x not in axes[k]]
    runs = []
    for s in sweeps:
        filters = {"sweep": s.id}
        if source.get("group"):
            filters = {"$and": [filters, {"group": source["group"]}]}
        listed = list(api.runs(f"{entity}/{project}", filters=filters, per_page=1000, include_sweeps=False))
        cache_dir = os.path.join(cache_root, s.id)
        os.makedirs(cache_dir, exist_ok=True)
        with ThreadPoolExecutor(workers) as pool:
            runs += pool.map(lambda r: fetch_run(r, axes, metrics, aliases, cache_dir, refresh), listed)
    newest = {}
    for r in runs:
        point = tuple(value_index(axes[k], r["axes"].get(k)) for k in axes)
        if point not in newest or (r.get("created") or "") > (newest[point].get("created") or ""):
            newest[point] = r
    if len(newest) < len(runs):
        print(f"{source_label(source)}: {len(runs) - len(newest)} runs repeat a point; kept the newest", flush=True)
    runs = list(newest.values())
    for r in runs:
        r["regime"] = classify_regime(r["curves"]["loss"]) if "loss" in r["curves"] else None
    print(f"{source_label(source)}: {len(runs)} runs from {', '.join(s.id for s in sweeps)}, "
          f"axes {list(axes)}", flush=True)
    return axes, runs


# ----------------------------------------------------------------------------- spec

def source_key(item):
    return (item.get("sweep"), item.get("group"), item.get("sweeps", "latest"))


def source_label(item):
    return item.get("sweep") or item.get("group") or "?"


def slug(text):
    return re.sub(r"[^A-Za-z0-9._=-]+", "-", str(text)).strip("-")


def section_id(sec):
    return slug(sec.get("id") or source_label(sec))


def video_name(video):
    base = lambda k: k.split("/")[-1].split(".")[-1]
    return slug(video.get("name") or (
        f"{source_label(video)}_{'-'.join(base(m) for m in video['rows'])}_by_{base(video['cols'])}"
        f"_per_{base(video['time'])}" + "".join(f"_{base(k)}{v}" for k, v in (video.get("fixed") or {}).items())))


def group_id(group):
    """A panel group's id: its `id:`, else its title in lower case; `title` is only the label."""
    return slug(group.get("id") or group["title"]).lower()


def video_place(spec, video):
    """-> (section id, group id or None) where a video is shown: its `section` path, such as
    "width-sweep/generalization" (or "width-sweep" for the end of the section), else the end of
    the first section that reads the same runs; (None, None) when no section does."""
    if video.get("section"):
        sec, _, grp = str(video["section"]).partition("/")
        return slug(sec), (slug(grp).lower() if grp else None)
    return next((section_id(s) for s in spec["sections"] if source_key(s) == source_key(video)), None), None


VIDEO_KEYS = {"section", "sweep", "group", "sweeps", "rows", "cols", "time", "fixed", "name", "title"}


def prepare_spec(spec):
    """The spec with each video that names only `section` reading that section's runs."""
    if not isinstance(spec.get("videos") or [], list):
        raise SystemExit("videos: must be a list of video specs")
    ids = {section_id(s): s for s in spec["sections"]}
    for v in spec.get("videos") or []:
        unknown = sorted(set(v) - VIDEO_KEYS)
        if unknown:
            raise SystemExit(f"video {v.get('name') or v}: unknown keys {unknown}; known: {sorted(VIDEO_KEYS)}"
                             + (" (`in` is now `section`)" if "in" in unknown else ""))
        sid = video_place(spec, v)[0] if v.get("section") else None
        if not (v.get("sweep") or v.get("group")) and sid in ids:
            v.update({k: ids[sid][k] for k in ("sweep", "group", "sweeps") if k in ids[sid]})
    return spec


def value_index(values, v):
    """Index of v among an axis's values: exact for None and strings, nearest for numbers."""
    if v is None or isinstance(v, (str, bool)):
        return values.index(v) if v in values else None
    nums = [(i, float(x)) for i, x in enumerate(values)
            if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if not nums:
        return None
    i, x = min(nums, key=lambda ix: abs(ix[1] - float(v)))
    return i if abs(x - float(v)) <= 1e-6 * max(1.0, abs(float(v))) else None


def axis_meta(spec, key, values):
    """Display information for one axis. Numeric axes are sorted ascending, so dropdowns,
    legends, color ramps and series read low to high. The spec's `labels` is a list (one per
    value, in the sweep's order) or a {value: label} map; unlabeled values show as they are."""
    a = (spec.get("axes") or {}).get(key, {})
    labels, fmt = a.get("labels"), a.get("format")
    show = lambda v: fmt.format(v) if fmt and isinstance(v, (int, float)) else str(v)
    if isinstance(labels, dict):
        shown = [str(labels[v]) if v in labels else show(v) for v in values]
    elif isinstance(labels, list):
        if len(labels) != len(values):
            print(f"warning: axis {key} has {len(labels)} labels for {len(values)} values; "
                  f"unlabeled values show as they are", flush=True)
        shown = [str(labels[i]) if i < len(labels) else show(v) for i, v in enumerate(values)]
    else:
        shown = [show(v) for v in values]
    pairs = list(zip(values, shown))
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        pairs.sort(key=lambda p: p[0])
    return {"key": key, "name": a.get("name", key),
            "values": [p[0] for p in pairs], "labels": [p[1] for p in pairs]}


def check_spec(spec, pulled):
    """Fail before rendering anything when the spec does not fit the sweeps it reads."""
    def on_axis(where, axes, key, value):
        if key not in axes:
            raise SystemExit(f"{where}: {key!r} is not an axis of its sweeps; axes: {list(axes)}")
        meta = axis_meta(spec, key, axes[key])
        if value_index(meta["values"], value) is None:
            raise SystemExit(f"{where}: {key} = {value!r} is not one of {meta['values']}")

    ids = [section_id(s) for s in spec["sections"]]
    if len(set(ids)) < len(ids):
        raise SystemExit(f"sections need distinct ids (set `id:`); got {ids}")
    for sec in spec["sections"]:
        where = f"section {sec.get('title', section_id(sec))!r}"
        axes, _ = pulled[source_key(sec)]
        if sec.get("color_by") not in axes:
            raise SystemExit(f"{where}: color_by {sec.get('color_by')!r} is not an axis; axes: {list(axes)}")
        for k, v in (sec.get("filters") or {}).items():
            if k == sec["color_by"]:
                raise SystemExit(f"{where}: {k} is both color_by and a filter")
            if v != "all":
                on_axis(where, axes, k, v)
            elif k not in axes:
                raise SystemExit(f"{where}: {k!r} is not an axis of its sweeps; axes: {list(axes)}")
    names = []
    sections = {section_id(s): s for s in spec["sections"]}
    paths = [f"{sid}/{group_id(g)}" for sid, s in sections.items() for g in s["groups"]]
    for sid, s in sections.items():
        gids = [group_id(g) for g in s["groups"]]
        if len(set(gids)) < len(gids):
            raise SystemExit(f"section {sid!r}: panel groups need distinct ids (set `id:`); got {gids}")
    for v in spec.get("videos") or []:
        where = f"video {video_name(v)!r}"
        axes, _ = pulled[source_key(v)]
        sid, gid = video_place(spec, v)
        if v.get("section") and (sid not in sections or (gid and f"{sid}/{gid}" not in paths)):
            raise SystemExit(f"{where}: section {v['section']!r} names no section or panel group; "
                             f"sections: {list(sections)}; groups: {paths}")
        if not v.get("rows"):
            raise SystemExit(f"{where}: needs at least one metric in rows")
        for k in (v.get("cols"), v.get("time")):
            if k not in axes:
                raise SystemExit(f"{where}: {k!r} is not an axis of its sweeps; axes: {list(axes)}")
        if v["cols"] == v["time"]:
            raise SystemExit(f"{where}: cols and time must be different axes")
        for k, x in (v.get("fixed") or {}).items():
            on_axis(where, axes, k, x)
        loose = set(axes) - {v["cols"], v["time"]} - set(v.get("fixed") or {})
        if loose:
            raise SystemExit(f"{where}: pin {sorted(loose)} in fixed, or several runs share each frame")
        names.append(video_name(v))
    if len(set(names)) < len(names):
        raise SystemExit(f"videos need distinct names (set `name:`); got {names}")


# ----------------------------------------------------------------------------- pages

def decimate(y, buckets=BUCKETS):
    """-> (steps, values) with every kept point at its own step.

    A curve no longer than 2 * buckets is kept whole. Otherwise this keeps its first and last
    steps, the min and max of each of `buckets` step ranges, and both sides of every switch
    between finite and non-finite values; non-finite points are kept as NaN, which the page
    draws as a gap.
    """
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    n = len(y)
    if n <= 2 * buckets:
        keep = np.arange(n)
    else:
        keep = {0, n - 1}
        edges = np.linspace(0, n, buckets + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            if finite[a:b].any():
                seg = np.where(finite[a:b], y[a:b], np.nan)
                keep |= {a + int(np.nanargmin(seg)), a + int(np.nanargmax(seg))}
        switch = np.flatnonzero(np.diff(finite.astype(np.int8)))
        keep |= set(switch.tolist()) | set((switch + 1).tolist())
        keep = np.asarray(sorted(keep))
    return keep, np.where(finite[keep], y[keep], np.nan).astype(np.float32)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def section_payload(spec, sec, axes, runs, videos):
    title = sec.get("title", section_id(sec))
    metas = [axis_meta(spec, k, v) for k, v in axes.items()]
    index = [[value_index(m["values"], r["axes"].get(m["key"])) for m in metas] for r in runs]
    keep = [i for i, ix in enumerate(index) if None not in ix]
    if not keep:
        raise SystemExit(f"section {title!r}: none of its {len(runs)} runs has a value on every axis")
    if len(keep) < len(runs):
        print(f"section {title!r}: leaving out {len(runs) - len(keep)} runs with a value off the axes", flush=True)
    steps = max(len(c) for i in keep for c in runs[i]["curves"].values())
    step_dtype = "<u2" if steps <= 65535 else "<u4"
    metrics = {}
    for p in (p for g in sec["groups"] for p in g["panels"]):
        if not any(p["metric"] in runs[i]["curves"] for i in keep):
            print(f"section {title!r}: no run logged {p['metric']!r}; its panel says so", flush=True)
            continue
        parts = [decimate(runs[i]["curves"][p["metric"]]) if p["metric"] in runs[i]["curves"]
                 else (np.zeros(0, int), np.zeros(0, np.float32)) for i in keep]
        offsets = np.cumsum([0] + [len(s) for s, _ in parts])
        metrics[p["metric"]] = {"s": b64(np.concatenate([s for s, _ in parts]).astype(step_dtype)),
                                "v": b64(np.concatenate([v for _, v in parts]).astype("<f4")),
                                "o": b64(offsets.astype("<u4"))}
    filters = {}
    for m in metas:
        if m["key"] != sec["color_by"]:
            v = (sec.get("filters") or {}).get(m["key"], "all")
            filters[m["key"]] = "all" if v == "all" else value_index(m["values"], v)
    return {"id": section_id(sec), "title": title, "text": sec.get("text", ""),
            "axes": metas, "runs": [index[i] for i in keep], "steps": int(steps),
            "step_bytes": 2 if step_dtype == "<u2" else 4, "metrics": metrics,
            "regimes": [runs[i].get("regime") for i in keep],
            "groups": [{**g, "id": group_id(g)} for g in sec["groups"]],
            "defaults": {"color_by": sec["color_by"], "filters": filters}, "videos": videos}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def vendor(cache_root):
    """The inlined libraries, downloaded once and checked against VENDOR's hashes."""
    os.makedirs(cache_root, exist_ok=True)
    out = {}
    for name, (url, digest) in VENDOR.items():
        path = os.path.join(cache_root, name)
        if not (os.path.exists(path) and sha256(path) == digest):
            part = path + ".part"
            try:
                urllib.request.urlretrieve(url, part)
                if sha256(part) != digest:
                    raise SystemExit(f"{url} does not match its pinned SHA-256; not inlining it")
                os.replace(part, path)
            finally:
                if os.path.exists(part):
                    os.remove(part)
        out[name] = open(path, encoding="utf-8").read().replace("</script", "<\\/script")
    return out


def write_page(path, title, text, sections, libs):
    data = json.dumps({"title": title, "text": text, "sections": sections}).replace("</", "<\\/")
    html = (PAGE.replace("{{TITLE}}", title).replace("{{UPLOT_CSS}}", libs["uplot.css"])
                .replace("{{UPLOT_JS}}", libs["uplot.js"]).replace("{{MARKED_JS}}", libs["marked.js"])
                .replace("{{MATHJAX_JS}}", libs["mathjax.js"]).replace("{{DATA}}", data))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(path, ROOT)} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


# ----------------------------------------------------------------------------- videos

def wrap_text(text, width, make):
    """text broken at spaces (never inside $...$) into as few lines as fit in `width` display
    units, balanced so they come out about equally long; `make(s)` places a throwaway matplotlib
    Text to measure. A word wider than `width` gets its own line."""
    words = re.split(r"\s+(?=(?:[^$]*\$[^$]*\$)*[^$]*$)", text.strip())
    measured = {}
    def wide(s):
        if s not in measured:
            t = make(s)
            measured[s] = t.get_window_extent(t.figure.canvas.get_renderer()).width
            t.remove()
        return measured[s]
    def greedy(limit):
        lines = [words[0]]
        for w in words[1:]:
            if wide(f"{lines[-1]} {w}") <= limit:
                lines[-1] += f" {w}"
            else:
                lines.append(w)
        return lines
    n = len(greedy(width))
    lo, hi = 0.0, width                      # the narrowest limit that still needs only n lines
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if len(greedy(mid)) <= n else (mid, hi)
    return "\n".join(greedy(hi))


def render_video(spec, video, axes, runs, titles, path, fps=4):
    """Rows = metrics, cols = values of video['cols'], one frame per value of video['time']
    (ascending for a numeric axis). check_spec has made sure every other axis is fixed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.lines import Line2D

    name = video_name(video)
    metas = {k: axis_meta(spec, k, v) for k, v in axes.items()}
    col_meta, time_meta = metas[video["cols"]], metas[video["time"]]
    fixed = {k: value_index(metas[k]["values"], v) for k, v in (video.get("fixed") or {}).items()}
    table = {}
    for r in runs:
        ix = {k: value_index(metas[k]["values"], r["axes"].get(k)) for k in axes}
        if None in ix.values() or any(ix[k] != v for k, v in fixed.items()):
            continue
        cell = (ix[video["cols"]], ix[video["time"]])
        if cell in table:
            raise SystemExit(f"video {name}: runs {table[cell]['name']} and {r['name']} share a frame cell")
        table[cell] = r
    if not table:
        raise SystemExit(f"video {name}: no run matches fixed = {video.get('fixed')}")
    rows, C, T = video["rows"], len(col_meta["values"]), len(time_meta["values"])

    # Limits from the runs that did not diverge: a log row reaches their largest value, so every
    # catapult peak stays in view while diverging curves leave the top; a linear row stops at the
    # 99th percentile, where a single spike would flatten everything else.
    log_rows = {m: titles.get(m, {}).get("log_y", False) for m in rows}
    usable = [r for r in table.values() if r.get("regime") != "diverged"] or list(table.values())
    limits = {}
    for m in rows:
        parts = [r["curves"][m] for r in usable if m in r["curves"]]
        v = np.concatenate(parts) if parts else np.empty(0, np.float32)
        v = v[np.isfinite(v) & ((v > 0) if log_rows[m] else True)]
        if not len(v):
            print(f"video {name}: no run logged {m!r}; its row stays empty", flush=True)
            limits[m] = (1e-3, 1.0) if log_rows[m] else (0.0, 1.0)
        elif log_rows[m]:
            limits[m] = (v.min() / 1.3, v.max() * 1.3)
        else:
            lo, hi = float(v.min()), float(np.percentile(v, 99))
            pad = 0.05 * (hi - lo) or 0.05 * abs(hi) or 1.0
            limits[m] = (lo - pad, hi + pad)

    plt.style.use("dark_background")
    fig, axs = plt.subplots(len(rows), C, figsize=(3.4 * C, 2.5 * len(rows) + 1.1),
                            squeeze=False, sharex=True)
    fig.patch.set_facecolor("#0f1115")
    handles = [Line2D([], [], color=c, lw=2, label=k) for k, c in REGIME_COLORS.items()]
    handles.append(Line2D([], [], color="#8b949e", lw=1, alpha=0.6, label=f"earlier {time_meta['name']}"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fixed_txt = "   ".join(f"{metas[k]['name']} = {metas[k]['labels'][v]}" for k, v in fixed.items())
    ylabels = {m: "" for m in rows}          # wrapped to the plot height after a first layout

    def draw(t):
        for i, m in enumerate(rows):
            for j in range(C):
                ax = axs[i, j]
                ax.clear()
                ax.set_facecolor("#171a21")
                for t_prev in range(t):
                    r = table.get((j, t_prev))
                    if r is not None and m in r["curves"]:
                        ax.plot(r["curves"][m], color="#8b949e", lw=0.7, alpha=0.18)
                r = table.get((j, t))
                if r is not None and m in r["curves"]:
                    reg = r.get("regime") or "monotonic"
                    ax.plot(r["curves"][m], color=REGIME_COLORS[reg], lw=1.6)
                    if i == 0:
                        ax.text(0.03, 0.05, reg, transform=ax.transAxes, ha="left", va="bottom",
                                fontsize=8, color=REGIME_COLORS[reg])
                if log_rows[m]:
                    ax.set_yscale("log")
                ax.set_ylim(*limits[m])
                ax.grid(color="#262a33", lw=0.6)
                if i == 0:
                    ax.set_title(f"{col_meta['name']} = {col_meta['labels'][j]}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(ylabels[m], fontsize=10)
                if i == len(rows) - 1:
                    ax.set_xlabel("step", fontsize=9)
                ax.tick_params(labelsize=8)
        fig.suptitle(f"{video['title']}     {time_meta['name']} = {time_meta['labels'][t]}"
                     f"   ({t + 1}/{T})     {fixed_txt}", fontsize=12)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Lay the grid out without y labels, then wrap each label at spaces so every line fits its
    # row's height. A wrapped label only adds width, so the next layout keeps these heights.
    draw(0)
    renderer = fig.canvas.get_renderer()
    for i, m in enumerate(rows):
        ylabels[m] = wrap_text(titles.get(m, {}).get("title", m), 0.95 * axs[i, 0].get_window_extent(renderer).height,
                               lambda s: axs[i, 0].text(0, 0, s, fontsize=10))

    anim = FuncAnimation(fig, draw, frames=T, interval=1000 // fps)
    # H.264 in yuv420p (what browsers and W&B play) needs even frame sides: pad by a pixel.
    anim.save(path, writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=[
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(table)} runs, {T} frames)", flush=True)
    return {"name": name,
            "caption": f"rows: {', '.join(titles.get(m, {}).get('title', m) for m in rows)} · "
                       f"columns: {col_meta['name']} · frames: {time_meta['name']}"
                       + (f" · {fixed_txt}" if fixed_txt else "")}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default="dashboard.yaml", help="file in configs/")
    ap.add_argument("--project", default=None,
                    help="W&B project to read, ENTITY/PROJECT or PROJECT (default: the spec's "
                         "wandb: block, else configs/base.yaml)")
    ap.add_argument("--list", nargs="?", const="", default=None, metavar="TEXT",
                    help="list the project's sweeps and groups (only rows containing TEXT) and exit")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--refresh", action="store_true", help="re-download every run")
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--upload", action="store_true", help="log the pages and videos to one W&B run")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    spec_path = a.spec if os.path.isabs(a.spec) else os.path.join(ROOT, "configs", a.spec)
    spec = yaml.safe_load(open(spec_path))
    base = load_config("base.yaml")
    entity, project = project_path(a.project, spec, base)

    import wandb
    api = wandb.Api(timeout=120)
    if a.list is not None:
        list_sweeps(api, entity, project, a.list)
        return

    outdir = a.outdir or os.path.join(ROOT, get_in(base, "paths.outputs"), "dashboard")
    cache = os.path.join(outdir, "cache")
    os.makedirs(os.path.join(outdir, "videos"), exist_ok=True)
    aliases = spec.get("aliases") or {}
    titles = {p["metric"]: p for g in spec.get("groups") or [] for p in g["panels"]}
    for sec in spec["sections"]:
        titles.update({p["metric"]: p for g in sec["groups"] for p in g["panels"]})
    spec = prepare_spec(spec)
    ids = {section_id(s): s for s in spec["sections"]}
    items = spec["sections"] + (spec.get("videos") or [])
    for item in items:
        if not (item.get("sweep") or item.get("group")):
            raise SystemExit(f"every section and video needs a `sweep` or a `group`: {item}")

    pulled, sources = {}, {}
    for item in items:
        sources.setdefault(source_key(item), item)
    for key, item in sources.items():
        metrics = {"loss"}                                    # videos and y-ranges use its regime
        for s in spec["sections"]:
            if source_key(s) == key:
                metrics |= {p["metric"] for g in s["groups"] for p in g["panels"]}
        for v in spec.get("videos") or []:
            if source_key(v) == key:
                metrics |= set(v["rows"])
        pulled[key] = pull(api, entity, project, item, sorted(metrics), aliases,
                           cache, a.refresh, a.workers)
    check_spec(spec, pulled)

    videos = {}
    if not a.no_videos:
        for v in spec.get("videos") or []:
            target, under = video_place(spec, v)
            sec = ids.get(target, {})
            v = {"title": sec.get("title", source_label(v)), **v}
            path = os.path.join(outdir, "videos", f"{video_name(v)}.mp4")
            info = render_video(spec, v, *pulled[source_key(v)], titles, path)
            with open(path, "rb") as f:
                info["src"] = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
            info.update(path=path, under=under)
            if target is None:
                print(f"video {info['name']}: no section reads its runs, so no page shows it "
                      f"(set `section:`); it is still in videos/ and uploaded", flush=True)
            videos.setdefault(target, []).append(info)

    libs = vendor(os.path.join(cache, "vendor"))
    sections, pages = [], {}
    for sec in spec["sections"]:
        payload = section_payload(spec, sec, *pulled[source_key(sec)],
                                  [{"src": v["src"], "caption": v["caption"], "under": v["under"]}
                                   for v in videos.get(section_id(sec), [])])
        sections.append(payload)
        page = os.path.join(outdir, f"dashboard_{payload['id']}.html")
        write_page(page, f"{spec['title']}: {payload['title']}", "", [payload], libs)
        pages[payload["id"]] = page
    write_page(os.path.join(outdir, "dashboard.html"), spec["title"], spec.get("text", ""), sections, libs)

    if a.upload:
        run = wandb.init(entity=entity, project=project, name="sweep-dashboard", group="dashboard",
                         job_type="dashboard", config={"spec": os.path.relpath(spec_path, ROOT)})
        log = {f"dashboard/{k}": wandb.Html(p, inject=False) for k, p in pages.items()}
        log.update({f"videos/{v['name']}": wandb.Video(v["path"], format="mp4")
                    for vs in videos.values() for v in vs})
        run.log(log)
        print(f"logged {len(log)} items to {run.url}", flush=True)
        run.finish()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>{{UPLOT_CSS}}</style>
<style>
:root { --bg:#0f1115; --panel:#171a21; --raised:#1d2129; --border:#2a2f3a; --text:#e6e6e6;
        --muted:#9aa4b2; --accent:#5aa9e6; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width:1300px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:24px; margin:0 0 8px; } h2 { font-size:20px; margin:40px 0 4px; }
h3 { font-size:15px; margin:22px 0 8px; color:var(--muted); font-weight:600;
     text-transform:uppercase; letter-spacing:.04em; }
a { color:var(--accent); } .md p { margin:6px 0; } .md code { background:var(--raised); padding:0 4px; border-radius:4px; }
nav { display:flex; gap:14px; flex-wrap:wrap; margin:8px 0 0; font-size:13px; }
.controls { position:sticky; top:0; z-index:5; background:rgba(15,17,21,.94); backdrop-filter:blur(4px);
            border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin:10px 0 4px;
            display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; }
.controls label { color:var(--muted); font-size:12px; display:flex; gap:6px; align-items:center; }
select { background:var(--raised); color:var(--text); border:1px solid var(--border);
         border-radius:6px; padding:3px 6px; font-size:13px; }
.legend { display:flex; flex-wrap:wrap; gap:4px 12px; font-size:12px; color:var(--muted); width:100%; }
.legend span i { display:inline-block; width:14px; height:3px; margin-right:5px; vertical-align:middle; border-radius:2px; }
.count { font-size:12px; color:var(--muted); margin-left:auto; }
.grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; }
@media (max-width: 820px) { .grid { grid-template-columns:minmax(0, 1fr); } }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:10px 10px 4px; min-width:0; }
.panel .head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
.panel .t { font-weight:600; font-size:13px; } .panel .d { font-size:12px; color:var(--muted); min-height:18px; }
.panel button { background:none; border:1px solid var(--border); color:var(--muted); border-radius:5px;
                font-size:11px; padding:1px 6px; cursor:pointer; }
.panel button.on { color:var(--accent); border-color:var(--accent); }
.readout { font-size:11px; color:var(--muted); height:16px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.empty { color:var(--muted); font-size:12px; padding:30px 0; text-align:center; }
.videos video { width:100%; border-radius:8px; border:1px solid var(--border); background:#000; margin-top:8px; }
.videos .d { font-size:12px; color:var(--muted); margin-bottom:14px; }
.uplot, .u-wrap { background:transparent; } .u-select { background:rgba(90,169,230,.15); }
.toolbar { display:flex; gap:10px; justify-content:flex-end; align-items:center; font-size:12px; color:var(--muted); }
button.btn { background:var(--raised); color:var(--text); border:1px solid var(--border); border-radius:6px;
             padding:3px 10px; font-size:12px; cursor:pointer; }
.note { margin:6px 0 4px; }
.note .add { background:none; border:1px dashed var(--border); color:var(--muted); border-radius:6px;
             padding:1px 8px; font-size:11px; cursor:pointer; }
.note textarea { width:100%; min-height:70px; background:var(--raised); color:var(--text); border:1px solid var(--accent);
                 border-radius:6px; padding:6px 8px; font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; resize:vertical; }
.note .preview { font-size:13px; border-left:2px solid var(--accent); padding:2px 0 2px 8px; cursor:text; }
.modal { position:fixed; inset:0; background:rgba(0,0,0,.6); z-index:20; display:flex; align-items:center; justify-content:center; }
.modal .box { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px; width:min(960px, 94vw);
              max-height:90vh; display:flex; flex-direction:column; gap:8px; }
.modal textarea { min-height:55vh; background:var(--raised); color:var(--text); border:1px solid var(--border); border-radius:6px;
                  padding:8px; font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
.modal .row { display:flex; gap:8px; justify-content:flex-end; align-items:center; color:var(--muted); font-size:12px; }
</style>
<script>window.MathJax = { tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] },
                           svg: { fontCache: 'global' }, startup: { typeset: false } };</script>
<script>{{MATHJAX_JS}}</script>
<script>{{MARKED_JS}}</script>
<script>{{UPLOT_JS}}</script>
</head><body><main id="app"></main>
<script type="application/json" id="data">{{DATA}}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const CAT = ['#5aa9e6','#f5a623','#7ee787','#e5534b','#bc8cff','#ffd866','#39c5cf','#f778ba','#a5d6ff','#d29922'];
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Markdown with LaTeX. Math and \$ reach MathJax exactly as written; marked's own lexer handles
// code spans and the other escapes, so a $ inside `code` is just a dollar sign.
marked.use({ extensions: [
  { name: 'dollar', level: 'inline',
    start(s) { const i = s.indexOf('\\$'); return i < 0 ? undefined : i; },
    tokenizer(s) { if (s.startsWith('\\$')) return { type: 'dollar', raw: '\\$' }; },
    renderer() { return '\\$'; } },
  { name: 'math', level: 'inline',
    start(s) { const i = s.indexOf('$'); return i < 0 ? undefined : i; },
    tokenizer(s) { const m = /^\$\$[\s\S]+?\$\$/.exec(s) || /^\$(?:\\[\s\S]|[^\\$\n]|\n(?![ \t]*\n))+?\$/.exec(s);
                   if (m) return { type: 'math', raw: m[0] }; },
    renderer(t) { return esc(t.raw); } },
]});
const md = text => marked.parse(String(text || ''));

let typesetting = Promise.resolve();   // MathJax calls one at a time
function typeset(node) {
  if (!window.MathJax || !MathJax.startup) return;
  typesetting = typesetting.then(() => MathJax.startup.promise).then(() => MathJax.typesetPromise([node]))
                           .catch(e => console.error(e));
}
function untypeset(node) { try { if (window.MathJax && MathJax.typesetClear) MathJax.typesetClear([node]); } catch (e) {} }

// Comments: Markdown + LaTeX notes on any panel, group, video or section. They are kept in this
// browser (localStorage, when the page may use it) and leave it through "Export notes".
const NOTES_KEY = 'agop-dashboard-notes';
const NOTES = (() => { try { return JSON.parse(localStorage.getItem(NOTES_KEY) || '{}'); } catch (e) { return {}; } })();
let noteCount = null;
function saveNotes() { try { localStorage.setItem(NOTES_KEY, JSON.stringify(NOTES)); } catch (e) {} updateCount(); }
function updateCount() { if (noteCount) { const n = Object.keys(NOTES).length; noteCount.textContent = `${n} comment${n === 1 ? '' : 's'}`; } }
function noteBox(id, label) {
  const box = el('div', 'note'), add = el('button', 'add', '+ ' + esc(label));
  const area = el('textarea'), preview = el('div', 'preview md');
  area.placeholder = 'Markdown with $LaTeX$; click outside to render, click the text to edit';
  area.value = NOTES[id] || '';
  const show = editing => { add.style.display = editing || NOTES[id] ? 'none' : '';
    area.style.display = editing ? '' : 'none'; preview.style.display = !editing && NOTES[id] ? '' : 'none'; };
  const refresh = () => { untypeset(preview); preview.innerHTML = md(NOTES[id] || ''); typeset(preview); };
  add.onclick = () => { show(true); area.focus(); };
  preview.onclick = () => { show(true); area.focus(); };
  area.oninput = () => { if (area.value.trim()) NOTES[id] = area.value; else delete NOTES[id]; saveNotes(); };
  area.onblur = () => { refresh(); show(false); };
  box.append(add, area, preview); refresh(); show(false);
  return box;
}

// A PNG of one panel as it looks now: title, legend and the uPlot canvas on the panel background.
function panelImage(u, title, legend) {
  const src = u.ctx.canvas, r = window.devicePixelRatio || 1, W = src.width + 20 * r;
  const meas = document.createElement('canvas').getContext('2d'); meas.font = `${11 * r}px sans-serif`;
  let x = 10 * r, rows = legend.length ? 1 : 0;
  const items = legend.map(l => { const w = 22 * r + meas.measureText(l.label).width + 12 * r;
    if (x + w > W - 10 * r && x > 10 * r) { rows++; x = 10 * r; } const it = { ...l, x, row: rows - 1 }; x += w; return it; });
  const top = 28 * r, legH = rows * 16 * r, H = top + legH + src.height + 10 * r;
  const c = document.createElement('canvas'); c.width = W; c.height = H; const g = c.getContext('2d');
  g.fillStyle = '#171a21'; g.fillRect(0, 0, W, H); g.textBaseline = 'middle';
  g.fillStyle = '#e6e6e6'; g.font = `600 ${13 * r}px sans-serif`; g.fillText(title, 10 * r, 14 * r);
  g.font = `${11 * r}px sans-serif`;
  items.forEach(it => { const y = top + it.row * 16 * r + 8 * r;
    g.fillStyle = it.color; g.fillRect(it.x, y - 1.5 * r, 14 * r, 3 * r); g.fillStyle = '#9aa4b2'; g.fillText(it.label, it.x + 20 * r, y); });
  g.drawImage(src, 10 * r, top + legH);
  return c.toDataURL('image/png');
}
function videoFrame(v) {
  if (!v.videoWidth) return null;
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0); return c.toDataURL('image/png');
}

function exportText() {
  const parts = (window.__dashboard || []).flatMap(S => S.exportMd());
  if (!parts.length) return '';
  return [`# ${DATA.title}`, '', `_Comments exported ${new Date().toLocaleString()}_`, '', ...parts].join('\n');
}
function showExport() {
  const text = exportText(), modal = el('div', 'modal'), box = el('div', 'box'); modal.appendChild(box);
  box.appendChild(el('div', 'row', '<span style="margin-right:auto">Markdown with the plots as inline PNG images</span>'));
  const ta = el('textarea'); ta.readOnly = true; ta.value = text || 'No comments yet: add one under any plot, group, video or section.';
  const row = box.querySelector('.row'), dl = el('button', 'btn', 'Download .md'), cp = el('button', 'btn', 'Copy'), cl = el('button', 'btn', 'Close');
  dl.onclick = () => { try { const a = el('a'); a.href = URL.createObjectURL(new Blob([text], { type: 'text/markdown' }));
    a.download = 'dashboard-comments.md'; document.body.appendChild(a); a.click(); a.remove(); }
    catch (e) { ta.select(); } };
  cp.onclick = () => { const fallback = () => { ta.focus(); ta.select(); try { document.execCommand('copy'); } catch (e) {} };
    (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(() => { cp.textContent = 'Copied'; }, fallback); };
  cl.onclick = () => modal.remove();
  modal.onclick = e => { if (e.target === modal) modal.remove(); };
  row.append(dl, cp, cl); box.appendChild(ta); document.body.appendChild(modal);
}

function bytes(b64) { const s = atob(b64), b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); return b.buffer; }
function viridis(t) { // a short sequential ramp for many-valued axes such as lr
  const s = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const x = Math.min(Math.max(t, 0), 1) * (s.length - 1), i = Math.min(Math.floor(x), s.length - 2), f = x - i;
  return 'rgb(' + s[i].map((c, k) => Math.round(c + f * (s[i + 1][k] - c))).join(',') + ')'; }
const fmt = v => v == null ? '–' : (Math.abs(v) >= 1e3 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(2) : v.toPrecision(3);
// Short tick labels (1e6, 7.5e4, 2.5e-4, 0.75), so wide numbers are never clipped.
const tick = v => v == null ? '' : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0))
  ? v.toExponential(1).replace('e+', 'e').replace('.0e', 'e') : String(+v.toPrecision(4));
// At most about six labelled decades on a log axis, however many it spans.
function logSplits(u, ai, min, max) {
  const a = Math.floor(Math.log10(min)), b = Math.ceil(Math.log10(max)), step = Math.max(1, Math.ceil((b - a) / 5)), out = [];
  for (let k = a; k <= b; k += step) { const v = Math.pow(10, k); if (v >= min && v <= max) out.push(v); }
  return out.length ? out : [min, max];
}

function Section(sec, root) {
  const L = sec.steps, runs = sec.runs, ax = sec.axes, X = Array.from({ length: L }, (_, i) => i);
  const Steps = sec.step_bytes === 2 ? Uint16Array : Uint32Array;
  const metrics = Object.fromEntries(Object.entries(sec.metrics).map(([k, m]) =>
    [k, { s: new Steps(bytes(m.s)), v: new Float32Array(bytes(m.v)), o: new Uint32Array(bytes(m.o)) }]));
  const state = { colorBy: sec.defaults.color_by, filters: { ...sec.defaults.filters }, log: {} };
  sec.groups.forEach(g => g.panels.forEach(p => state.log[p.metric] = !!p.log_y));
  const axisIdx = Object.fromEntries(ax.map((a, i) => [a.key, i]));
  const wrap = el('section'); wrap.id = sec.id; root.appendChild(wrap);
  wrap.appendChild(el('h2', null, esc(sec.title)));
  if (sec.text) wrap.appendChild(el('div', 'md', md(sec.text)));
  wrap.appendChild(noteBox(`${sec.id}::section`, 'comment on this section'));
  const bar = el('div', 'controls'); wrap.appendChild(bar);
  const body = el('div'); wrap.appendChild(body);
  // Each group's frame (heading, text, grid, videos, comment) is built once; a control change
  // redraws only the panel grids, so videos being watched and comments being typed stay put.
  const vids = [], videos = sec.videos || [];
  function videoBlock(i, parent) {
    const v = videos[i], vid = el('video'); Object.assign(vid, { controls: true, loop: true, muted: true, playsInline: true });
    vid.src = v.src; vids[i] = vid; parent.append(vid, el('div', 'd', esc(v.caption)), noteBox(`${sec.id}::video::${i}`, 'comment on this video'));
  }
  const frames = sec.groups.map(g => {
    body.appendChild(el('h3', null, esc(g.title)));
    if (g.text) body.appendChild(el('div', 'md', md(g.text)));
    const grid = el('div', 'grid'); body.appendChild(grid);
    const box = el('div', 'videos'); body.appendChild(box);
    videos.forEach((v, i) => { if (v.under === g.id) videoBlock(i, box); });
    const key = `${sec.id}::group::${g.id}`, old = `${sec.id}::group::${g.title}`;   // keyed by id; move a title-keyed comment
    if (key !== old && NOTES[old] && !NOTES[key]) { NOTES[key] = NOTES[old]; delete NOTES[old]; saveNotes(); }
    body.appendChild(noteBox(key, `comment on ${g.title}`));
    return { g, grid };
  });
  if (videos.some(v => !v.under)) {                 // videos without a group close the section
    const box = el('div', 'videos'); wrap.appendChild(box); box.appendChild(el('h3', null, 'Videos'));
    videos.forEach((v, i) => { if (!v.under) videoBlock(i, box); });
  }
  let charts = [], legendItems = [];

  function runSet() {
    return [`color by ${ax[axisIdx[state.colorBy]].name}`, ...ax.filter(a => a.key !== state.colorBy).map(a =>
      `${a.name} = ${state.filters[a.key] === 'all' ? 'all' : a.labels[state.filters[a.key]]}`)].join(' · ');
  }
  function videoMd(i) {
    const n = NOTES[`${sec.id}::video::${i}`]; if (!n) return [];
    const img = videoFrame(vids[i]);
    return ['#### Video', '', `_${videos[i].caption}_`, '', ...(img ? [`![video frame](${img})`, ''] : []), n, ''];
  }
  // This section's comments as Markdown, with images of the commented panels as they look now.
  function exportMd() {
    const out = [], note = id => NOTES[`${sec.id}::${id}`];
    sec.groups.forEach(g => {
      const lines = [];
      if (note(`group::${g.id}`)) lines.push(note(`group::${g.id}`), '');
      g.panels.forEach(p => { const n = note(`panel::${p.metric}`); if (!n) return;
        const c = charts.find(c => c.metric === p.metric);
        lines.push(`#### ${p.title}`, '');
        if (p.text) lines.push(p.text, '');
        if (c) lines.push(`![${p.title}](${panelImage(c.u, p.title, legendItems)})`, '');
        lines.push(n, ''); });
      videos.forEach((v, i) => { if (v.under === g.id) lines.push(...videoMd(i)); });
      if (lines.length) out.push(`### ${g.title}`, '', ...lines);
    });
    const rest = videos.flatMap((v, i) => v.under ? [] : videoMd(i));
    if (rest.length) out.push('### Videos', '', ...rest);
    if (!note('section') && !out.length) return [];
    return [`## ${sec.title}`, '', `_Run set: ${runSet()}_`, '', ...(note('section') ? [note('section'), ''] : []), ...out];
  }
  (window.__dashboard = window.__dashboard || []).push({ id: sec.id, state, charts: () => charts.map(c => c.u), exportMd });

  function controls() {
    bar.innerHTML = '';
    const cb = el('label', null, 'color by');
    const s = el('select'); ax.forEach(a => s.add(new Option(a.name, a.key, false, a.key === state.colorBy)));
    s.onchange = () => { const old = state.colorBy; state.colorBy = s.value;
      delete state.filters[s.value]; state.filters[old] = 'all'; controls(); render(); };
    cb.appendChild(s); bar.appendChild(cb);
    ax.filter(a => a.key !== state.colorBy).forEach(a => {
      const l = el('label', null, esc(a.name));
      const f = el('select'); f.add(new Option('all', 'all'));
      a.labels.forEach((lab, i) => f.add(new Option(lab, i, false, state.filters[a.key] === i)));
      f.onchange = () => { state.filters[a.key] = f.value === 'all' ? 'all' : +f.value; render(); };
      l.appendChild(f); bar.appendChild(l);
    });
    bar.appendChild(el('span', 'count'));
    bar.appendChild(el('div', 'legend'));
  }

  function selected() {
    const c = axisIdx[state.colorBy];
    return runs.map((r, i) => i).filter(i => ax.every((a, k) =>
      a.key === state.colorBy || state.filters[a.key] === 'all' || runs[i][k] === state.filters[a.key]))
      .sort((i, j) => runs[i][c] - runs[j][c]);
  }

  function colorOf(v) { const a = ax[axisIdx[state.colorBy]];
    return a.values.length <= CAT.length ? CAT[v % CAT.length] : viridis(v / (a.values.length - 1)); }

  function label(i) { const r = runs[i];
    return ax.filter(a => a.key === state.colorBy || state.filters[a.key] === 'all')
             .map(a => (a.key !== state.colorBy ? a.name + '=' : '') + a.labels[r[axisIdx[a.key]]]).join(' · '); }

  // One run's points at their own steps: undefined between them (drawn through), null where the
  // run is NaN/inf or, on a log axis, not positive (drawn as a gap).
  function curve(m, r, log) { const a = new Array(L);
    for (let k = m.o[r]; k < m.o[r + 1]; k++) { const v = m.v[k]; a[m.s[k]] = Number.isFinite(v) && (!log || v > 0) ? v : null; }
    return a; }

  // The y range of the runs that did not diverge (all of them if every one did), kept under
  // 1e15: uPlot draws no ticks once a scale needs more than 17 integer digits.
  function yRange(m, sel, log) {
    const keep = sel.filter(i => sec.regimes[i] !== 'diverged'), pool = keep.length ? keep : sel;
    let lo = Infinity, hi = -Infinity;
    for (const r of pool) for (let k = m.o[r]; k < m.o[r + 1]; k++) { const v = m.v[k];
      if (Number.isFinite(v) && (!log || v > 0)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    if (!(lo <= hi)) return log ? [1e-3, 1] : [0, 1];
    hi = Math.min(hi, 1e15); lo = Math.min(Math.max(lo, log ? 1e-300 : -1e15), hi);
    if (log) return [lo / 1.3, hi * 1.3];
    const pad = (hi - lo) * 0.05 || Math.abs(hi) * 0.05 || 1;
    return [lo - pad, hi + pad];
  }

  function render() {
    const y = window.scrollY; body.style.minHeight = body.offsetHeight + 'px';   // keep the page from shrinking
    charts.forEach(c => c.u.destroy()); charts = [];
    frames.forEach(({ grid }) => { untypeset(grid); grid.innerHTML = ''; });
    const sel = selected(), c = axisIdx[state.colorBy], a = ax[c];
    bar.querySelector('.count').textContent = `${sel.length} run${sel.length === 1 ? '' : 's'}`;
    const shown = [...new Set(sel.map(i => runs[i][c]))];
    legendItems = shown.map(v => ({ label: a.labels[v], color: colorOf(v) }));
    bar.querySelector('.legend').innerHTML = shown.map(v =>
      `<span><i style="background:${colorOf(v)}"></i>${esc(a.labels[v])}</span>`).join('');
    frames.forEach(({ g, grid }) => {
      g.panels.forEach(p => {
        const box = el('div', 'panel'); grid.appendChild(box);
        const head = el('div', 'head'); box.appendChild(head);
        head.appendChild(el('span', 't', esc(p.title)));
        const btn = el('button', state.log[p.metric] ? 'on' : '', 'log y'); head.appendChild(btn);
        btn.onclick = () => { state.log[p.metric] = !state.log[p.metric]; render(); };
        box.appendChild(el('div', 'd md', p.text ? md(p.text) : ''));
        const readout = el('div', 'readout'); box.appendChild(readout);
        const plot = el('div'); plot.style.minHeight = '210px'; box.appendChild(plot);   // uPlot sizes itself a tick later
        box.appendChild(noteBox(`${sec.id}::panel::${p.metric}`, 'comment'));
        const m = metrics[p.metric];
        if (!m || !sel.length) { plot.appendChild(el('div', 'empty', m ? 'no runs match' : 'not logged by this sweep')); return; }
        const log = state.log[p.metric], range = yRange(m, sel, log);
        const data = [X, ...sel.map(i => curve(m, i, log))];
        const series = [{}, ...sel.map(i => ({ label: label(i), stroke: colorOf(runs[i][c]), width: 1.4,
                                              points: { show: false } }))];
        const axis = { stroke: '#9aa4b2', grid: { stroke: '#262a33', width: 1 }, ticks: { stroke: '#262a33', width: 1 },
                       font: '11px -apple-system, sans-serif' };
        const opts = { width: Math.max(plot.clientWidth || box.clientWidth - 20, 200), height: 210, series,
          legend: { show: false }, cursor: { sync: { key: sec.id }, points: { show: false } },
          scales: { x: { time: false }, y: log ? { distr: 3, log: 10, range: () => range } : { range: () => range } },
          axes: [{ ...axis, label: 'step', labelSize: 16, labelFont: '11px sans-serif' },
                 { ...axis, size: 56, values: (u, splits) => splits.map(tick),
                   ...(log ? { splits: logSplits, filter: (u, splits) => splits } : {}) }],   // label every chosen decade
          hooks: { setCursor: [u => { const k = u.cursor.idx, idxs = u.cursor.idxs || [];
            if (k == null) { readout.textContent = ''; return; }
            const vals = sel.map((r, s) => { const j = idxs[s + 1];
              return `${series[s + 1].label}: ${fmt(j == null ? null : u.data[s + 1][j])}${j != null && j !== k ? ' @' + j : ''}`; });
            readout.textContent = `step ${k}` + (vals.length <= 6 ? '  ·  ' + vals.join('  ·  ') : `  ·  ${vals.length} runs`); }] } };
        charts.push({ u: new uPlot(opts, data, plot), el: plot, metric: p.metric });
      });
    });
    frames.forEach(({ grid }) => typeset(grid));
    requestAnimationFrame(() => { body.style.minHeight = ''; window.scrollTo(0, y); });   // after uPlot has sized
  }
  controls(); render();
  let t; window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(() =>
    charts.forEach(({ u, el }) => u.setSize({ width: Math.max(el.clientWidth, 200), height: 210 })), 150); });
}

const app = document.getElementById('app');
{ const bar = el('div', 'toolbar'); noteCount = el('span'); const ex = el('button', 'btn', 'Export comments');
  ex.onclick = showExport; bar.append(noteCount, ex); app.appendChild(bar); updateCount(); }
if (DATA.sections.length > 1 || DATA.text) {
  app.appendChild(el('h1', null, esc(DATA.title)));
  if (DATA.text) app.appendChild(el('div', 'md', md(DATA.text)));
  const nav = el('nav'); app.appendChild(nav);
  DATA.sections.forEach(s => { const a = el('a', null, esc(s.title)); a.href = '#' + s.id; nav.appendChild(a); });
}
DATA.sections.forEach(s => Section(s, app));
typeset(app);
</script></body></html>
"""

if __name__ == "__main__":
    main()
