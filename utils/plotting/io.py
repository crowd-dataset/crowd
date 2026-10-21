import atexit
import concurrent.futures
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import webbrowser

import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.io._utils import validate_coerce_fig_to_dict
from plotly.offline import get_plotlyjs

import common
from custom_logger import CustomLogger
from utils.plotting import hover
from utils.plotting.constants import TOPOJSON_URL

logger = CustomLogger(__name__)  # use custom logger

# Without this, kaleido bakes a "Loading [MathJax]..." box into PDF/EPS exports; no figure uses LaTeX.
if pio.kaleido.scope is not None:  # None where plotly and kaleido versions do not match (images then fail anyway)
    pio.kaleido.scope.mathjax = None
    pio.kaleido.scope.topojson = TOPOJSON_URL  # see constants.TOPOJSON_URL


# Bump when a change here changes the saved files without changing the figures (e.g. how EPS is made): every
# figure is then saved again once.
RENDER_VERSION = 1
HASHES = os.path.join(common.root_dir, "figures", "figure_hashes.json")  # committed with the figures
RENDER_THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))


_LONG_DECIMALS = re.compile(r"(\.\d{9})\d+")


def _digest(*parts) -> str:
    """Short hash of the parts (figure JSON, sizes, scripts, versions): what a saved file depends on. Decimals past
    the 9th are left out: sums in a different order (polars works in parallel) change the last digits from run to
    run, which no figure shows."""
    h = hashlib.sha256()
    for part in (RENDER_VERSION, plotly.__version__, *parts):
        h.update(_LONG_DECIMALS.sub(r"\1", str(part)).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


class _Hashes:
    """figure name -> {format: hash of what it was saved from}, kept in HASHES: a format is saved again only when its
    hash changed or its file is missing, so unchanged figures are not re-rendered (and their HTML, which plotly gives
    a random id at every save, does not change in git)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.saved = None

    def _load(self):
        if self.saved is None:
            try:
                with open(HASHES, encoding="utf-8") as f:
                    self.saved = json.load(f)
            except (OSError, ValueError):
                self.saved = {}

    def unchanged(self, name, fmt, digest, path) -> bool:
        with self.lock:
            self._load()
            return self.saved.get(name, {}).get(fmt) == digest and os.path.exists(path)

    def record(self, name, fmt, digest):
        with self.lock:
            self._load()
            self.saved.setdefault(name, {})[fmt] = digest

    def write(self):
        with self.lock:
            if self.saved is None:
                return
            os.makedirs(os.path.dirname(HASHES), exist_ok=True)
            with open(HASHES + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self.saved, f, indent=0, sort_keys=True)
            os.replace(HASHES + ".tmp", HASHES)


_hashes = _Hashes()


class _Renderer:
    """PNG and EPS rendered in RENDER_THREADS threads, each with its own kaleido (a Chromium process), while the
    analysis builds the next figures. wait() waits for all of them (also at exit)."""

    def __init__(self):
        self.pool = None
        self.local = threading.local()
        self.futures = []

    def _scope(self):
        if not hasattr(self.local, "scope"):
            from kaleido.scopes.plotly import PlotlyScope
            scope = PlotlyScope()
            scope.plotlyjs = os.path.join(os.path.dirname(os.path.abspath(plotly.__file__)), "package_data",
                                          "plotly.min.js")  # as plotly's own scope
            scope.mathjax = None  # else kaleido bakes a "Loading [MathJax]..." box into PDF/EPS exports
            scope.topojson = TOPOJSON_URL
            self.local.scope = scope
        return self.local.scope

    def _render(self, fig_dict, name, ext, path, copies, width, height, scale, digest):
        try:
            if ext == "png":
                data = self._scope().transform(fig_dict, format="png", width=width, height=height, scale=scale)
                with open(path, "wb") as f:
                    f.write(data)
            else:
                # kaleido's own EPS export runs the same `pdftops -eps`, but under its bundled libraries, which the
                # system's pdftops cannot load on recent Linux ("PDF to EPS conversion failed")
                with open(path + ".pdf", "wb") as f:
                    f.write(self._scope().transform(fig_dict, format="pdf", width=width, height=height))
                subprocess.run(["pdftops", "-eps", path + ".pdf", path], check=True, capture_output=True)
                os.remove(path + ".pdf")
            for copy in copies:
                shutil.copy(path, copy)
            _hashes.record(name, ext, digest)
        except Exception as e:  # one figure that cannot be saved must not stop the others
            logger.error(f"Could not save {ext} image {name}: {e}")

    def submit(self, *args):
        if self.pool is None:
            self.pool = concurrent.futures.ThreadPoolExecutor(RENDER_THREADS)
        self.futures.append(self.pool.submit(self._render, *args))

    def wait(self):
        for future in self.futures:
            future.result()
        self.futures = []
        _hashes.write()


_renderer = _Renderer()
atexit.register(_renderer.wait)


def wait_for_figures():
    """Wait until every figure queued for rendering is saved (and the figure hashes are written)."""
    _renderer.wait()


@functools.lru_cache(maxsize=1)
def _can_save_eps() -> bool:
    """Whether to write EPS: not when config has "save_eps": false (the GitHub Action: EPS files are made on purpose,
    not at every weekly run), and only where poppler's pdftops can convert kaleido's PDF (warns once when not)."""
    if not common.get_configs("save_eps", default=True):
        return False
    if shutil.which("pdftops"):
        return True
    logger.warning("pdftops (poppler-utils) is not installed: EPS figures are not saved.")
    return False


class IO:
    """Saving of plotly figures (HTML, PNG, EPS) to _output and figures/: only formats whose figure changed since they
    were last saved (see _Hashes); PNG and EPS are rendered in the background (call wait_for_figures() at the end)."""

    @staticmethod
    def _open_html() -> bool:
        """Open each saved HTML figure in the browser only if the optional `open_html_figures` config is true."""
        try:
            return bool(common.get_configs("open_html_figures"))
        except KeyError:
            return False

    @staticmethod
    def _static_basemap(fig) -> None:
        """Static images use the map background without place labels: at world scale they mix local languages.
        The interactive HTML keeps them, so place names show when zooming in."""
        if fig.layout.map.style == "carto-positron":
            fig.layout.map.style = "carto-positron-nolabels"
        for layer in fig.layout.map.layers or ():
            layer.source = [src.replace("/light_all/", "/light_nolabels/") for src in layer.source or ()]

    @staticmethod
    def _uses_popups(fig) -> bool:
        """Whether any trace carries popup keys ("@..." hover texts, see hover.py)."""
        for t in fig.data:
            for attr in ("hovertext", "text"):
                v = getattr(t, attr, None)
                if v is not None and not isinstance(v, str) and any(str(k).startswith("@") for k in v):
                    return True
        return False

    def save_plotly_figure(self, fig, filename, width=1600, height=900, scale=1, save_final=True, save_png=True,
                           save_eps=True, post_script=None, html_fig=None, save_html=True):
        """
        Saves a Plotly figure as HTML, PNG and EPS in _output, and copies them to figures/ (`save_final`).

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): width of the PNG and EPS images in pixels. Defaults to 1600.
            height (int, optional): height of the PNG and EPS images in pixels. Defaults to 900.
            scale (int, optional): Scaling factor for the PNG image. Defaults to 1.
            save_final (bool, optional): whether to save the "good" final figure.
            save_png, save_eps, save_html (bool, optional): which formats to save. EPS needs poppler's pdftops and
                is skipped without it.
            post_script (str, optional): JavaScript run after the HTML figure loads (`{plot_id}` is its div id).
            html_fig (plotly.graph_objs.Figure, optional): another figure for the HTML version, e.g. one without the
                labels placed for the static images.
        """
        output_final = os.path.join(common.root_dir, 'figures')
        os.makedirs(common.output_dir, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)
        folders = [common.output_dir] + ([output_final] if save_final else [])

        # where it is checked whether a file is current: figures/ (kept in git) or, without save_final, _output
        final = folders[-1]

        if save_html:
            html_source = html_fig if html_fig is not None else fig
            digest = _digest(html_source.to_json(), post_script, hover.POPUPS_JS, hover.POPUPS_FILE, TOPOJSON_URL)
            if _hashes.unchanged(filename, "html", digest, os.path.join(final, f"{filename}.html")):
                logger.info(f"Html file for {filename} unchanged.")
            else:
                self._save_html(html_source, filename, folders, post_script)
                _hashes.record(filename, "html", digest)

        # static images cannot use interactive menus (e.g., dropdowns), so leave them out
        static = go.Figure(fig)
        static.layout.updatemenus = ()  # assignment: update_layout(updatemenus=[]) would keep the existing menus
        self._static_basemap(static)
        fig_dict, static_json = None, None
        for ext in [e for e, wanted in (("png", save_png), ("eps", save_eps and _can_save_eps())) if wanted]:
            static_json = static_json or static.to_json()
            digest = _digest(static_json, ext, width, height, scale if ext == "png" else None)
            if _hashes.unchanged(filename, ext, digest, os.path.join(final, f"{filename}.{ext}")):
                logger.info(f"{ext.capitalize()} file for {filename} unchanged.")
                continue
            if pio.kaleido.scope is None:  # plotly and kaleido versions that do not match
                logger.error(f"Could not save {ext} image {filename}: kaleido is not available")
                continue
            logger.info(f"Saving {ext} file for {filename}.")
            fig_dict = fig_dict or validate_coerce_fig_to_dict(static, True)
            path = os.path.join(common.output_dir, f"{filename}.{ext}")
            copies = [os.path.join(folder, f"{filename}.{ext}") for folder in folders[1:]]
            _renderer.submit(fig_dict, filename, ext, path, copies, width, height, scale, digest)

    def _save_html(self, fig, filename, folders, post_script):
        """Write the HTML once and copy it to the other folders. plotly.js is not embedded (~4.6 MB per file) but
        written once as plotly.min.js next to the figures and loaded by relative path: htmlpreview.github.io, used
        for the README links, only loads scripts hosted on GitHub, so plotly's CDN would not work there. The HTML
        fills the browser window: a size set for the static images (e.g., for label layout) is dropped."""
        logger.info(f"Saving html file for {filename}.")
        interactive = go.Figure(fig)
        interactive.layout.width = None
        interactive.layout.height = None
        # hover popups given as keys are filled in from popups.js, loaded like plotly.min.js (see hover.py)
        popups = self._uses_popups(interactive)
        if popups:
            post_script = hover.POPUPS_JS + (post_script or "")
        html = interactive.to_html(include_plotlyjs="directory", config={"topojsonURL": TOPOJSON_URL},
                                   post_script=post_script)
        if popups:
            plotly_js = '<script charset="utf-8" src="plotly.min.js"></script>'
            popups_js = f'<script charset="utf-8" src="{hover.POPUPS_FILE}"></script>'
            html = html.replace(plotly_js, plotly_js + popups_js, 1)
        paths = [os.path.join(folder, filename + ".html") for folder in folders]
        with open(paths[0], "w", encoding="utf-8") as f:
            f.write(html)
        for folder, path in zip(folders, paths):
            if path != paths[0]:
                shutil.copy(paths[0], path)
            bundle = os.path.join(folder, "plotly.min.js")  # as write_html(include_plotlyjs="directory") does
            if not os.path.exists(bundle):
                with open(bundle, "w", encoding="utf-8") as f:
                    f.write(get_plotlyjs())
        if self._open_html():
            webbrowser.open("file://" + os.path.abspath(paths[0]))
