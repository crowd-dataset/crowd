import functools
import os
import shutil
import subprocess
import webbrowser

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

import common
from custom_logger import CustomLogger
from utils.plotting import hover
from utils.plotting.constants import TOPOJSON_URL

logger = CustomLogger(__name__)  # use custom logger

# Without this, kaleido bakes a "Loading [MathJax]..." box into PDF/EPS exports; no figure uses LaTeX.
pio.kaleido.scope.mathjax = None
pio.kaleido.scope.topojson = TOPOJSON_URL  # see constants.TOPOJSON_URL


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
    """Saving of plotly figures (HTML, PNG, EPS) to _output and figures/."""

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

        if save_html:
            self._save_html(html_fig if html_fig is not None else fig, filename, folders, post_script)

        # static images cannot use interactive menus (e.g., dropdowns), so leave them out
        static = go.Figure(fig)
        static.layout.updatemenus = ()  # assignment: update_layout(updatemenus=[]) would keep the existing menus
        self._static_basemap(static)
        for ext in [e for e, wanted in (("png", save_png), ("eps", save_eps and _can_save_eps())) if wanted]:
            logger.info(f"Saving {ext} file for {filename}.")
            path = os.path.join(common.output_dir, f"{filename}.{ext}")
            try:
                if ext == "png":
                    static.write_image(path, width=width, height=height, scale=scale)
                else:
                    # kaleido's own EPS export runs the same `pdftops -eps`, but under its bundled libraries, which
                    # the system's pdftops cannot load on recent Linux ("PDF to EPS conversion failed")
                    static.write_image(path + ".pdf", width=width, height=height)
                    subprocess.run(["pdftops", "-eps", path + ".pdf", path], check=True, capture_output=True)
                    os.remove(path + ".pdf")
            except (ValueError, subprocess.CalledProcessError) as e:
                logger.error(f"Could not save {ext} image {filename}: {e}")
                return
            for folder in folders[1:]:
                shutil.copy(path, os.path.join(folder, f"{filename}.{ext}"))

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
