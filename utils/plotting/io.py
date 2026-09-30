import os
import common
import plotly.graph_objects as go
import plotly.io as pio
import shutil
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger

# Without this, kaleido bakes a "Loading [MathJax]..." box into PDF/EPS exports; no figure uses LaTeX.
pio.kaleido.scope.mathjax = None


class IO:
    def __init__(self) -> None:
        pass

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

    def save_plotly_figure(self, fig, filename, width=1600, height=900, scale=1, save_final=True, save_png=True,
                           save_eps=True, post_script=None):
        """
        Saves a Plotly figure as HTML, PNG, SVG, and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): width of the PNG and EPS images in pixels. Defaults to 1600.
            height (int, optional): height of the PNG and EPS images in pixels. Defaults to 900.
            scale (int, optional): Scaling factor for the PNG image. Defaults to 3.
            save_final (bool, optional): whether to save the "good" final figure.
            post_script (str, optional): JavaScript run after the HTML figure loads (`{plot_id}` is its div id).
        """
        # Create directory if it doesn't exist
        output_final = os.path.join(common.root_dir, 'figures')
        os.makedirs(common.output_dir, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)

        # Save as HTML. plotly.js is not embedded (~4.6 MB per file) but written once as plotly.min.js next to the
        # figures and loaded by relative path: htmlpreview.github.io, used for the README links, only loads scripts
        # hosted on GitHub, so plotly's CDN would not work there.
        # The HTML fills the browser window: a size set for the static images (e.g., for label layout) is dropped.
        logger.info(f"Saving html file for {filename}.")
        interactive = go.Figure(fig)
        interactive.layout.width = None
        interactive.layout.height = None
        interactive.write_html(os.path.join(common.output_dir, filename + ".html"), include_plotlyjs="directory",
                               auto_open=self._open_html(), post_script=post_script)
        # also save the final figure
        if save_final:
            interactive.write_html(os.path.join(output_final, filename + ".html"), include_plotlyjs="directory",
                                   post_script=post_script)

        # static images cannot use interactive menus (e.g., dropdowns), so leave them out
        static = go.Figure(fig)
        static.layout.updatemenus = ()  # assignment: update_layout(updatemenus=[]) would keep the existing menus
        self._static_basemap(static)

        try:
            # Save as PNG
            if save_png:
                logger.info(f"Saving png file for {filename}.")
                static.write_image(os.path.join(common.output_dir, filename + ".png"), width=width, height=height,
                                   scale=scale)
                # also save the final figure
                if save_final:
                    shutil.copy(os.path.join(common.output_dir, filename + ".png"),
                                os.path.join(output_final, filename + ".png"))

            # Save as EPS
            if save_eps:
                logger.info(f"Saving eps file for {filename}.")
                static.write_image(os.path.join(common.output_dir, filename + ".eps"), width=width, height=height)
                # also save the final figure
                if save_final:
                    shutil.copy(os.path.join(common.output_dir, filename + ".eps"),
                                os.path.join(output_final, filename + ".eps"))
        except ValueError as e:
            logger.error(f"Value error raised when attempted to save image {filename}: {e}")
