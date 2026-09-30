import os
import common
import plotly as py
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

    def save_plotly_figure(self, fig, filename, width=1600, height=900, scale=1, save_final=True, save_png=True,
                           save_eps=True):
        """
        Saves a Plotly figure as HTML, PNG, SVG, and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): width of the PNG and EPS images in pixels. Defaults to 1600.
            height (int, optional): height of the PNG and EPS images in pixels. Defaults to 900.
            scale (int, optional): Scaling factor for the PNG image. Defaults to 3.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # Create directory if it doesn't exist
        output_final = os.path.join(common.root_dir, 'figures')
        os.makedirs(common.output_dir, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)

        # Save as HTML. plotly.js is not embedded (~4.6 MB per file) but written once as plotly.min.js next to the
        # figures and loaded by relative path: htmlpreview.github.io, used for the README links, only loads scripts
        # hosted on GitHub, so plotly's CDN would not work there.
        logger.info(f"Saving html file for {filename}.")
        py.offline.plot(fig, filename=os.path.join(common.output_dir, filename + ".html"),
                        include_plotlyjs="directory", auto_open=self._open_html())
        # also save the final figure
        if save_final:
            py.offline.plot(fig, filename=os.path.join(output_final, filename + ".html"), auto_open=False,
                            include_plotlyjs="directory")

        # static images cannot use interactive menus (e.g., dropdowns), so leave them out
        static = go.Figure(fig)
        static.layout.updatemenus = ()  # assignment: update_layout(updatemenus=[]) would keep the existing menus

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
