"""Contain function to display or store logging messages."""
import logging
import sys
from typing import Union, Optional


def logs(
        show_level: Optional[Union[int, str]] = None,
        save_level: Optional[Union[int, str]] = None,
        program_name: Optional[str] = None,
        path: Optional[str] = None,
        threads: bool = False,
        multiproc: bool = False,
        show_color: bool = True
) -> None:
    """
    Initialize the root logger: messages of show_level and above go to stdout.

    Parameters
    ----------
    show_level : int or {'debug', 'info', 'warning', 'error', 'critical'},
        optional. Pass a log level to display logs of that level and above.
    save_level, program_name, path : ignored
        Kept for old callers: logs are no longer saved to a file (no caller did).
    threads : bool, default False
        Add the thread name to log messages. Useful when using threading.
    multiproc : bool, default False
        Add the process name to log messages. Useful when using
        multiprocessing.
    show_color : bool, default True
        If you have the coloredlogs package installed the messages will be
        colored.

    Note that log levels can be one of the listed strings or an integer between
    1 and 100. If you want to get all possible log messages, use a log level of
    1.
    """
    logger_root = logging.getLogger()
    fmt_items = ('%(asctime)s',
                 '%(levelname)-8s',
                 '%(threadName)s' if threads else None,
                 '%(processName)s' if multiproc else None,
                 '%(name)s',
                 '%(message)s')
    fmt = ' - '.join((item for item in fmt_items if item is not None))
    formatter = logging.Formatter(fmt)
    logging.addLevelName(5, "VERBOSE")
    logger_root.setLevel(5)

    if show_level and show_color:
        try:
            import coloredlogs  # type: ignore
        except ImportError:
            show_color = False
        else:
            coloredlogs.install(fmt=fmt,
                                level=_convert_logging_level(show_level),
                                stream=sys.stdout)
    if show_level and not show_color:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(_convert_logging_level(show_level))
        stream_handler.setFormatter(formatter)
        logger_root.addHandler(stream_handler)
    _logging_level_threshold()


def _logging_level_threshold():
    """
    Set the level threshold for a couple of internal and external modules.
    """
    for mod_name in ['requests', 'matplotlib', 'numexpr.utils',
                     'urllib3.connectionpool', 'PIL.TiffImagePlugin']:
        logging.getLogger(mod_name).setLevel(logging.WARNING)


def _convert_logging_level(level: Union[int, str]) -> int:
    """Convert the user-provided level to a logging level integer."""
    if isinstance(level, int):
        assert 0 < level <= 50, 'Logging level must be between 0 and 50.'
        return level
    if not hasattr(logging, level.upper()):
        raise ValueError('Unknown logging level name: {}.'.format(level))
    return getattr(logging, level.upper())
