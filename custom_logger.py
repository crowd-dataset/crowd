import logging


class CustomLogger:
    """Logger that handles string formatting.

    Contains a logging.Logger object. Copies the various logging.Logger
    methods. The purpose is to accept str.format() style formatting.
    With this custom class, messages may contain '{}' where the next
    arguments will be placed. Doesn't work with keyword arguments for the
    formatting. A message without arguments is logged as it is, so an
    f-string may contain braces.

    Examples
    --------
    >>> CustomLogger(__name__).info('{} videos', 3)
    """

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(5)

    def debug(self, msg, *args, **kwargs):
        """Log msg.format(*args) at DEBUG level."""
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log msg.format(*args) at INFO level."""
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log msg.format(*args) at WARNING level."""
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log msg.format(*args) at ERROR level."""
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log msg.format(*args) at CRITICAL level."""
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """Log msg at level, formatted with args when there are any; kwargs (exc_info, ...) go to logging."""
        if self.logger.isEnabledFor(level):
            if args:
                msg = msg.format(*args)
            self.logger._log(level, msg, args=(), **kwargs)
