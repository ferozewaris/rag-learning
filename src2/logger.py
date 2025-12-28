"""Logger Module."""
import logging
import sys


class Logger:
    """Customized logger class to log messages."""

    _logger = None

    def __new__(cls, *args, **kwargs):
        """Singleton Implementation of Logger."""
        if cls._logger is None:
            cls._logger = cls.app_log(*args, **kwargs)
        return cls._logger

    @staticmethod
    def app_log(
        level="DEBUG",
        logger_name="Chat BOT",
        log_file="st.log",
    ):
        """
        To initiate the logger.

        :param level: Log level
        :param logger_name: Name of the logger
        :param log_file: Filepath in which logs are to be stored
        :return: logger object
        """
        logger = logging.getLogger(logger_name)
        log_format = logging.Formatter(
            "%(asctime)s — %(name)s — %(module)s — %(levelname)s — %(message)s"
        )
        log_level = logging.getLevelName(level)
        logger.setLevel(log_level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        return logger