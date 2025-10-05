import logging
import sys

class Logger:
    """
    Logger class to handle logging to both file and stdout.
    """
    def __init__(self, name=__name__, log_file="project.log", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent double logging
        self._setup_file_handler(log_file, level)
        self._setup_stream_handler(level)

    def _setup_file_handler(self, log_file, level):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def _setup_stream_handler(self, level):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        stream_handler.setFormatter(stream_formatter)
        self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger
