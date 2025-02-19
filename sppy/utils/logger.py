"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/19 17:34
Description: 
    

"""

import logging


class Logger:

    def __init__(
        self,
        num_slots: int = 50,
        log_to_console: bool = True,
        log_to_file: bool = False,
        file_name: str = "",
        directory="logs/",
    ):
        self.num_slots = num_slots
        file_logger = logging.getLogger("file_logger")
        console_logger = logging.getLogger("console_only_logger")
        file_logger.setLevel(logging.DEBUG)  # Record logs of all levels

        if log_to_file:
            log_file = directory + file_name  # the full directory of the log file
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            # Define the log formatter in the file log
            formatter_file_head = logging.Formatter("%(asctime)s - %(message)s")
            file_handler.setFormatter(formatter_file_head)
            file_logger.addHandler(file_handler)  # Add the handler to loggers

        # Set up the handlers
        if log_to_console:
            # Create a separate logger for console-only messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_logger.addHandler(console_handler)  # Only attach console handler

        self.file_logger = file_logger
        self.console_logger = console_logger

    def console_header_sddp(self):
        num_slots = self.num_slots
        author = "Dr. Zhen Chen"
        self.console_logger.info(f"{author:-^{num_slots}}")
        str1 = "Iteration"
        str2 = "Objective value"
        width1 = 20
        width2 = num_slots - width1
        self.console_logger.info(f"{str1:<{width1}}{str2:>{width2}}")
        self.console_logger.info("-" * num_slots)

    def console_body_sddp(self, iteration, objective_value):
        width1 = 20
        width2 = self.num_slots - width1
        self.console_logger.info(
            f"{iteration:<{width1}}{objective_value:>{width2}.4f}."
        )

    def file_header_sddp(self, message):
        self.file_logger.info(message)

    def file_body_sddp(self, message):
        formatter_file_body= logging.Formatter("%(message)s")
        for handler in self.file_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setFormatter(formatter_file_body)


if __name__ == "__main__":
    logger = Logger()
    logger.console_header_sddp()
    logger.console_body_sddp(iteration=100, objective_value=100)
