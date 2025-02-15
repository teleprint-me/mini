"""
Module: mini.args.base
Description: Base class for command-line argument parsing.
"""

from argparse import ArgumentParser, Namespace


class BaseArgs:
    def __init__(self, description: str = "Mini CLI Tool"):
        self.parser = ArgumentParser(description=description)
        self.parser.add_argument(
            "-v", "--verbose", action="store_true", help="Enable verbose mode"
        )

    def parse_args(self) -> Namespace:
        raise NotImplementedError("Subclasses must implement parse_args method")
