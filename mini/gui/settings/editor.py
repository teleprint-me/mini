"""
Copyright © 2023 Austin Berrio
Module: mini.gui.settings.editor
Description: UI editor settings for the settings window.
"""

import logging
import shutil

import dearpygui.dearpygui as dpg

from mini.common.logger import get_logger


class EditorSettingsTab:
    """Settings window for the MiniGUI application."""

    def __init__(self, gui):
        """Creates the settings window and registers itself with MiniGUI."""
        self.gui = gui
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG)

        # Default Paths (if available)
        self.linter = self.find_executable([".venv/bin/flake8", "flake8"])
        self.formatter = self.find_executable([".venv/bin/black", "black"])
        self.compiler = self.find_executable([".venv/bin/python", "python3"])

    def find_executable(self, candidates):
        """Finds an available executable from a list of possible paths."""
        for exe in candidates:
            if shutil.which(exe):
                return exe
        return ""  # Return empty if none are found

    def create_editor_settings(self):
        """Creates the Editor settings tab."""
        dpg.add_text("Set Linter Path:")
        dpg.add_input_text(
            label="Linter Path", default_value=self.linter, tag="editor_linter"
        )

        dpg.add_text("Set Formatter Path:")
        dpg.add_input_text(
            label="Formatter Path", default_value=self.formatter, tag="editor_formatter"
        )

        dpg.add_text("Set Compiler / Interpreter:")
        dpg.add_input_text(
            label="Compiler / Interpreter",
            default_value=self.compiler,
            tag="editor_compiler",
        )

        dpg.add_text("Compiler Explanation:")
        dpg.add_text(
            "For interpreted languages, this field represents the runtime environment.",
            wrap=300,
        )
