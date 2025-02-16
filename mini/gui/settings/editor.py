"""
Copyright © 2023 Austin Berrio
Module: mini.gui.settings.editor
Description: UI editor settings for the settings window.
"""

import logging
from pathlib import Path

import dearpygui.dearpygui as dpg

from mini.common.logger import get_logger
from mini.gui.fonts.fuzzer import FontFuzzer


class EditorSettingsTab:
    """Settings window for the MiniGUI application."""

    def __init__(self, gui):
        """Creates the settings window and registers itself with MiniGUI."""
        self.gui = gui
        self.linter = ""
        self.formatter = ".venv/bin/black"
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG)

    def create_editor_settings(self):
        """Creates the Editor settings tab."""
        dpg.add_text("Set linter path:")
        dpg.add_input_text(label="Linter Path")
