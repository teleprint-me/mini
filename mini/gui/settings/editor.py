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
        self.linter = self.find_linter_path()
        self.formatter = self.find_formatter_path()
        self.compiler = self.find_compiler_path()

    def find_executable(self, candidates):
        """Finds an available executable from a list of possible paths."""
        for exe in candidates:
            if shutil.which(exe):
                return exe
        return ""  # Return empty if none are found

    def find_linter_path(self):
        """Sets the linter path."""
        return self.find_executable([".venv/bin/flake8", "flake8"])

    def find_formatter_path(self):
        """Sets the formatter path."""
        return self.find_executable([".venv/bin/black", "black"])

    def find_compiler_path(self):
        """Sets the compiler path."""
        return self.find_executable([".venv/bin/python", "python3"])

    def validate_path(self, app_data, label_tag, attribute_name):
        """Validates a given path and updates the UI accordingly."""
        if shutil.which(app_data):
            # Update the attribute with the valid path
            setattr(self, attribute_name, app_data)
            # Reset color to default if valid
            dpg.configure_item(label_tag, color=(255, 255, 255))
            self.logger.info(
                f"{attribute_name.replace('_', ' ').capitalize()} set to: {app_data}"
            )
        else:
            dpg.configure_item(label_tag, color=(255, 0, 0))  # Highlight as invalid
            setattr(self, attribute_name, "")  # Reset to empty if invalid

    def set_linter_path(self, sender, app_data):
        """Sets the linter path."""
        self.validate_path(app_data, "editor_linter_label", "linter")

    def set_formatter_path(self, sender, app_data):
        """Sets the formatter path."""
        self.validate_path(self, app_data, "editor_formatter_label", "formatter")

    def set_compiler_path(self, sender, app_data):
        """Sets the compiler / interpreter path."""
        self.validate_path(app_data, "editor_compiler_label", "compiler")

    def create_editor_settings(self):
        """Creates the Editor settings tab."""
        dpg.add_text("Set Linter Path:", tag="editor_linter_label")
        dpg.add_input_text(
            default_value=self.linter,
            callback=self.set_linter_path,
            tag="editor_linter",
            hint=self.find_linter_path(),
        )

        dpg.add_text("Set Formatter Path:", tag="editor_formatter_label")
        dpg.add_input_text(
            default_value=self.formatter,
            callback=self.set_formatter_path,
            tag="editor_formatter",
            hint=self.find_formatter_path(),
        )

        dpg.add_text("Set Compiler / Interpreter:", tag="editor_compiler_label")
        dpg.add_input_text(
            default_value=self.compiler,
            callback=self.set_compiler_path,
            tag="editor_compiler",
            hint=self.find_compiler_path(),
        )

        dpg.add_text("Compiler Explanation:")
        dpg.add_text(
            "For interpreted languages, this field represents the runtime environment.",
            wrap=300,
        )
