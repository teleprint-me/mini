"""
Copyright © 2023 Austin Berrio
Module: mini.gui.settings.window
Description: Settings window for the MiniGUI application.
"""

import dearpygui.dearpygui as dpg

from mini.gui.settings.editor import EditorSettingsTab
from mini.gui.settings.ui import UISettingsTab


class SettingsWindow:
    def __init__(self, gui):
        """Creates the settings window and registers itself with MiniGUI."""
        self.gui = gui
        self.ui_settings = UISettingsTab(gui)  # Instantiate font settings
        self.editor_settings = EditorSettingsTab(gui)
        self.setup_ui()
        self.gui.register_window("settings", existing_window=True)

    def setup_ui(self):
        """Creates the settings window layout with tabbed categories."""
        with dpg.window(
            label="Settings",
            tag="settings",
            show=False,
            pos=(260, 10),
            width=400,
            height=300,
        ):
            with dpg.tab_bar():
                with dpg.tab(label="UI"):
                    self.ui_settings.create_ui_settings()
                with dpg.tab(label="Editor"):
                    self.editor_settings.create_editor_settings()
                with dpg.tab(label="Tokenizer"):
                    dpg.add_text("Tokenizer Settings Placeholder")
                with dpg.tab(label="Trainer"):
                    dpg.add_text("Trainer Settings Placeholder")
                with dpg.tab(label="Generator"):
                    dpg.add_text("Generator Settings Placeholder")
                with dpg.tab(label="Graph"):
                    dpg.add_text("Graph Settings Placeholder")
