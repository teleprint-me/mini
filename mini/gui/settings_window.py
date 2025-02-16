from pathlib import Path

import dearpygui.dearpygui as dpg

from mini.gui.fonts.fuzzer import FontFuzzer


class SettingsWindow:
    def __init__(self, gui):
        """Creates the settings window and registers itself with MiniGUI."""
        self.gui = gui
        self.font_options = []  # Store available fonts
        self.selected_font = None

        self.setup_ui()
        self.gui.register_window(
            "settings", existing_window=True
        )  # Tell MiniGUI to use this window!

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
                    self.create_ui_settings()
                with dpg.tab(label="Editor"):
                    dpg.add_text("Editor Settings Placeholder")
                with dpg.tab(label="Tokenizer"):
                    dpg.add_text("Tokenizer Settings Placeholder")
                with dpg.tab(label="Trainer"):
                    dpg.add_text("Trainer Settings Placeholder")
                with dpg.tab(label="Generator"):
                    dpg.add_text("Generator Settings Placeholder")
                with dpg.tab(label="Graph"):
                    dpg.add_text("Graph Settings Placeholder")

    def create_ui_settings(self):
        """Creates the UI settings tab, including font selection."""
        dpg.add_text("Select Font:")
        self.font_options = FontFuzzer.list_fonts(filter_common=True)
        self.selected_font = dpg.add_combo(
            [Path(f).stem for f in self.font_options],  # Just show font names
            callback=self.set_font,
            default_value=Path(self.gui.current_font).stem,
        )

    def set_font(self, sender, app_data):
        """Updates the UI font."""
        selected_font = FontFuzzer.locate_font(app_data)
        if selected_font:
            self.gui.set_font(selected_font)
            print(f"Font changed to: {selected_font}")
