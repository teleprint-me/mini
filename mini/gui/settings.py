"""
Module: mini.gui.settings
Description: Settings window for the MiniGUI application.
"""

from pathlib import Path

import dearpygui.dearpygui as dpg

from mini.gui.fonts.fuzzer import FontFuzzer


class UISettingsWindow:
    """Settings window for the MiniGUI application."""

    def __init__(self, gui):
        """Creates the settings window and registers itself with MiniGUI."""
        self.gui = gui
        self.font_path = FontFuzzer.locate_font("Noto Sans Mono")
        self.font_size = 16
        self.font_size_multiplier = 2
        self.font_scale_factor = 0.5
        self.font_options = []  # Store available fonts

    def create_ui_settings(self):
        """Creates the UI settings tab, including font selection."""
        dpg.add_text("Select Font:")

        self.font_options = FontFuzzer.list_fonts(filter_common=True)
        dpg.add_combo(
            [Path(f).stem for f in self.font_options],
            callback=self.set_font_type,
            default_value=Path(self.font_path).stem,
        )

        dpg.add_text("Font Size:")
        dpg.add_input_int(
            min_value=8,
            max_value=72,
            callback=self.set_font_size,
            default_value=self.font_size,
        )

        dpg.add_text("Font Size Multiplier:")
        dpg.add_input_int(
            min_value=1,
            max_value=5,
            callback=self.set_font_size_multiplier,
            default_value=self.font_size_multiplier,
        )

        dpg.add_text("Global Font Scale Factor:")
        dpg.add_input_float(
            min_value=0.1,
            max_value=2.0,
            step=0.1,
            format="%.1f",
            callback=self.set_font_scale_factor,
            default_value=self.font_scale_factor,
        )

    def set_font_type(self, sender, app_data):
        """Updates the UI font."""
        selected_font = FontFuzzer.locate_font(app_data)
        if selected_font:
            self.set_font(font_path=selected_font)  # Use the fixed function!

    def set_font_size(self, sender, app_data):
        """Updates the UI font size."""
        self.set_font(font_size=app_data)

    def set_font_size_multiplier(self, sender, app_data):
        """Updates the UI font size multiplier."""
        self.set_font(font_size_multiplier=app_data)

    def set_font_scale_factor(self, sender, app_data):
        """Updates the UI font scale."""
        self.set_font(font_scale_factor=app_data)

    def set_font(
        self,
        font_path=None,
        font_size=None,
        font_size_multiplier=None,
        font_scale_factor=None,
    ):
        """Applies a new font dynamically and prevents blurriness."""
        # Default to current values if not provided
        font_path = font_path or self.font_path
        font_size = font_size or self.font_size
        font_size_multiplier = font_size_multiplier or self.font_size_multiplier
        font_scale_factor = font_scale_factor or self.font_scale_factor

        # Avoid redundant updates
        if (
            font_path == self.font_path
            and font_size == self.font_size
            and font_size_multiplier == self.font_size_multiplier
            and font_scale_factor == self.font_scale_factor
        ):
            return  # No need to reapply the same font

        # Store updated values
        self.font_path = font_path
        self.font_size = font_size
        self.font_size_multiplier = font_size_multiplier
        self.font_scale_factor = font_scale_factor

        # 🔥 Fix: Preserve DearPyGui's default scaling method
        adjusted_font_size = font_size * font_size_multiplier  # Multiply font size
        scale_factor = font_scale_factor  # Apply final scaling

        # Remove existing fonts
        if dpg.does_item_exist("font_registry"):
            dpg.delete_item("font_registry")

        # Apply new font
        with dpg.font_registry(tag="font_registry"):
            dpg.add_font(self.font_path, adjusted_font_size, tag="active_font")
            dpg.bind_font("active_font")
            dpg.set_global_font_scale(scale_factor)  # Avoid scaling artifacts

        # Log the change
        print(
            f"Font changed to: {Path(self.font_path).stem}, "
            f"Size: {self.font_size} x {self.font_size_multiplier} = {adjusted_font_size}, "
            f"Scale: {self.font_scale_factor}"
        )


class SettingsWindow:
    def __init__(self, gui):
        """Creates the settings window and registers itself with MiniGUI."""
        self.gui = gui
        self.ui_settings = UISettingsWindow(gui)  # Instantiate font settings
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
                    self.ui_settings.create_ui_settings()  # Register font settings
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
