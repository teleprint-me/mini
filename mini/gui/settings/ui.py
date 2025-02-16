"""
Copyright © 2023 Austin Berrio
Module: mini.gui.settings.ui
Description: UI tab settings for the settings window.
"""

import logging
from pathlib import Path

import dearpygui.dearpygui as dpg

from mini.common.logger import get_logger
from mini.gui.fonts.fuzzer import FontFuzzer


class UISettingsTab:
    """UI settings tab for the MiniGUI application."""

    def __init__(self, gui):
        """Creates the settings window and registers itself with MiniGUI."""
        self.gui = gui
        self.font_path = FontFuzzer.locate_font("Noto Sans Mono")
        self.font_size = 16
        self.font_size_multiplier = 2
        self.font_scale_factor = 0.5
        self.font_options = []  # Store available fonts
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG)

    def create_ui_settings(self):
        """Creates the UI settings tab, including font selection."""
        dpg.add_text("Select Font:")

        self.font_options = FontFuzzer.list_fonts(filter_common=True)
        dpg.add_combo(
            [Path(f).stem for f in self.font_options],
            callback=self.set_font_type,
            default_value=Path(self.font_path).stem,
        )

        # NOTE: min_value and max_value are not respected.
        dpg.add_text("Font Size:")
        dpg.add_input_int(
            callback=self.set_font_size,
            default_value=self.font_size,
        )

        dpg.add_text("Font Size Multiplier:")
        dpg.add_input_int(
            callback=self.set_font_size_multiplier,
            default_value=self.font_size_multiplier,
        )

        dpg.add_text("Global Font Scale Factor:")
        dpg.add_input_float(
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
        clamped_size = max(8, min(app_data, 72))  # Clamp between 8 and 72
        self.set_font(font_size=clamped_size)

    def set_font_size_multiplier(self, sender, app_data):
        """Updates the UI font size multiplier."""
        clamped_multiplier = max(0, min(app_data, 5))  # Clamp between 0 and 5
        self.set_font(font_size_multiplier=clamped_multiplier)

    def set_font_scale_factor(self, sender, app_data):
        """Updates the UI font scale."""
        clamped_scale = max(0.1, min(app_data, 1.0))  # Clamp between 0.1 and 1.0
        self.set_font(font_scale_factor=clamped_scale)

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
        self.logger.debug(
            f"Font changed to: {Path(self.font_path).stem}, "
            f"Size: {self.font_size} x {self.font_size_multiplier} = {adjusted_font_size}, "
            f"Scale: {self.font_scale_factor:.2f}"
        )
