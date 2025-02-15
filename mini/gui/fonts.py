"""
Module: mini.gui.fonts
Description: Provides a data class and utility functions to locate font files on various operating systems.
"""

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Font:
    """Data class to store font information and provide lookup methods."""

    name: str

    @classmethod
    def font_dirs(cls) -> List[Path]:
        """Return a list of font directories based on the operating system."""
        font_paths = {
            "Linux": cls.linux_font_dirs(),
            "Darwin": cls.darwin_font_dirs(),
            "Windows": cls.windows_font_dirs(),
        }
        return font_paths.get(platform.system(), [])

    @staticmethod
    def linux_font_dirs() -> List[Path]:
        """Return common Linux font directories."""
        return [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path(os.path.expanduser("~/.fonts")),
        ]

    @staticmethod
    def darwin_font_dirs() -> List[Path]:
        """Return common macOS font directories."""
        return [
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts"),
            Path(os.path.expanduser("~/Library/Fonts")),
        ]

    @staticmethod
    def windows_font_dirs() -> List[Path]:
        """Return common Windows font directories."""
        return [
            Path("C:\\Windows\\Fonts"),
            Path(os.path.expandvars("%LOCALAPPDATA%\\Microsoft\\Windows\\Fonts")),
        ]

    @classmethod
    def locate_font(cls, font_name: str) -> Optional[Path]:
        """Locate the font file in common system directories."""
        for font_dir in cls.font_dirs():
            if font_dir.exists():
                for root, _, files in os.walk(font_dir):
                    for file in files:
                        if font_name.lower() in file.lower():
                            return Path(root) / file
        return None

    @classmethod
    def is_font_common(cls, font_name: str, filter_common: bool) -> bool:
        """Check if the filename contains common font names."""
        if not filter_common:
            return True  # Don't filter; accept all fonts.
        common_fonts = {
            "noto",
            "dejavu",
            "arial",
            "times",
            "ubuntu",
            "liberation",
            "fira",
            "roboto",
        }
        return any(f in font_name.lower() for f in common_fonts)

    @classmethod
    def list_fonts(cls, filter_common=True) -> List[str]:
        """List available font files, optionally filtering for common font families."""
        fonts = []
        for font_dir in cls.font_dirs():
            if font_dir.exists():
                for root, _, files in os.walk(font_dir):
                    for file in files:
                        if file.lower().endswith(
                            (".ttf", ".otf")
                        ) and cls.is_font_common(file, filter_common):
                            fonts.append(file)
        return sorted(fonts)  # Sort alphabetically for easier lookup


# Example Usage:
if __name__ == "__main__":
    font_path = Font.locate_font("NotoSansMono-Regular")
    print(f"Font found: {font_path}" if font_path else "Font not found")

    available_fonts = Font.list_fonts()
    print(f"Available Fonts: {available_fonts[:5]}")  # Show first 5 fonts
    print(f"There are {len(available_fonts)} fonts available.")
