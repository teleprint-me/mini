"""
Copyright © 2023 Austin Berrio
Module: mini.gui.fonts.fuzzer
Description: Utility for fuzzing font files.
"""

import os
import platform
import re
from difflib import get_close_matches
from itertools import chain
from pathlib import Path
from typing import List, Optional


class FontFuzzer:
    """Provides font lookup and management functions."""

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
    def normalize_font_name(cls, font_name: str) -> str:
        """Normalize font names by removing spaces, dashes, and converting to lowercase."""
        return re.sub(r"[\s_-]+", "", font_name).lower()

    @classmethod
    def list_fonts(
        cls, filter_common=True, search_query: Optional[str] = None
    ) -> List[str]:
        """List available fonts, optionally filtering by common families or search query."""
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
        fonts = []
        search_query = cls.normalize_font_name(search_query) if search_query else None

        for font_dir in cls.font_dirs():
            if font_dir.exists():
                for root, _, files in os.walk(font_dir):
                    for file in files:
                        if file.lower().endswith((".ttf", ".otf")):
                            normalized_file = cls.normalize_font_name(file)
                            if search_query and search_query not in normalized_file:
                                continue  # Skip if query is specified and doesn't match

                            if not filter_common or any(
                                f in normalized_file for f in common_fonts
                            ):
                                fonts.append(file)

        return sorted(fonts)  # Sort alphabetically for easier lookup

    @classmethod
    def locate_font_directly(cls, font_file: str) -> Optional[Path]:
        """Find the full path of a font file by name."""
        for font_dir in cls.font_dirs():
            if font_dir.exists():
                for root, _, files in os.walk(font_dir):
                    if font_file in files:
                        return Path(root) / font_file
        return None

    @classmethod
    def locate_font(
        cls, font_name: str, n: int = 3, cutoff: float = 0.4
    ) -> Optional[Path]:
        """Locate the best matching font file using fuzzy searching and fallback substring matching."""
        normalized_font = cls.normalize_font_name(font_name)

        # Check bundled fonts in mini/gui/assets/
        asset_font_dir = Path(__file__).parent / "assets"
        assets = chain(asset_font_dir.glob("*.ttf"), asset_font_dir.glob("*.otf"))
        for font in assets:
            if cls.normalize_font_name(font_name) in cls.normalize_font_name(font.name):
                return font

        # Get the list of available font files (no filtering)
        available_fonts = cls.list_fonts(filter_common=False)

        # Apply fuzzy matching
        best_matches = get_close_matches(
            normalized_font, available_fonts, n=n, cutoff=cutoff
        )

        # If no fuzzy match found, fallback to substring search
        if not best_matches:
            best_matches = [
                f
                for f in available_fonts
                if normalized_font in cls.normalize_font_name(f)
            ]

        # Return the first best match, if found
        if best_matches:
            return cls.locate_font_directly(best_matches[0])

        return None  # No good match found


# CLI Interface for Testing
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Font Management CLI Tool")
    parser.add_argument("--locate", type=str, help="Name of the font to locate")
    parser.add_argument(
        "--n", type=int, default=3, help="Number of matches to return (default: 3)"
    )
    parser.add_argument(
        "--cutoff", type=float, default=0.4, help="Fuzzy matching cutoff (default: 0.4)"
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter fonts for common families (default: False)",
    )
    parser.add_argument(
        "--list", nargs="?", const=True, help="List available fonts (optional filter)"
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=5,
        help="Limit the number of fonts displayed (default: 5)",
    )

    args = parser.parse_args()

    if args.list:
        available_fonts = FontFuzzer.list_fonts(
            filter_common=args.filter,
            search_query=args.list if args.list is not True else None,
        )
        print(f"There are {len(available_fonts)} fonts available.")
        for font in available_fonts[: args.clip]:
            print(font)
    elif args.locate:
        font_path = FontFuzzer.locate_font(args.locate, n=args.n, cutoff=args.cutoff)
        print(
            f"Font found: {font_path}"
            if font_path
            else f"Font not found: {args.locate}"
        )
    else:
        print(
            "No action specified. Use --list or --locate. Use --help for more information."
        )
        exit(1)
