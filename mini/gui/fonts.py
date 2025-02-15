import os
import platform


def find_font_path(font_name, return_all=False):
    """Search for a font file by name across common font directories.

    Args:
        font_name (str): The name (or part of it) of the font file.
        return_all (bool): If True, return all matches instead of the first.

    Returns:
        str or list: Absolute path(s) to the font file(s) if found, else None.
    """

    # Define common font directories for different platforms
    font_dirs = []
    system = platform.system()

    if system == "Linux":
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
        ]
    elif system == "Darwin":  # macOS
        font_dirs = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    elif system == "Windows":
        font_dirs = [
            "C:\\Windows\\Fonts",
            os.path.expandvars("%LOCALAPPDATA%\\Microsoft\\Windows\\Fonts"),
        ]

    matches = []

    for font_dir in font_dirs:
        if os.path.exists(font_dir):  # Ensure the directory exists before searching
            for root, _, files in os.walk(font_dir):
                for file in files:
                    if font_name.lower() in file.lower():  # Case-insensitive match
                        matches.append(os.path.join(root, file))
                        if not return_all:
                            return matches[0]  # Return first match immediately

    return matches if return_all else None


# Example usage
font_path = find_font_path("NotoSansMono-Regular")
print(f"Font found: {font_path}" if font_path else "Font not found")
