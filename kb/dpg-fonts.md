---
title: 'Blurry Font Rendering in Dear PyGui (DPG)'
type: 'issue'
version: 1
date: '2025-02-16'
modified: '2025-02-16'
license: 'cc-by-nc-sa-4.0'
---

# **Blurry Font Rendering in Dear PyGui (DPG)**

## **Description**

When dynamically updating fonts in **Dear PyGui (DPG)**, users may experience
**blurriness**, especially after modifying font size or switching between fonts.
This issue appears to be related to **how DPG handles font rendering and
scaling**, particularly on high-DPI displays.

## **Symptoms**

- Fonts appear **sharp on initial load** but become **blurry after switching**.
- The issue **affects all font families equally**.
- Adjusting the **global font scale** improves readability but does **not fully
  restore the original clarity**.
- Behavior **varies across operating systems and display settings**.
- **HiDPI settings in SDL-based backends** may exacerbate the problem.

## **Workarounds & Solutions**

### **1. Adjusting Font Scaling**

A common workaround is to **load fonts at double the intended size** and then
use **global font scaling** to counteract the blurring.

```python
import dearpygui.dearpygui as dpg

def set_font(font_path, font_size):
    """Applies a new font dynamically with scaling adjustments."""
    with dpg.font_registry():
        dpg.add_font(font_path, font_size * 2, tag="active_font")
        dpg.bind_font("active_font")
        dpg.set_global_font_scale(0.5)  # Reduces scaling artifacts
```

**Limitations**:

- This mitigates the blurriness **but does not fully restore the original
  sharpness**.
- The exact **scaling factor may vary** depending on the OS and backend.

### **2. Rebuilding DearPyGui from Source (Advanced)**

If a more **permanent fix** is needed, modifying and recompiling **DearPyGui**
can improve font rendering.

**Steps (Windows Example)**

1. Clone the **DearPyGui** repository.
2. Modify **imgui_impl_dx11.cpp**:
   - Replace:
     ```cpp
     D3D11_FILTER_MIN_MAG_MIP_LINEAR
     ```
   - With:
     ```cpp
     D3D11_FILTER_MIN_MAG_MIP_POINT
     ```
3. Build DearPyGui using:
   ```sh
   python setup.py build
   ```
4. Replace the compiled **\_dearpygui.pyd** file in your **site-packages**
   directory.

### **3. Disabling HiDPI in SDL (Experimental)**

DearPyGui uses **SDL for rendering**. Some users reported that **HiDPI scaling
in SDL can contribute to blurry text**. Disabling HiDPI scaling in the **SDL
backend** may help.

## **Current Status**

- **No official fix from DPG** as of 2025.
- **HiDPI scaling behavior remains inconsistent**.
- **Font scaling trick (Option 1) provides the best balance between usability
  and ease of implementation.**

**Final Verdict:**  
✅ **Use the font scaling trick for now.**  
🚧 **Wait for future updates or explore SDL settings for fine-tuning.**

### **Summary**

While **there is no perfect fix**, the **font scaling trick** helps **reduce
blurriness**. If absolute clarity is needed, **rebuilding DPG** is an option,
but it requires significant setup. **Future updates may improve this issue**, so
keep an eye on **DearPyGui’s development**.

### **Related References**

- [GitHub Issue #1380 - Font Scaling Bug](https://github.com/hoffstadt/DearPyGui/issues/1380)
- [GitHub Issue #773 - Rebuilding DearPyGui for Fix](https://github.com/hoffstadt/DearPyGui/issues/773)
- [GitHub Issue #4768 - SDL Backend HiDPI Issues](https://github.com/ocornut/imgui/issues/4768)
