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

When dynamically updating fonts in Dear PyGui, users may experience blurriness,
particularly when modifying font size or switching between fonts. This issue
persists across various systems and appears to be related to how DPG handles
font rendering and scaling.

## **Observed Behavior**

- Fonts appear sharp on initial load.
- Changing the font size or type causes blurriness.
- The issue affects all font families equally.
- Adjusting the global font scale improves but does not completely resolve the
  issue.

## **Known Workarounds**

### **1. Adjusting Font Size & Scaling**

A commonly recommended workaround involves loading the font at double the
desired size and then scaling it down.

```python
import dearpygui.dearpygui as dpg

def set_font(font_path, font_size):
    with dpg.font_registry():
        new_font = dpg.add_font(font_path, font_size * 2)
        dpg.bind_font(new_font)
        dpg.set_global_font_scale(0.5)  # Prevents scaling artifacts
```

See the full discussion and limitations:

- [GitHub Issue #1380](https://github.com/hoffstadt/DearPyGui/issues/1380)

### **2. Rebuilding DearPyGui from Source (Advanced)**

If you require a more permanent fix, modifying and recompiling DPG is an option.
**This requires modifying the source code and building DPG manually**, which may
not be feasible for all users.

### **Steps (Windows Example)**

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
3. Build DearPyGui (`python setup.py build`).
4. Replace `_dearpygui.pyd` in your **site-packages** directory.

See the full discussion and instructions:

- [GitHub Issue #773](https://github.com/hoffstadt/DearPyGui/issues/773)

## **Limitations**

- **Platform-dependent**: The issue is more pronounced on certain operating
  systems.
- **No official fix**: As of 2025, DearPyGui has not provided a native fix for
  this problem.
- **Rebuilding DPG is tedious**: Modifying and recompiling the library is not a
  practical solution for most users.

## **Conclusion**

While there is no perfect fix, applying **font scaling tricks** can help
mitigate the issue. If a more precise solution is needed, **rebuilding DPG**
might be an option, though it requires advanced setup.
