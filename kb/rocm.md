---
title: 'ROCm + PyTorch Setup on Arch Linux'
type: 'issue'
version: 1.1
date: '2025-01-30'
modified: '2025-03-02'
license: 'cc-by-nc-sa-4.0'
---

# **ROCm + PyTorch Setup on Arch Linux**

## **1. Installing ROCm Packages**

To install the required ROCm packages, run:

```sh
yay -S rocm-core rocm-hip-runtime rocm-language-runtime rocm-opencl-runtime miopen-hip rocminfo rocm-device-libs rocm-smi-lib rocm-ml-libraries rccl
```

After installation, verify that ROCm is installed:

```sh
ls -l /opt/rocm
```

If the directory exists, ROCm is installed correctly.

## **2. Setting Up Environment Variables**

Edit your **~/.zshrc** (or `~/.bashrc` if using Bash) and add:

```sh
# Paths
PATH_ROCM="/opt/rocm:/opt/rocm/lib:/opt/rocm/share:/opt/rocm/bin"

# Export Environment Variables for ROCm and PyTorch
export PATH="${PATH_ROCM}:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/opt/rocm/lib:$LIBRARY_PATH"
export C_INCLUDE_PATH="/opt/rocm/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/opt/rocm/include:$CPLUS_INCLUDE_PATH"

# ROCm Device Visibility
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export TRITON_USE_ROCM=1

# ROCm Architecture and Overrides
export AMDGPU_TARGETS="gfx1102"
export HCC_AMDGPU_TARGET="gfx1102"
export PYTORCH_ROCM_ARCH="gfx1102"
export HSA_OVERRIDE_GFX_VERSION="11.0.2"  # Ensure correct RDNA3 detection
export ROCM_PATH="/opt/rocm"
export ROCM_HOME="/opt/rocm"

# PyTorch ROCm Memory Management
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:False,garbage_collection_threshold:0.8"

# Optional: Disable hipBLASLt if issues occur
export USE_HIPBLASLT=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

Apply changes:

```sh
source ~/.zshrc
```

## **3. Ensuring Correct GPU Detection (gfx1102 Fix)**

PyTorch **2.6 and earlier** did **not support `gfx1102`**, causing **HIP
errors**. To ensure proper GPU detection:

Check detected GPU architectures:

```sh
rocm_agent_enumerator
```

Expected output:

```
gfx1102
```

Check if PyTorch recognizes it:

```sh
python -c "import torch; print(torch.cuda.get_arch_list())"
```

If `gfx1102` **is missing**, upgrade to **PyTorch Nightly**:

```sh
pip uninstall torch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.2.4
```

If you **must stay on stable releases**, wait for **PyTorch 2.7+**, where
`gfx1102` support is officially included.

## **4. Fixing `amdgpu.ids` Issue**

ROCm and PyTorch sometimes searches for `amdgpu.ids` in the wrong location. Fix it by
creating a **symlink**:

```sh
sudo mkdir -p /opt/amdgpu/share/libdrm
sudo ln -s /usr/share/libdrm/amdgpu.ids /opt/amdgpu/share/libdrm/amdgpu.ids
```

## **5. Installing PyTorch with ROCm**

Remove old installations:

```sh
pip uninstall torch
pip cache purge
```

For **stable PyTorch**:

```sh
pip install torch --index-url https://download.pytorch.org/whl/rocm
```

For **nightly PyTorch (if `gfx1102` isn’t recognized)**:

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.2.4
```

## **6. Verifying Installation**

Run:

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.device_count())       # 1
print(torch.cuda.get_device_name(0))   # AMD Radeon RX 7600 XT
```

Check detected architectures:

```python
print(torch.cuda.get_arch_list())  # Should list "gfx1102"
```

If `gfx1102` **is not listed**, PyTorch doesn’t support it in your build—use the
**nightly version**.

## **7. Running PyTorch on ROCm**

Test a basic CUDA tensor operation:

```python
import torch
x = torch.randn(3, 3).cuda()
print(x)
```

If no errors occur, your **setup is correct**.

## **8. Debugging and Monitoring**

### **Check ROCm System Status**

```sh
rocm-smi
```

Monitor VRAM usage:

```sh
watch -n 1 rocm-smi --showmeminfo vram
```

Verify `amdgpu` module is loaded:

```sh
dmesg | grep amdgpu
lsmod | grep amdgpu
```

### **If Out-of-Memory Errors Occur**

```python
import torch
torch.cuda.empty_cache()
```

## **9. Ensuring Future Stability**

### **Lock Current Working Configuration**

Since **rolling releases can break ROCm**, lock your working setup:

```sh
pip freeze | grep torch > requirements.txt
```

To restore later:

```sh
pip install -r requirements.txt
```

### **Monitor Upcoming PyTorch Releases**

To avoid **surprise breakage**, check for ROCm-related updates:

- **[PyTorch Release Notes](https://github.com/pytorch/pytorch/releases)**
- **[ROCm Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)**

### **Backup Your Environment Variables**

```sh
cp ~/.zshrc ~/.zshrc.working
```

If something **breaks after an update**, revert:

```sh
cp ~/.zshrc.working ~/.zshrc
source ~/.zshrc
```

## **10. Conclusion**

This guide provides a **stable**, **repeatable**, and **future-proof** ROCm +
PyTorch setup for **Arch Linux**. 🚀  
If issues arise after **future updates**, check:

1. **PyTorch’s `cuda.get_arch_list()` output**
2. **ROCm’s `rocm_agent_enumerator` detection**
3. **ROCm/PyTorch release notes for breaking changes**

## **Changelog**

### **v1.1 - 2025-03-01**

✅ Added **gfx1102 support fixes**  
✅ Included **PyTorch nightly build workaround**  
✅ Explained **handling future ROCm/PyTorch breakage**  
✅ Added **config backup & restoration instructions**

---

## Related Issues:

- https://github.com/ROCm/ROCm/issues/2536
- https://github.com/ROCm/ROCm/issues/4208
- https://github.com/pytorch/pytorch/issues/147626

## Related PR:

- https://github.com/pytorch/pytorch/pull/147761

## Related Sources:

- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
- https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html#architecture-support-compatibility-matrix
- https://rocm.docs.amd.com/en/docs-6.0.2/conceptual/gpu-isolation.html