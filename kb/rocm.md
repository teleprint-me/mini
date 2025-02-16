---
title: "ROCm + PyTorch Setup on Arch Linux"
type: "issue"
version: 1
date: "2025-01-30"
modified: "2025-02-13"
license: "cc-by-nc-sa-4.0"
---

# ROCm + PyTorch Setup on Arch Linux

## **1. Installing ROCm Packages**
To install the required ROCm packages, use the following command:

```sh
yay -S rocm-core rocm-hip-runtime rocm-language-runtime rocm-opencl-runtime miopen-hip rocminfo rocm-device-libs rocm-smi-lib rocm-ml-libraries rccl
```

After installation, verify that ROCm is present:

```sh
ls -l /opt/rocm
```

If the directory exists, ROCm is installed.

## **2. Setting Up Environment Variables**
Add the following lines to your `~/.zshrc` (or `~/.bashrc` if using Bash):

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
export HSA_OVERRIDE_GFX_VERSION=11.0.0 # RDNA3
export ROCM_PATH="/opt/rocm"
export ROCM_HOME="/opt/rocm"

# PyTorch ROCm Memory Management
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:False,garbage_collection_threshold:0.8"

# Optional: Disable hipBLASLt if issues occur
export USE_HIPBLASLT=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

After editing, apply the changes:
```sh
source ~/.zshrc
```

## **3. Fixing amdgpu.ids Issue**
ROCm sometimes searches for `amdgpu.ids` in an incorrect path. Fix this by creating a symbolic link:

```sh
sudo mkdir -p /opt/amdgpu/share/libdrm
sudo ln -s /usr/share/libdrm/amdgpu.ids /opt/amdgpu/share/libdrm/amdgpu.ids
```

## **4. Installing PyTorch with ROCm**
Remove any existing PyTorch installations:
```sh
pip uninstall torch torchvision torchaudio
pip cache purge
```

Then install PyTorch with ROCm support:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm
```

## **5. Verifying Installation**
Run the following Python script to verify PyTorch detects the GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

Expected output:
```
True
1
AMD Radeon RX 7600 XT
```

## **6. Running PyTorch on ROCm**
Try running a simple tensor operation on the GPU:

```python
import torch
x = torch.randn(3, 3).cuda()
print(x)
```

If no errors occur, PyTorch with ROCm is working correctly.

## **7. Debugging and Monitoring**
Check ROCm system status:
```sh
rocm-smi
```

Monitor GPU memory usage:
```sh
watch -n 1 rocm-smi --showmeminfo vram
```

Check if `amdgpu` module is loaded:
```sh
dmesg | grep amdgpu
lsmod | grep amdgpu
```

If you encounter out-of-memory errors, try forcing PyTorch to free memory:
```python
import torch
torch.cuda.empty_cache()
```

## **8. Backup Your Working Configuration**
Once everything works, back up your configuration:
```sh
cp ~/.zshrc ~/.zshrc.working
```

---

Following these steps ensures a **stable** and **repeatable** ROCm + PyTorch setup. 🚀

