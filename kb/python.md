---
title: "Python Versioning and Compatibility Challenges"
type: "technical"
version: 1
date: "2024-12-24"
modified: "2024-12-30"
license: "cc-by-nc-sa-4.0"
---

# Python Versioning and Compatibility Challenges

Arch Linux has recently updated Python from version **3.12.x** to **3.13.x**, removing support for **3.12.x** packages. This change can cause significant issues for projects or environments that depend on the older version, as Python 3.12.x packages are no longer available in the system's repositories.

## **Understanding Package Incompatibilities**

Installing multiple Python versions globally (system-wide) may seem like a straightforward solution, but it can introduce complications, including:

- Dependency conflicts between projects.
- Difficulty managing multiple Python versions across different environments.
- Potential for breaking critical system tools that rely on a specific Python version.

To avoid these issues, it's essential to isolate Python installations. There are three common approaches to achieve this:

## **Approaches to Managing Multiple Python Versions**

### **1. Install Multiple Versions Globally**
   - Not Recommended: This approach can lead to conflicts and breakages in system tools or other projects.
   - Use cautiously only if you are comfortable with system-level configurations and troubleshooting.

### **2. Use Docker Containers**
   - Recommended for advanced setups or when absolute isolation is required.
   - **Pros**: Encapsulates Python and dependencies completely.
   - **Cons**: Can introduce complexity, especially when dealing with runtime or hardware dependencies.

### **3. Install Locally and Isolate**
   - **Preferred Solution**: Build and install Python locally within your user environment.
   - **Pros**:
      - Encapsulates the Python installation and its dependencies.
      - Avoids conflicts with system tools or global packages.
   - **Cons**: Requires manual setup but offers long-term simplicity and flexibility.

If you prefer the lightweight and versatile solution, installing Python locally is the best approach. This method isolates the Python installation from the system environment, avoiding dependency conflicts while still allowing full control over the version and configuration.

## **Why Local Installation?**

Local installation offers:

- **Encapsulation**: Your Python environment is entirely separate from the system environment, minimizing conflicts.
- **Flexibility**: You can manage multiple versions of Python without affecting the system or other users.
- **Simplicity**: Once set up, the environment is easy to use and maintain, without relying on additional tools like Docker.

This guide will walk you through building and setting up Python 3.12.8 locally, enabling `pip`, and configuring a user-specific environment for seamless integration with your projects.

## **How to Build and Set Up Python 3.12.8 Locally**

This guide walks through downloading, building, and installing Python 3.12.8 from source, along with enabling `pip` and configuring custom references for isolated usage.

### **1. Download Python Source Code**

Use `wget` or `curl` to retrieve the compressed tar archive of Python 3.12.8 from the official Python website.

```sh
wget https://www.python.org/ftp/python/3.12.8/Python-3.12.8.tgz
```

Expand the archive:

```sh
tar -xvzf Python-3.12.8.tgz
cd Python-3.12.8
```

### **2. Configure the Build**

Prepare the build with optimizations and ensure `pip` is included.

```sh
./configure --enable-optimizations --with-ensurepip=install --prefix=$HOME/.local/share/python/3.12.8
```

Explanation:

- `--enable-optimizations`: Enables Profile-Guided Optimization (PGO) and Link-Time Optimization (LTO) for better performance.
- `--with-ensurepip=install`: Ensures `pip` is included during the build.
- `--prefix`: Specifies the installation directory (local to your user).

### **3. Build Python**

Compile Python using all available CPU cores for faster execution.

```sh
make -j $(nproc)
```

### **4. Install Python Locally**

Install Python in the specified directory from the `--prefix` option during configuration.

```sh
make install
```

### **5. Verify the Installation**

Check the installed Python version and ensure it points to the correct binary.

```sh
$HOME/.local/share/python/3.12.8/bin/python3.12 --version
```

### **6. Enable `pip`**

Bootstrap `pip` after installation:

```sh
$HOME/.local/share/python/3.12.8/bin/python3.12 -m ensurepip --default-pip
```

Verify `pip` is available:

```sh
$HOME/.local/share/python/3.12.8/bin/python3.12 -m pip --version
```

### **7. Create a Custom Wrapper for `pip`**

Since `pip` doesn’t create a standalone executable, create a wrapper script to call `pip` via Python:

```sh
nano $HOME/.local/share/python/3.12.8/bin/pip3.12
```

Add the following content:

```bash
#!/usr/bin/env bash

PYTHONHOME=$HOME/.local/share/python/3.12.8
export PYTHONHOME

$PYTHONHOME/bin/python3.12 -m pip "$@"
```

Make the script executable:

```sh
chmod +x $HOME/.local/share/python/3.12.8/bin/pip3.12
```

### **8. Create a Symlink for Easy Access**

To make Python and `pip` easily accessible, create symbolic links in a directory already in your `PATH` (e.g., `~/.bin`):

```sh
ln -sf $HOME/.local/share/python/3.12.8/bin/python3.12 ~/.bin/python3.12
ln -sf $HOME/.local/share/python/3.12.8/bin/pip3.12 ~/.bin/pip3.12
```

Verify the symlinks work:

```sh
python3.12 --version
pip3.12 --version
```

### **9. Create and Use a Virtual Environment**

Test the setup by creating and activating a virtual environment:

```sh
python3.12 -m venv ~/my_project/.venv
source ~/my_project/.venv/bin/activate
```

Install a package within the virtual environment to confirm everything works:

```sh
pip3.12 install torch
python -c "import torch; print(torch.__version__)"
```

## **Summary**

You’ve successfully:

1. Downloaded and built Python 3.12.8 locally.
2. Configured `pip` for isolated usage.
3. Set up symbolic links for easy access.
4. Verified everything works with a virtual environment.

This process ensures you have a standalone, isolated Python setup that doesn’t interfere with system-wide configurations. You can repeat these steps for future Python versions or share them as a reference for others!

---

References:

- [Python Guide](https://devguide.python.org/getting-started/setup-building/)
- [Python FTP](https://www.python.org/ftp/python/3.12.8/)
