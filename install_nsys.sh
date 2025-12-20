#!/usr/bin/env bash
# set -euo pipefail

#############################################
# Nsight Systems GUI Version Installer
# - Removes old CLI nsys
# - Installs GUI version to ~/tools/nsys_gui
# - Updates PATH
# - Verifies installation
#############################################

echo "========================================="
echo "  Installing Nsight Systems (GUI version)"
echo "========================================="

# --------------------------
# 1. Remove old CLI nsys
# --------------------------
if [ -f /usr/local/bin/nsys ]; then
    echo "[INFO] Removing old /usr/local/bin/nsys ..."
    sudo rm -f /usr/local/bin/nsys
else
    echo "[INFO] No system-level nsys found. Skipping removal."
fi

# --------------------------
# 2. Prepare install paths
# --------------------------
INSTALL_ROOT=~/tools/nsys_gui
RUN_FILE=nsys_gui.run
NSYS_URL="https://developer.download.nvidia.com/devtools/nsight-systems/NsightSystems-linux-public-2025.5.1.121-3638078.run"

echo "[INFO] Creating directory: $INSTALL_ROOT"
mkdir -p "$INSTALL_ROOT"

# --------------------------
# 3. Download GUI installer
# --------------------------
echo "[INFO] Downloading Nsight Systems GUI package..."
wget -q --show-progress "$NSYS_URL" -O "$RUN_FILE"

chmod +x "$RUN_FILE"

# --------------------------
# 4. Extract (no sudo needed)
# --------------------------
echo "[INFO] Extracting GUI package to $INSTALL_ROOT ..."
./"$RUN_FILE" --accept --noexec --target "$INSTALL_ROOT"

# --------------------------
# 5. Add to PATH
# --------------------------
NSYS_BIN_DIR=$(find "$INSTALL_ROOT" -type f -name nsys -printf "%h\n" | head -n 1)

if ! grep -q "$NSYS_BIN_DIR" ~/.bashrc; then
    echo "export PATH=$NSYS_BIN_DIR:\$PATH" >> ~/.bashrc
    echo "[INFO] PATH updated in ~/.bashrc"
else
    echo "[INFO] PATH already contains nsys bin directory."
fi

# Apply new PATH settings
source ~/.bashrc

# --------------------------
# 6. Verify
# --------------------------
echo "[INFO] Verifying installation..."
nsys --version || (echo "[ERROR] nsys not working!" && exit 1)

echo "========================================="
echo " Nsight Systems GUI successfully installed!"
echo " Binary path: $NSYS_BIN_DIR"
echo "========================================="
