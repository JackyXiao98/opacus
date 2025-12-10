#!/usr/bin/env bash
# set -e  # 出错就退出
# set -o pipefail  # 确保管道中任何命令失败都会导致脚本失败

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "=== Creating virtual environment ==="
uv venv

echo "=== Activating environment ==="
# 注意：source 只对当前 shell 生效，若要持续生效请在手动执行脚本后再 source
source .venv/bin/activate

echo "=== Installing local package in editable mode ==="
uv pip install -e .

echo "=== Done! ==="
