#!/usr/bin/env bash
#
# provision-vm.sh — Provision the Alpine VM after interactive setup
# Run this after you've completed setup-alpine, rebooted from disk,
# and the VM is running.
set -e

curl -LsSf https://astral.sh/uv/install.sh | sh
sudo cp ~/.local/bin/uv /usr/local/bin/
sudo cp ~/.local/bin/uvx /usr/local/bin/

uv sync # pyproject.toml was copied
echo 'source ~/.venv/bin/activate' >> ~/.profile
