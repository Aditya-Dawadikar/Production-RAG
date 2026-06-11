#!/usr/bin/env bash
set -e

echo "Updating system..."
sudo apt update && sudo apt upgrade -y

echo "Installing basics..."
sudo apt install -y curl wget git build-essential ubuntu-drivers-common ca-certificates gnupg

echo "Installing NVIDIA server GPU driver..."
sudo ubuntu-drivers install --gpgpu

echo "Installing Docker..."
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

echo "Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

echo "Configuring Docker GPU runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "Done. Reboot now:"
echo "sudo reboot"