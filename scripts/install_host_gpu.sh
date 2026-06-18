#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Unsupported OS: /etc/os-release not found" >&2
  exit 1
fi

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" && "${ID:-}" != "debian" ]]; then
  echo "This installer targets Ubuntu/Debian. Install Docker and NVIDIA Container Toolkit manually, then run scripts/deploy.sh" >&2
  exit 1
fi

echo "[1/5] Installing base packages"
$SUDO apt-get update
$SUDO apt-get install -y ca-certificates curl gnupg lsb-release unzip

if ! command -v docker >/dev/null 2>&1; then
  echo "[2/5] Installing Docker Engine"
  tmp_script="$(mktemp)"
  curl -fsSL https://get.docker.com -o "$tmp_script"
  $SUDO sh "$tmp_script"
  rm -f "$tmp_script"
else
  echo "[2/5] Docker already installed: $(docker --version)"
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin is not available. Reinstall Docker Engine or install docker-compose-plugin." >&2
  exit 1
fi

echo "[3/5] Installing NVIDIA Container Toolkit"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

$SUDO apt-get update
$SUDO apt-get install -y nvidia-container-toolkit

echo "[4/5] Configuring Docker NVIDIA runtime"
$SUDO nvidia-ctk runtime configure --runtime=docker
$SUDO systemctl restart docker

echo "[5/5] Checking host GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found. Install NVIDIA driver on the host before GPU deployment." >&2
fi

echo "Done. If your user is not in the docker group yet, run: sudo usermod -aG docker $USER"
echo "Then relogin, or continue with sudo docker compose ..."
