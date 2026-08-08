#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_root="${NGC_INSTALL_ROOT:-${project_root}/.tools}"
ngc_version="4.34.10"
archive_sha256="58bc4d7b6901551a095ce5a43a0065b1c22d5eaf1a164d40029ee3fda11ccd47"
url="https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/${ngc_version}/files/ngccli_linux.zip"

if command -v ngc >/dev/null 2>&1; then
  ngc --version
  exit 0
fi

mkdir -p "${install_root}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

curl -fL "${url}" -o "${tmp_dir}/ngccli_linux.zip"
echo "${archive_sha256}  ${tmp_dir}/ngccli_linux.zip" | sha256sum -c -
unzip -q "${tmp_dir}/ngccli_linux.zip" -d "${install_root}"
chmod u+x "${install_root}/ngc-cli/ngc"

echo "NGC CLI installed. For this shell run:"
echo "  export PATH=\"${install_root}/ngc-cli:\$PATH\""
echo "Then configure your account once with:"
echo "  ngc config set"

