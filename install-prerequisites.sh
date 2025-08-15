#!/bin/bash
#
# Script Name: install-prerequisites.sh
# Description:
#   Checks for the presence of required development tools and installs
#   any that are missing. Supports Ubuntu/Debian systems.
#   At the end, prints a summary table with each tool's status and version.
#
# Prerequisites Checked:
#   - Python3
#   - pip
#   - Go
#   - make
#   - jq
#   - wheel (Python package)
#   - Docker
#   - docker-compose
#   - Azure CLI
#   - Azure CLI extension: confcom
#
# Notes:
#   - Requires sudo privileges for package installation.
#   - After adding the user to the 'docker' group, a re-login or `newgrp docker`
#     may be needed for changes to take effect.

set -e

echo "=== Checking and installing prerequisites ==="

# Update apt package index
sudo apt update -y

# Arrays to track results
pkg_names=()
pkg_status=()
pkg_version=()

# Record result
record_result() {
    pkg_names+=("$1")
    pkg_status+=("$2")
    pkg_version+=("$3")
}

# Check & install function
check_and_install() {
    local cmd="$1"
    local pkg="$2"
    local install_cmd="$3"
    local ver_cmd="$4"

    if command -v "$cmd" >/dev/null 2>&1; then
        local ver
        ver=$($ver_cmd 2>/dev/null | head -n 1)
        echo "[OK] $pkg is already installed: $ver"
        record_result "$pkg" "Already Installed" "$ver"
    else
        echo "[Installing] $pkg..."
        eval "$install_cmd"
        local ver
        ver=$($ver_cmd 2>/dev/null | head -n 1)
        record_result "$pkg" "Installed Now" "$ver"
    fi
}

# --- Python3 ---
check_and_install python3 python3 "sudo apt install -y python3" "python3 --version"

# --- pip ---
check_and_install pip pip "sudo apt install -y python3-pip" "pip --version"

# --- Go ---
check_and_install go golang-go "sudo apt install -y golang-go" "go version"

# --- make ---
check_and_install make make "sudo apt install -y make" "make --version"

# --- jq ---
check_and_install jq jq "sudo apt install -y jq" "jq --version"

# --- wheel ---
if python3 -m pip show wheel >/dev/null 2>&1; then
    ver=$(python3 -m pip show wheel | grep Version: | awk '{print $2}')
    echo "[OK] wheel is already installed: version $ver"
    record_result "wheel (pip)" "Already Installed" "$ver"
else
    echo "[Installing] wheel..."
    python3 -m pip install wheel
    ver=$(python3 -m pip show wheel | grep Version: | awk '{print $2}')
    record_result "wheel (pip)" "Installed Now" "$ver"
fi

# --- Docker ---
if command -v docker >/dev/null 2>&1; then
    ver=$(docker -v)
    echo "[OK] docker is already installed: $ver"
    record_result "docker" "Already Installed" "$ver"
else
    echo "[Installing] docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    ver=$(docker -v)
    record_result "docker" "Installed Now" "$ver"
fi

# --- docker-compose ---
check_and_install docker-compose docker-compose "sudo apt install -y docker-compose" "docker-compose --version"

# --- Azure CLI ---
if command -v az >/dev/null 2>&1; then
    ver=$(az version --query '[].azure-cli' --output tsv 2>/dev/null || az version | grep azure-cli | head -n1)
    echo "[OK] Azure CLI is already installed: $ver"
    record_result "Azure CLI" "Already Installed" "$ver"
else
    echo "[Installing] Azure CLI..."
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    ver=$(az version --query '[].azure-cli' --output tsv 2>/dev/null || az version | grep azure-cli | head -n1)
    record_result "Azure CLI" "Installed Now" "$ver"
fi

# --- Azure CLI extension: confcom ---
if az extension show --name confcom >/dev/null 2>&1; then
    ver=$(az extension show --name confcom --query version -o tsv)
    echo "[OK] Azure CLI extension 'confcom' is already installed: $ver"
    record_result "Azure CLI ext: confcom" "Already Installed" "$ver"
else
    echo "[Installing] Azure CLI extension 'confcom'..."
    az extension add --name confcom -y
    ver=$(az extension show --name confcom --query version -o tsv)
    record_result "Azure CLI ext: confcom" "Installed Now" "$ver"
fi

# --- Docker group setup ---
if groups "$USER" | grep &>/dev/null '\bdocker\b'; then
    echo "[OK] User '$USER' is already in docker group."
else
    echo "[Adding] User '$USER' to docker group..."
    sudo usermod -aG docker "$USER"
    echo "You may need to log out and log back in for docker group changes to take effect."
fi

# --- Summary Table ---
echo
echo "=== Installation Summary ==="
printf "%-30s | %-17s | %-30s\n" "Package" "Status" "Version"
printf "%-30s | %-17s | %-30s\n" "------------------------------" "-----------------" "------------------------------"

for i in "${!pkg_names[@]}"; do
    printf "%-30s | %-17s | %-30s\n" "${pkg_names[$i]}" "${pkg_status[$i]}" "${pkg_version[$i]}"
done

echo "=== All prerequisites are installed. ==="
