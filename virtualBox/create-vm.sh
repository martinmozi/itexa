#!/usr/bin/env bash
#
# create-vm.sh — Create VirtualBox VM and boot Alpine ISO
# After running this, do the interactive Alpine setup manually,
# then run provision-vm.sh to finish configuration.
#
set -e

# ── Configuration ──────────────────────────────────────────────
VM_NAME="itexa-vm"
VM_MEMORY=2048
VM_CPUS=2
VM_DISK_SIZE=8000
VM_IP="192.168.56.10"
ISO_URL="https://dl-cdn.alpinelinux.org/alpine/v3.21/releases/x86_64/alpine-virt-3.21.3-x86_64.iso"
ISO_FILE="alpine-virt-3.21.3-x86_64.iso"

HOSTONLY_IF=""

# ── Download ISO ───────────────────────────────────────────────
if [ ! -f "$ISO_FILE" ] || [ "$(stat -c%s "$ISO_FILE" 2>/dev/null || stat -f%z "$ISO_FILE" 2>/dev/null)" -lt 10000000 ]; then
  echo "Downloading Alpine Linux ISO..."
  rm -f "$ISO_FILE"
  curl -L -o "$ISO_FILE" "$ISO_URL"
fi

ISO_FULL_PATH="$(cygpath -w "$(realpath "$ISO_FILE")" 2>/dev/null || realpath "$ISO_FILE")"

# ── Cleanup old VM ─────────────────────────────────────────────
if VBoxManage showvminfo "$VM_NAME" &>/dev/null; then
  echo "Removing existing VM '$VM_NAME'..."
  VBoxManage controlvm "$VM_NAME" poweroff 2>/dev/null || true
  sleep 3
  VBoxManage unregistervm "$VM_NAME" --delete 2>/dev/null || true
fi

# ── Create VM ──────────────────────────────────────────────────
echo "Creating VM '$VM_NAME'..."
VBoxManage createvm --name "$VM_NAME" --ostype "Linux_64" --register

VBoxManage modifyvm "$VM_NAME" \
  --memory "$VM_MEMORY" \
  --cpus "$VM_CPUS" \
  --nic1 nat \
  --boot1 dvd --boot2 disk --boot3 none --boot4 none \
  --graphicscontroller vmsvga \
  --vram 16 \
  --audio-enabled off

VM_DIR=$(VBoxManage showvminfo "$VM_NAME" --machinereadable | grep "^CfgFile=" | sed 's/CfgFile="//;s/[/\\][^/\\]*"$//')
DISK_PATH="${VM_DIR}/${VM_NAME}.vdi"

VBoxManage createmedium disk --filename "$DISK_PATH" --size "$VM_DISK_SIZE" --format VDI
VBoxManage storagectl "$VM_NAME" --name "SATA" --add sata --controller IntelAhci --portcount 2
VBoxManage storageattach "$VM_NAME" --storagectl "SATA" --port 0 --device 0 --type hdd --medium "$DISK_PATH"
VBoxManage storageattach "$VM_NAME" --storagectl "SATA" --port 1 --device 0 --type dvddrive --medium "$ISO_FULL_PATH"

# ── Host-only network ──────────────────────────────────────────
HOSTONLY_IF=$(VBoxManage list hostonlyifs | grep "^Name:" | head -1 | sed 's/^Name:[ ]*//')
if [ -z "$HOSTONLY_IF" ]; then
  echo "Creating host-only network..."
  HOSTONLY_IF=$(VBoxManage hostonlyif create 2>&1 | grep -oP "'.*'" | tr -d "'")
fi
VBoxManage hostonlyif ipconfig "$HOSTONLY_IF" --ip 192.168.56.1 --netmask 255.255.255.0
VBoxManage modifyvm "$VM_NAME" --nic2 hostonly --hostonlyadapter2 "$HOSTONLY_IF"
VBoxManage modifyvm "$VM_NAME" --natpf1 "ssh,tcp,,2222,,22"
VBoxManage modifyvm "$VM_NAME" --natpf1 "http,tcp,,80,,80"
VBoxManage modifyvm "$VM_NAME" --natpf1 "https,tcp,,443,,443"

# ── Start VM with GUI ─────────────────────────────────────────
echo "Starting VM with GUI..."
VBoxManage startvm "$VM_NAME" --type gui

echo ""
echo "════════════════════════════════════════════════════════════"
echo " VM '$VM_NAME' is running. Do the interactive setup now:"
echo ""
echo " 1. Login as: root  (no password)"
echo " 2. Run: setup-alpine"
echo " 3. Answer the prompts:"
echo "    - Keyboard:    us / us"
echo "    - Hostname:    itexa-vm"
echo "    - Interface:   eth0, dhcp"
echo "    - DNS:         8.8.8.8"
echo "    - Timezone:    UTC"
echo "    - Proxy:       none"
echo "    - Mirror:      1  (or f for fastest)"
echo "    - SSH server:  openssh"
echo "    - NTP:         chrony"
echo "    - Disk:        sda, sys"
echo "    - Root password: admin123!"
echo ""
echo " 4. After install completes, type: poweroff"
echo " 5. Eject ISO: run this command:"
echo "    VBoxManage modifyvm $VM_NAME --boot1 disk --boot2 none"
echo "    VBoxManage storageattach $VM_NAME --storagectl SATA --port 1 --device 0 --type dvddrive --medium emptydrive"
echo " 6. Start VM again:"
echo "    VBoxManage startvm $VM_NAME --type headless"
echo " 7. Then run: ./provision-vm.sh"
echo "════════════════════════════════════════════════════════════"
