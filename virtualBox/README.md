# Alpine Linux VM Setup for itexa-vm

Automated two-script setup for an Alpine Linux virtual machine on VirtualBox with SSH access, Python 3.13, and FastAPI.

## Prerequisites

Install these on your Windows host before starting:

- **VirtualBox** — [download](https://www.virtualbox.org/wiki/Downloads) and ensure `VBoxManage` is in your PATH
- **Git Bash** — [download](https://git-scm.com/downloads) (provides the bash shell on Windows)
- **sshpass** — needed for automated SSH login in the provisioning script
- **curl** — comes with Git Bash

### Installing sshpass on Windows

sshpass is not bundled with Git Bash. Options:

1. **MSYS2**: `pacman -S sshpass`
2. **Chocolatey**: `choco install sshpass`
3. **Build from source**: download from [sourceforge](https://sourceforge.net/projects/sshpass/) and compile in MSYS2

## VM Specifications

| Setting          | Value                    |
|------------------|--------------------------|
| VM Name          | `itexa-vm`               |
| OS               | Alpine Linux 3.21 (x86_64) |
| RAM              | 2048 MB                  |
| CPUs             | 2                        |
| Disk             | 8 GB (VDI)              |
| NIC 1            | NAT                      |
| NIC 2            | Host-only (192.168.56.10)|

## Quick Start

### Step 1: Create the VM

```bash
chmod +x create-vm.sh
./create-vm.sh
```

This downloads the Alpine ISO, creates the VirtualBox VM with networking configured, and opens a GUI window where you will do the interactive Alpine installation.

### Step 2: Interactive Alpine Setup

A VirtualBox window will open with the Alpine live environment. Follow these steps inside the VM console:

**Login:**

```
localhost login: root
```

No password is needed on the live ISO.

**Run the installer:**

```
setup-alpine
```

**Answer the prompts as follows:**

| Prompt                    | Answer              |
|---------------------------|---------------------|
| Keyboard layout           | `us`               |
| Keyboard variant          | `us`               |
| Hostname                  | `itexa-vm`         |
| Network interface         | `eth0`             |
| IP address for eth0       | `dhcp`             |
| Manual network config?    | `n`                |
| Network interface (again) | `eth1`             |
| IP address for eth1       | `192.168.56.10`    |
| Netmask for eth1          | `255.255.255.0`    |
| Gateway for eth1          | `none`             |
| Manual network config?    | `n`                |
| Do any more interfaces?   | `done`             |
| Root password             | `admin123!`        |
| Timezone                  | `UTC`              |
| Proxy                     | `none`             |
| NTP client                | `chrony`           |
| APK mirror                | `1` (or `f` for fastest) |
| SSH server                | `openssh`          |
| Disk                      | `sda`              |
| How to use disk           | `sys`              |
| Erase disk?               | `y`                |

**After installation completes:**

```
poweroff
```

### Step 3: Switch Boot to Disk and Eject ISO

Back in your Git Bash terminal, run:

```bash
VBoxManage modifyvm itexa-vm --boot1 disk --boot2 none
VBoxManage storageattach itexa-vm --storagectl SATA --port 1 --device 0 --type dvddrive --medium emptydrive
```

### Step 4: Start the VM Headless

```bash
VBoxManage startvm itexa-vm --type headless
```

### Step 5: Provision the VM

- run `provision-root.sh` under root
- run `provision-user.sh` under projekt user

This connects via SSH and automatically:

- Installs `sudo`, `curl`, `bash`
- Creates user `projekt` with password `projekt123!` and passwordless sudo
- Configures SSH for both `root` and `projekt` users
- Sets up the host-only static IP `192.168.56.10` on `eth1`
- Installs [uv](https://github.com/astral-sh/uv) package manager
- Installs Python 3.13 with a virtual environment
- Installs FastAPI, uvicorn, pydantic, and simple-websocket-server

## Connecting to the VM

After provisioning, the VM is accessible two ways:

### Host-only Network (recommended)

All ports are directly accessible from your Windows host at `192.168.56.10`:

```bash
ssh projekt@192.168.56.10
```

Web services running in the VM are available at:

- `http://192.168.56.10` (port 80)
- `https://192.168.56.10` (port 443)
- Any other port your application listens on

### NAT Port Forwarding (localhost)

These ports are forwarded from your Windows localhost to the VM:

| Host               | VM               |
|--------------------|------------------|
| `127.0.0.1:2222`   | `22` (SSH)      |
| `127.0.0.1:80`     | `80` (HTTP)     |
| `127.0.0.1:443`    | `443` (HTTPS)   |

```bash
ssh -p 2222 projekt@127.0.0.1
```

> **Note:** If you already have services running on ports 80 or 443 on your Windows machine (IIS, Apache, Skype, etc.), the NAT port forwards will fail with a conflict. The host-only route (`192.168.56.10`) is not affected.

## Credentials

| User      | Password       | SSH Access |
|-----------|----------------|------------|
| `root`    | `admin123!`    | Yes        |
| `projekt` | `projekt123!`  | Yes (with passwordless sudo) |

## VM Management

### Start / Stop

```bash
# Start headless
VBoxManage startvm itexa-vm --type headless

# Start with GUI (for debugging)
VBoxManage startvm itexa-vm --type gui

# Graceful shutdown
VBoxManage controlvm itexa-vm acpipowerbutton

# Force power off
VBoxManage controlvm itexa-vm poweroff
```

### SSH into the VM

```bash
# Via host-only (any port)
ssh projekt@192.168.56.10

# Via NAT
ssh -p 2222 projekt@127.0.0.1
```

### Check VM Status

```bash
VBoxManage showvminfo itexa-vm --machinereadable | grep "^VMState="
```

### Delete the VM Completely

```bash
VBoxManage controlvm itexa-vm poweroff 2>/dev/null || true
sleep 2
VBoxManage unregistervm itexa-vm --delete
```

## Python Environment

The `projekt` user has a Python virtual environment at `~/venv` that activates automatically on login. It includes:

- Python 3.13 (installed via uv)
- FastAPI
- uvicorn
- pydantic
- simple-websocket-server

To run a FastAPI app:

```bash
ssh projekt@192.168.56.10
cd /path/to/your/app
uvicorn main:app --host 0.0.0.0 --port 80
```

Then access it from your Windows browser at `http://192.168.56.10`.

## Troubleshooting

### SSH connection refused

Make sure the VM is running and has finished booting:

```bash
VBoxManage showvminfo itexa-vm --machinereadable | grep "^VMState="
```

If the VM is running but SSH is not responding, start it with GUI to see what's happening:

```bash
VBoxManage controlvm itexa-vm poweroff
VBoxManage startvm itexa-vm --type gui
```

### Cannot reach 192.168.56.10

The host-only network adapter (`eth1`) may not be up. SSH in via NAT and bring it up:

```bash
ssh -p 2222 root@127.0.0.1
ifup eth1
```

If that fails, check if the interface is configured:

```bash
cat /etc/network/interfaces
```

You should see an `eth1` block with address `192.168.56.10`. If missing, re-run `./provision-vm.sh`.

### Port 80/443 conflict on localhost

If another service is using port 80 or 443 on your Windows host, the NAT forwarding will not work for those ports. You can either stop the conflicting service, or just use the host-only IP (`192.168.56.10`) which is unaffected.

### VM boots to black screen or UEFI shell

The ISO was not ejected before rebooting. Power off the VM and run:

```bash
VBoxManage modifyvm itexa-vm --boot1 disk --boot2 none
VBoxManage storageattach itexa-vm --storagectl SATA --port 1 --device 0 --type dvddrive --medium emptydrive
VBoxManage startvm itexa-vm --type headless
```

## File Structure

```
.
├── README.md           # This file
├── create-vm.sh        # Step 1: Creates VM and boots Alpine ISO
└── provision-vm.sh     # Step 5: Provisions VM over SSH
```

# export import VM
- VBoxManage export itexa-vm -o itexa-vm.ova
- VBoxManage import itexa-vm.ova
