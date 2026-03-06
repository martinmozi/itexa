set -e
export USER_NAME="projekt"
export USER_PASSWORD="projekt123!"
export ROOT_PASSWORD="admin123!"

# Add repos if missing
grep -q "alpine/v3.21/main" /etc/apk/repositories || echo "https://dl-cdn.alpinelinux.org/alpine/v3.21/main" >> /etc/apk/repositories
grep -q "alpine/v3.21/community" /etc/apk/repositories || echo "https://dl-cdn.alpinelinux.org/alpine/v3.21/community" >> /etc/apk/repositories
apk update
apk add sudo curl bash

# Create user (BusyBox adduser: no -D flag, use --disabled-password)
adduser -s /bin/bash --disabled-password ${USER_NAME}
echo "${USER_NAME}:${USER_PASSWORD}" | chpasswd

# Sudo
mkdir -p /etc/sudoers.d
echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}
chmod 440 /etc/sudoers.d/${USER_NAME}

# SSH: allow password auth and both users
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i '/^AllowUsers/d' /etc/ssh/sshd_config
echo "AllowUsers root ${USER_NAME}" >> /etc/ssh/sshd_config
rc-service sshd restart

# Static IP on host-only adapter (eth1)
if ! grep -q "auto eth1" /etc/network/interfaces; then
  printf '\nauto eth1\niface eth1 inet static\n    address 192.168.56.10\n    netmask 255.255.255.0\n' >> /etc/network/interfaces
  echo "Added eth1 config to /etc/network/interfaces"
fi
ifup eth1 2>/dev/null || true
rc-update add networking boot 2>/dev/null || true

echo "Done! Test with: ssh -p 2222 projekt@127.0.0.1"