# Setup home computers

## Computers
- Gateway computer: Ubuntu
- GPU server: Ubuntu 22.04
- Diskstation: Linux
- IPC: Ubuntu 22.04
- Brad Laptop: Windows 10
- Kristine Laptop: Windows 11
- Kristine NUC: Windows 10
- TV NUC: Windows 11
- TV PC: Windows 8
- 

## GPU Server

### NAS Setup
[ZFS Setup](https://ubuntu.com/tutorials/setup-zfs-storage-pool#3-creating-a-zfs-pool)

```cmd
sudo fdisk -l

Disk /dev/sdb: 14.55 TiB, 16000900661248 bytes, 31251759104 sectors
Disk model: ST16000NT001-3LV
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes


Disk /dev/sdc: 14.55 TiB, 16000900661248 bytes, 31251759104 sectors
Disk model: ST16000NT001-3LV
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes


Disk /dev/sdd: 14.55 TiB, 16000900661248 bytes, 31251759104 sectors
Disk model: ST16000NT001-3LV
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes


Disk /dev/sde: 14.55 TiB, 16000900661248 bytes, 31251759104 sectors
Disk model: ST16000NT001-3LV
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes
```

NAS setup
https://ubuntu.com/server/docs/service-nfs
https://ubuntu.com/tutorials/setup-zfs-storage-pool#1-overview
https://wiki.debian.org/ZFS
https://www.youtube.com/watch?v=FwL0yrGV2O0

List disks to get names:
```cmd
sudo fdisk -l
```

Check drives.  Get drive names and pysical sector size from fdisk -l listing:
```cmd
sudo badblocks -b 4096 -sw /dev/sdb
```
Create alias to specific seral number so disks are tracked by disk serail number
List drive serial number.  Serial number rather than 
```cmd
cd /dev/disk/by-id
ll
ll ata-ST16000* -ltr
```

Response
```cmd
lrwxrwxrwx 1 root root   9 Mar 25 14:36 ata-ST16000NT001-3LV101_ZR60N3M6 -> ../../sdb
lrwxrwxrwx 1 root root   9 Mar 25 14:36 ata-ST16000NT001-3LV101_ZRS0G3T5 -> ../../sdc
lrwxrwxrwx 1 root root   9 Mar 25 14:36 ata-ST16000NT001-3LV101_ZR70J3AC -> ../../sdd
lrwxrwxrwx 1 root root   9 Mar 25 14:36 ata-ST16000NT001-3LV101_ZRS04P4K -> ../../sde
```

Create alias to disk serial numbers
```cmd
cd /etc/zfs
sudo nano vdev_id.conf
```

Create aliases for disks:
```
alias farm_a_0 /dev/disk/by-id/ata-ST16000NT001-3LV101_ZR60N3M6
alias farm_a_1 /dev/disk/by-id/ata-ST16000NT001-3LV101_ZRS0G3T5
alias farm_b_0 /dev/disk/by-id/ata-ST16000NT001-3LV101_ZR70J3AC
alias farm_b_1 /dev/disk/by-id/ata-ST16000NT001-3LV101_ZRS04P4K
```

Load alias:
```cmd
sudo udevadmin trigger
```

Check names are loaded
```
ll /dev/disk/by-vdev
```

Create zfs pool
```cmd
sudo zpool create farm mirror /dev/disk/by-id/ata-ST16000NT001-3LV101_ZR60N3M6 /dev/disk/by-id/ata-ST16000NT001-3LV101_ZRS0G3T5 mirror /dev/disk/by-id/ata-ST16000NT001-3LV101_ZR70J3AC /dev/disk/by-id/ata-ST16000NT001-3LV101_ZRS04P4K
# If need to force setup
sudo zpool create -f farm mirror /dev/disk/by-id/ata-ST16000NT001-3LV101_ZR60N3M6 /dev/disk/by-id/ata-ST16000NT001-3LV101_ZRS0G3T5 mirror /dev/disk/by-id/ata-ST16000NT001-3LV101_ZR70J3AC /dev/disk/by-id/ata-ST16000NT001-3LV101_ZRS04P4K
sudo zfs set compression=lz4 farm
```

Inspect pool:
```cmd
zpool status
zpool list -v
zsf get compression farm
```
Create shared drive
```cmd
sudo zfs create farm/fam
sudo chown -R blarson:blarson /farm
sudo chmod 777 /farm
```

```cmd
df -h | grep farm
farm                                               29T  128K   29T   1% /farm
```


Repair pool if degrated
```cmd
sudo zpool status # check state
sudo zpool scrub farm # test state
sudo zpool clear farm # clear errors of test successful

Install [network file system (NSF)](https://ubuntu.com/server/docs/service-nfs)
```cmd
sudo apt install nfs-kernel-server
sudo systemctl start nfs-kernel-server.service
```
Add shared zfs drive to /etc/exports
```
sudo echo "/farm *(rw,async,no_subtree_check)" > /etc/exports
```

Apply configuration:
```cmd
sudo exportfs -a
```

Speed test
```cmd
sudo dd if=/dev/zero of=/farm/test1.img bs=1G count=100 oflag=dsync
```

## SMP configuration:

Try Samba rather than NSF because Windows 10 home (Kitchen NUC) cannot use NSF

```cmd
sudo apt update
sudo apt install samba
```
We can check if the installation was successful by running:
```cmd
whereis samba
```

```cmd
sudo nano /etc/samba/smb.conf
```
Edit samba configuration
```cmd
[sambashare]
    comment = RAID Share
    path = /farm
    read only = no
    browsable = yes
```
Restart Samba for it to take effect.  Pass-through firewall
```cmd

sudo ufw allow samba
```

Setting up User Accounts
```cmd
sudo smbpasswd -a blarson
```

Copy files to RAID
From Diskstation
```cmd
sudo rsync -avz --progress /volume1/Data/ blarson@192.168.0.163:/farm/data
sudo rsync -avz --progress /volume1/Pictures/ blarson@192.168.0.163:/farm/pictures

ssh blarson@192.168.1.95
sudo rsync -avz --progress /volume1/Videos/ blarson@192.168.0.163:/farm/videos
sudo rsync -avz --progress /volume1/Data/ blarson@192.168.0.163:/farm/data
sudo rsync -avz --progress /volume1/Pictures/ blarson@192.168.0.163:/farm/pictures

```

sudo useradd kristine
sudo smbpasswd -a kristine

rsnapshot is used for backups

### Add users
groups
sudo cat /etc/group | grep users
sudo groupadd users
sudo groupadd parents
sudo groupadd jared
sudo usermod -a -G users blarson
groups
sudo useradd -M -N -G users brad
sudo useradd -M -N -G users kristine
sudo useradd -M -N -G users aaron
sudo useradd -M -N -G users spencer
sudo useradd -M -N -G users elise
sudo useradd -M -N -G users sarah
sudo useradd -M -N -G users nathan
sudo useradd -M -N -G users julia
sudo useradd -M -N -G users kylee
sudo useradd -M -N -G users annika
sudo useradd -M -N -G users guest

sudo usermod -aG users blarson
sudo usermod -aG users kristine
sudo usermod -aG users brad
sudo usermod -aG users aaron
sudo usermod -aG users spencer
sudo usermod -aG users elise
sudo usermod -aG users sarah
sudo usermod -aG users nathan
sudo usermod -aG users julia
sudo usermod -aG users kylee
sudo usermod -aG users annika
sudo usermod -aG users guest
grep 'users' /etc/group

sudo usermod -aG parents blarson
sudo usermod -aG parents brad
sudo usermod -aG parents kristine

sudo usermod -aG jared blarson
sudo usermod -aG jared brad

# The newgrp command is used to change the current group ID during a login session.
newgrp jared
newgrp parents
newgrp users

sudo chown root:root /farm/videos
sudo chown root:root /farm/data
sudo chown root:root /farm/pictures
sudo chown annika:users /farm/data/Annika

sudo chown -R bhlarson:bhlarson /farm/data/Brad
sudo chown -R kristine:parents /farm/data/Kristine
sudo chown -R aaron:users /farm/data/Aaron
sudo chown -R spencer:users /farm/data/Spencer
sudo chown -R elise:users /farm/data/Elise
sudo chown -R sarah:users /farm/data/Sarah
sudo chown -R nathan:users /farm/data/Nathan
sudo chown -R julia:users /farm/data/Julia
sudo chown -R kylee:users /farm/data/Kylee
sudo chown -R Annika:users /farm/data/Annika

sudo chmod -R 640 /farm/data/Brad

sudo chown root:root /farm/pictures
### Sync data


gio trash <file>
