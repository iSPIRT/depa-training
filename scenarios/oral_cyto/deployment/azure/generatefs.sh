#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

while getopts ":d:k:i:" options; do
    case $options in 
        d)dataPath=$OPTARG;;
        k)keyFilePath=$OPTARG;;
        i)encryptedImage=$OPTARG;;
    esac
done

echo Encrypting $dataPath with key $keyFilePath and generating $encryptedImage
deviceName=cryptdevice1
deviceNamePath="/dev/mapper/$deviceName"

if [ -f "$keyFilePath" ]; then
    echo "[!] Encrypting dataset using $keyFilePath"
else
    echo "[!] Generating keyfile..."
    dd if=/dev/random of="$keyFilePath" count=1 bs=32
    truncate -s 32 "$keyFilePath"
fi

echo "[!] Creating encrypted image..."

response=`du -s $dataPath`
read -ra arr <<< "$response"
# Add a 64MB safety margin to the raw data size, then round UP to the next
# power of 2. The previous formula (round-to-nearest power of 2, doubled)
# undersized datasets whose raw size landed just below a power-of-2
# midpoint — e.g. 42MB raw produced a 64MB image, leaving ~43MB usable
# after LUKS+ext4 overhead, with no headroom for cp on the long tail.
target_kb=$(($arr + 65536))
power=$(echo "scale=0; l($target_kb)/l(2)" | bc -l)
size=$(echo "scale=0; 2^($power+1)" | bc -l)

# cryptsetup requires 16M or more; floor at 128M for safety. The +64MB
# margin above already forces size >= 128M in practice; this is
# defense-in-depth in case the formula is ever revised.
if (($((size)) < 131072)); then
    size="131072"
fi
size=$size"K"

echo "Data size: $size"

rm -f "$encryptedImage"
touch "$encryptedImage"
truncate --size $size "$encryptedImage"

sudo cryptsetup luksFormat --type luks2 "$encryptedImage" \
    --key-file "$keyFilePath" -v --batch-mode --sector-size 4096 \
    --cipher aes-xts-plain64 \
    --pbkdf pbkdf2 --pbkdf-force-iterations 1000

sudo cryptsetup luksOpen "$encryptedImage" "$deviceName" \
    --key-file "$keyFilePath" \
    --integrity-no-journal --persistent

echo "[!] Formatting as ext4..."

sudo mkfs.ext4 "$deviceNamePath"

echo "[!] Mounting..."

mountPoint=`mktemp -d`
echo "Mounting to $mountPoint"
sudo mount -t ext4 "$deviceNamePath" "$mountPoint" -o loop

echo "[!] Copying contents to encrypted device..."

# The /* is needed to copy folder contents instead of the folder + contents
sudo cp -r $dataPath/* "$mountPoint"
sudo rm -rf "$mountPoint/lost+found"
ls "$mountPoint"

echo "[!] Closing device..."

sudo umount "$mountPoint"
sleep 2
sudo cryptsetup luksClose "$deviceName"
