#!/bin/bash

MODELDIR=~/depa-training/scenarios/$SCENARIO/modeller

ACCOUNT_KEY=$(az storage account keys list --account-name $AZURE_STORAGE_ACCOUNT_NAME --only-show-errors | jq -r .[0].value)

az storage blob download \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_OUTPUT_CONTAINER_NAME \
  --file $MODELDIR/output.img \
  --name data.img \
  --account-key $ACCOUNT_KEY

encryptedImage=$MODELDIR/output.img
keyFilePath=$MODELDIR/output_key.bin

echo Decrypting $encryptedImage with key $keyFilePath
deviceName=cryptdevice1
deviceNamePath="/dev/mapper/$deviceName"

sudo cryptsetup luksOpen "$encryptedImage" "$deviceName" \
    --key-file "$keyFilePath" \
    --integrity-no-journal --persistent

mountPoint=`mktemp -d`
sudo mount -t ext4 "$deviceNamePath" "$mountPoint" -o loop

cp -r $mountPoint/* $MODELDIR/output/

echo "[!] Closing device..."

sudo umount "$mountPoint"
sleep 2
sudo cryptsetup luksClose "$deviceName"