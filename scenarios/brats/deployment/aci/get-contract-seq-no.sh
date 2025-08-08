#!/bin/bash

# Extract the line containing 'Submitted ... as transaction'
line=$(grep "Submitted.*as transaction" ~/depa-training/../contract-ledger/tmp/contracts/log.txt)

# Extract the number after the dot in the transaction (e.g., 15 from 2.15)
seq_no=$(echo "$line" | sed -n 's/.*transaction [0-9]*\.\([0-9]*\).*/\1/p')

echo $seq_no