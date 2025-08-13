#!/bin/bash

export contract_log_file=~/depa-training/../contract-ledger/tmp/contracts/log.txt

# Check if the contract log file exists
if [ ! -f $contract_log_file ]; then
    echo "Contract log file not found at $contract_log_file"
    echo "Please manually set the CONTRACT_SEQ_NO environment variable"
    exit 1
fi

# Extract the line containing 'Submitted ... as transaction'
line=$(grep "Submitted.*as transaction" $contract_log_file)

# Extract the number after the dot in the transaction (e.g., 15 from 2.15)
seq_no=$(echo "$line" | sed -n 's/.*transaction [0-9]*\.\([0-9]*\).*/\1/p')

echo $seq_no