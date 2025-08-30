#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <ip_address>"
    exit 1
fi

IP_ADDRESS=$1

# Update SSH config HostName for lambda host
sed -i "/^Host lambda$/,/^Host\|^$/ s/^  HostName .*/  HostName $IP_ADDRESS/" ~/.ssh/config

scp ~/.tmux.conf lambda:
ssh lambda 'echo "set -o vi" >> ~/.bashrc
pip install -r sts/bindings/requirements.txt
pip install -U torch'
