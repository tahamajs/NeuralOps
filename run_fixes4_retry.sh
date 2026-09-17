#!/bin/bash
while true; do
    ssh gpu "echo Connection OK && cd /d D:\SVD_Project && python -u run_fixes4.py"
    if [ $? -eq 0 ]; then
        echo "Sweep completed successfully."
        break
    else
        echo "Connection failed, retrying in 10 seconds..."
        sleep 10
    fi
done
