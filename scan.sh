#!/bin/bash
input="targets.txt"
output="reachable.txt"
> "$output"

while IFS= read -r target; do
    (
        ip=$(echo "$target" | cut -d: -f1)
        port=$(echo "$target" | cut -d: -f2)
        if nc -zv -w 2 "$ip" "$port" 2>&1 | grep -q succeeded; then
            echo "$target" >> "$output"
            echo "✓ $target"
        else
            echo "✗ $target"
        fi
    ) &
done < "$input"
wait
echo "Done. Reachable endpoints saved to $output"