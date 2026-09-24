#!/bin/bash
while true; do
  out=$(ssh hadi-windows "powershell -Command \"Get-Content D:\svd_project\logs\aegis_fixed_run3.log -Tail 100 -ErrorAction SilentlyContinue\"" 2>/dev/null)
  
  if echo "$out" | grep -q "aegis_v2 on sst2 ==="; then
     if echo "$out" | grep -q "HFGitExporter"; then
         echo "$out" | grep -A 5 "BENCHMARK QUALITY SWEEP SUMMARY TABLE"
         echo "$out" | grep "HFGitExporter"
         break
     fi
  fi
  sleep 5
done
