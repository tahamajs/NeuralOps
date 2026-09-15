# Ensure result directories match standard runner expectations
if ((Test-Path "D:\SVD_Project\results\unified_adaptive_sparse_attention_attention") -and -not (Test-Path "D:\SVD_Project\results\unified_adaptive_sparse_attention")) {
    Copy-Item -Path "D:\SVD_Project\results\unified_adaptive_sparse_attention_attention" -Destination "D:\SVD_Project\results\unified_adaptive_sparse_attention" -Recurse -Force
    Write-Host "Created unified_adaptive_sparse_attention copy."
}

if ((Test-Path "D:\SVD_Project\results\unified_dobi_svd") -and -not (Test-Path "D:\SVD_Project\results\unified_dobi")) {
    Copy-Item -Path "D:\SVD_Project\results\unified_dobi_svd" -Destination "D:\SVD_Project\results\unified_dobi" -Recurse -Force
    Write-Host "Created unified_dobi copy."
}

if ((Test-Path "D:\SVD_Project\results\unified_dynamic_entropy_svd") -and -not (Test-Path "D:\SVD_Project\results\unified_dynamic_entropy")) {
    Copy-Item -Path "D:\SVD_Project\results\unified_dynamic_entropy_svd" -Destination "D:\SVD_Project\results\unified_dynamic_entropy" -Recurse -Force
    Write-Host "Created unified_dynamic_entropy copy."
}
Write-Host "Folder sync completed."
