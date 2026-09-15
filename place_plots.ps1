$plots = Get-ChildItem "D:\SVD_Project\results\module_plots\*_dissection_profile.png"
Write-Host "Found $($plots.Count) plot files to place into module folders..."

foreach ($p in $plots) {
    $modName = $p.BaseName.Replace("_dissection_profile", "")
    $targetDir = "D:\SVD_Project\results\unified_$modName"
    if (Test-Path $targetDir) {
        Copy-Item -Path $p.FullName -Destination $targetDir -Force
    }
}
Write-Host "Successfully placed profile plots inside each individual module directory."
