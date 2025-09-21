# General build script for YoloRealTime_Detection using PyInstaller
# Place this file in the project root and run in PowerShell:
#   ./build.ps1

$pythonFiles   = Get-ChildItem -Filter *.py | Select-Object -First 1
$iconFiles     = Get-ChildItem -Filter *.ico | Select-Object -First 1
$requirements  = "requirements.txt"

if (-not $pythonFiles) {
    Write-Host "ERROR: YoloRealTime_Detection.py not found in the current folder." -ForegroundColor Red
    exit 1
}
if (-not $iconFiles) {
    Write-Host "ERROR: No icon (.ico) file found in icons\ folder." -ForegroundColor Red
    exit 1
}
if (!(Test-Path $requirements)) {
    Write-Host "ERROR: $requirements not found. Please place it in the current folder." -ForegroundColor Red
    exit 1
}

$script = $pythonFiles.Name
$name   = [System.IO.Path]::GetFileNameWithoutExtension($script)
$icon   = $iconFiles.FullName

# Read hidden imports from requirements.txt
$hiddenImports = Get-Content $requirements | Where-Object { $_ -and $_ -notmatch "^#" }
$hiddenImportArgs = ($hiddenImports | ForEach-Object { "--hidden-import $_" }) -join " "

# Build argument array for PyInstaller
$pyinstallerArgs = @()
$pyinstallerArgs += "--name"; $pyinstallerArgs += $name
$pyinstallerArgs += "--onedir"
$pyinstallerArgs += "--contents-directory"; $pyinstallerArgs += "dist\$name"
$pyinstallerArgs += "--noconsole"
$pyinstallerArgs += "--icon=$icon"

# Automatically include all folders in the current directory (excluding files and hidden/system folders)
$folders = Get-ChildItem -Directory | Where-Object { $_.Name -ne '.git' -and $_.Name -ne '__pycache__' }
foreach ($folder in $folders) {
    $pyinstallerArgs += "--add-data"
    $pyinstallerArgs += "$($folder.Name);$($folder.Name)"
}

# Add hidden imports
if ($hiddenImports.Count -gt 0) {
    foreach ($import in $hiddenImports) {
        $pyinstallerArgs += "--hidden-import"; $pyinstallerArgs += $import
    }
}
$pyinstallerArgs += $script

# Show the command and ask for confirmation
$pyinstallerArgsString = $pyinstallerArgs -join ' '
Write-Host "PyInstaller will run with the following command:" -ForegroundColor Yellow
Write-Host "pyinstaller $pyinstallerArgsString" -ForegroundColor Cyan

# Confirm with the user
$confirm = Read-Host "Do you want to continue? (Y/N)"
if ($confirm -ne 'Y' -and $confirm -ne 'y') {
    Write-Host "Build cancelled by user." -ForegroundColor Red
    exit 0
}

# Run PyInstaller
Write-Host "Building..." -ForegroundColor Green
pyinstaller @pyinstallerArgs
Write-Host "Build finished." -ForegroundColor Green
Pause