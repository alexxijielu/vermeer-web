$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    python -m venv "$PSScriptRoot\.venv"
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& $python -m pip install -r "$PSScriptRoot\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw 'Could not install model conversion dependencies.' }
$checkpoints = Join-Path $PSScriptRoot 'checkpoints'
New-Item -ItemType Directory -Force $checkpoints | Out-Null
$downloads = @(
    @{ Name = 'vermeer_B.ckpt'; Url = 'https://huggingface.co/microsoft/vermeer-B/resolve/main/vermeer_B.ckpt'; Size = 446472981 },
    @{ Name = 'vermeer_L.ckpt'; Url = 'https://huggingface.co/microsoft/vermeer-L/resolve/main/vermeer_L.ckpt'; Size = 1376604619 },
    @{ Name = 'vermeer_XL.ckpt'; Url = 'https://huggingface.co/microsoft/vermeer-XL/resolve/main/vermeer_XL.ckpt'; Size = 3106325011 },
    @{ Name = 'vq_ds16_c2i.pt'; Url = 'https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt'; Size = 287920306 }
)
foreach ($item in $downloads) {
    $destination = Join-Path $checkpoints $item.Name
    if (-not (Test-Path $destination) -or (Get-Item $destination).Length -ne $item.Size) {
        Write-Host "Downloading $($item.Name)..."
        Invoke-WebRequest $item.Url -OutFile "$destination.partial"
        if ((Get-Item "$destination.partial").Length -ne $item.Size) { throw "Incomplete checkpoint: $($item.Name)" }
        Move-Item -LiteralPath "$destination.partial" -Destination $destination -Force
    }
}
& $python "$PSScriptRoot\export_models.py"
if ($LASTEXITCODE -ne 0) { throw 'Export or numerical validation failed.' }
& $python "$PSScriptRoot\prepare_references.py"
if ($LASTEXITCODE -ne 0) { throw 'Preparing original TIFF reference fields failed.' }
Write-Host 'Browser model files prepared. Run npm install, then npm start.'
