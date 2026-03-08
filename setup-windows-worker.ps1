# LTX-2 Windows Worker Setup Script
# Run from PowerShell on the Windows machine (192.168.1.158)
# This pulls source from Linux (192.168.1.117), copies weights, builds, and starts the worker.

$ErrorActionPreference = "Stop"
$LINUX_HOST = "a@192.168.1.229"
$LINUX_CAKE = "/home/a/cake"
$LINUX_WEIGHTS = "/home/a/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
$CAKE_DIR = "C:\cake"
$MODELS_DIR = "C:\cake-models"

Write-Host "=== Step 1: Sync source code ===" -ForegroundColor Cyan

# Create directories
New-Item -ItemType Directory -Force -Path $CAKE_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$MODELS_DIR\transformer" | Out-Null

# Sync entire cake source (excludes target/ and .git internals)
Write-Host "Pulling source from Linux..."
scp -r "${LINUX_HOST}:${LINUX_CAKE}/Cargo.toml" "$CAKE_DIR\"
scp -r "${LINUX_HOST}:${LINUX_CAKE}/Cargo.lock" "$CAKE_DIR\"
scp -r "${LINUX_HOST}:${LINUX_CAKE}/cake-cli" "$CAKE_DIR\"
scp -r "${LINUX_HOST}:${LINUX_CAKE}/cake-core" "$CAKE_DIR\"
scp "${LINUX_HOST}:${LINUX_CAKE}/topology-ltx2.yml" "$CAKE_DIR\"

Write-Host "=== Step 2: Copy transformer weights (~36GB) ===" -ForegroundColor Cyan

# Check if weights already exist
$weightCount = (Get-ChildItem "$MODELS_DIR\transformer\*.safetensors" -ErrorAction SilentlyContinue).Count
if ($weightCount -ge 8) {
    Write-Host "Transformer weights already present ($weightCount shards), skipping download."
} else {
    Write-Host "Copying transformer shards from Linux... (this takes ~5 min on 10GbE)"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/config.json" "$MODELS_DIR\transformer\"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/diffusion_pytorch_model.safetensors.index.json" "$MODELS_DIR\transformer\"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/diffusion_pytorch_model-00001-of-00008.safetensors" "$MODELS_DIR\transformer\"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/diffusion_pytorch_model-00002-of-00008.safetensors" "$MODELS_DIR\transformer\"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/diffusion_pytorch_model-00003-of-00008.safetensors" "$MODELS_DIR\transformer\"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/diffusion_pytorch_model-00004-of-00008.safetensors" "$MODELS_DIR\transformer\"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/diffusion_pytorch_model-00005-of-00008.safetensors" "$MODELS_DIR\transformer\"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/diffusion_pytorch_model-00006-of-00008.safetensors" "$MODELS_DIR\transformer\"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/diffusion_pytorch_model-00007-of-00008.safetensors" "$MODELS_DIR\transformer\"
    scp "${LINUX_HOST}:${LINUX_WEIGHTS}/transformer/diffusion_pytorch_model-00008-of-00008.safetensors" "$MODELS_DIR\transformer\"
}

Write-Host "=== Step 3: Build ===" -ForegroundColor Cyan

Set-Location $CAKE_DIR

# Patch workspace to exclude cake-mobile (not needed on worker)
(Get-Content "$CAKE_DIR\Cargo.toml") -replace 'members = \["cake-core", "cake-cli", "cake-mobile"\]', 'members = ["cake-core", "cake-cli"]' | Set-Content "$CAKE_DIR\Cargo.toml"

cargo build --release --features cuda
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

Write-Host "=== Step 4: Open firewall ===" -ForegroundColor Cyan

# Add firewall rule (idempotent)
netsh advfirewall firewall show rule name="cake-worker" >$null 2>&1
if ($LASTEXITCODE -ne 0) {
    netsh advfirewall firewall add rule name="cake-worker" dir=in action=allow protocol=tcp localport=10128
    Write-Host "Firewall rule added for port 10128"
} else {
    Write-Host "Firewall rule already exists"
}

Write-Host "=== Step 5: Start worker ===" -ForegroundColor Green
Write-Host "Model path: $MODELS_DIR"
Write-Host "Listening on: 0.0.0.0:10128"
Write-Host ""

.\target\release\cake.exe worker `
    --model $MODELS_DIR `
    --name win5090 `
    --topology topology-ltx2.yml `
    --address 0.0.0.0:10128 `
    --image-model-arch ltx2 `
    --ltx-version 2
