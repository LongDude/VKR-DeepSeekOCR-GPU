# init-venv.ps1
param(
    [string]$RootDir = ".",
    [string]$HostVenv = ".venv-dev"
)

# Проверка Python 3.11
if (Get-Command py -ErrorAction SilentlyContinue) {
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'Using "py -3.11"' -ForegroundColor Green
        $pythonCmd = "py -3.11"
    }
} elseif (Get-Command python3.11 -ErrorAction SilentlyContinue) {
    Write-Host 'Using "python3.11"' -ForegroundColor Green
    $pythonCmd = "python3.11"
} else {
    Write-Host "No python 3.11 interpreter found" -ForegroundColor Red
    exit 1
}

# Создание виртуального окружения
$venvPath = Join-Path $RootDir -ChildPath "venv" | Join-Path -ChildPath $HostVenv
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath" -ForegroundColor Yellow
    Invoke-Expression "$pythonCmd -m venv `"$venvPath`""
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Активация виртуального окружения
$activateScript = Join-Path $venvPath -ChildPath "Scripts" | Join-Path -ChildPath "Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment" -ForegroundColor Green
    & $activateScript
} else {
    Write-Host "Virtual environment activation script not found" -ForegroundColor Red
    exit 1
}

# Создание файлов требований
$requirementsPath = Join-Path $RootDir -ChildPath "venv" | Join-Path -ChildPath "requirements.txt"
$freezePath = Join-Path $RootDir -ChildPath "venv" | Join-Path -ChildPath "requirements.freeze"

if (-not (Test-Path $requirementsPath)) {
    New-Item -Path $requirementsPath -ItemType File -Force | Out-Null
}

if (-not (Test-Path $freezePath)) {
    New-Item -Path $freezePath -ItemType File -Force | Out-Null
}

# Установка пакетов
Write-Host "Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel

Write-Host "Installing PyTorch with CUDA 12.8..." -ForegroundColor Yellow
pip install `
    --index-url https://download.pytorch.org/whl/nightly/cu128 `
    --pre torch torchvision torchaudio

Write-Host "Installing other dependencies..." -ForegroundColor Yellow
pip install `
    transformers==4.46.3 `
    accelerate>=0.30.0 `
    addict `
    easydict `
    einops

# Установка дополнительных требований
if (Test-Path $requirementsPath) {
    Write-Host "Installing requirements from $requirementsPath" -ForegroundColor Yellow
    pip install -r $requirementsPath
}

# Заморозка зависимостей
Write-Host "Freezing requirements..." -ForegroundColor Yellow
pip freeze > $freezePath

Write-Host "Virtual environment setup complete!" -ForegroundColor Green