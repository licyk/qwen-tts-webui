python -V > $null
if ((!$?)){
   Write-Host "未找到 Python，终止运行"
   Read-Host | Out-Null
   exit 1
}

python -c "import venv" > $null
if ($? -eq 0){
   Write-Host "未找到 Python venv 模块，终止运行"
   Read-Host | Out-Null
   exit 1
}

Set-Location $PSScriptRoot

if (!(Test-Path .\venv\Scripts\activate.ps1)) {
   Write-Host "创建虚拟环境中..."
   python -m venv venv
}

.\venv\Scripts\activate.ps1

python launch.py @args

Read-Host | Out-Null