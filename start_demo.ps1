# UrbanCadastral AI-ML Demo Launcher (PowerShell)
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "UrbanCadastral AI-ML: LightUNet Aerial Building Footprint Engine" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = $PSScriptRoot

Write-Host "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Yellow
Start-Process -FilePath "py" -ArgumentList "-3.13 ml/api/server.py" -WorkingDirectory $ProjectRoot

Write-Host "[2/2] Starting Vite Frontend on http://localhost:5174 ..." -ForegroundColor Yellow
Start-Process -FilePath "cmd.exe" -ArgumentList "/c npm run dev" -WorkingDirectory (Join-Path $ProjectRoot "frontend")

Start-Sleep -Seconds 2
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Application launched successfully!" -ForegroundColor Green
Write-Host "Frontend URL:  http://localhost:5174/" -ForegroundColor White
Write-Host "Backend API:   http://127.0.0.1:8000/" -ForegroundColor White
Write-Host "API Docs:      http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Green
