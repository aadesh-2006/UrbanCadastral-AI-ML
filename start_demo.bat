@echo off
title UrbanCadastral AI-ML Demo Launcher
echo ================================================================
echo UrbanCadastral AI-ML: LightUNet Aerial Building Footprint Engine
echo ================================================================
echo.
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "UrbanCadastral API Backend" py -3.13 ml/api/server.py

echo Starting Vite Frontend on http://localhost:5174 ...
cd frontend
start "UrbanCadastral Vite Frontend" cmd /c "npm run dev"
cd ..

echo.
echo ================================================================
echo Servers are launching!
echo Once ready, open your browser to: http://localhost:5174/
echo API Documentation available at: http://127.0.0.1:8000/docs
echo ================================================================
pause
