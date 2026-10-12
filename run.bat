@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Creando entorno virtual...
    python -m venv .venv || goto :error
)
rem Se reinstala si la instalacion anterior no termino (marca .venv\.instalado)
if not exist .venv\.instalado (
    echo Instalando dependencias...
    .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
    echo ok> .venv\.instalado
)
if not exist models\hand_landmarker.task goto :models
if not exist models\face_landmarker.task goto :models
goto :run
:models
.venv\Scripts\python.exe download_models.py || goto :error
:run
.venv\Scripts\python.exe app.py %*
pause
exit /b 0
:error
echo.
echo Algo fallo durante la instalacion. Revisa los mensajes de arriba.
pause
exit /b 1
