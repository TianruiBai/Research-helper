@echo off
setlocal EnableDelayedExpansion
REM =============================================================
REM  Research Field Intelligence Tool -- Launcher
REM  Auto-detects GPU platform, installs deps, starts services
REM =============================================================
echo.
echo  Research Field Intelligence Tool
echo  ================================
echo.

REM =============================================================
REM  STEP 1 -- Detect GPU platform
REM  Priority: NVIDIA (nvidia-smi) > Intel Arc > AMD dGPU > iGPU
REM  On laptops with hybrid graphics (iGPU + dGPU), the dGPU wins.
REM =============================================================
echo [1/5] Detecting GPU platform...
set GPU_PLATFORM=cpu
set "GPU_NAME=Unknown"
set "IGPU_NAME="
set "NVIDIA_VRAM_MB=0"

REM -- Step 1a: Try nvidia-smi first (most reliable for NVIDIA dGPUs)
REM    nvidia-smi always sees the dGPU even on Optimus laptops
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    for /f "usebackq tokens=1,* delims=," %%A in (`nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader^,nounits 2^>nul`) do (
        set "GPU_PLATFORM=nvidia"
        set "GPU_NAME=%%A"
        for /f "tokens=*" %%V in ("%%B") do set "NVIDIA_VRAM_MB=%%V"
    )
)

REM -- Step 1b: If no NVIDIA found, scan WMI for Intel Arc or AMD dGPU
if "!GPU_PLATFORM!"=="cpu" (
    for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"`) do (
        REM -- Skip virtual display adapters
        echo %%G | findstr /i "Virtual Parsec Remote" >nul 2>&1
        if errorlevel 1 (
            REM -- Check Intel Arc discrete GPU (not iGPU like UHD/Iris)
            echo %%G | findstr /i "Intel.*Arc" >nul 2>&1
            if not errorlevel 1 (
                set GPU_PLATFORM=intel_arc
                set "GPU_NAME=%%G"
            )
            REM -- Check AMD discrete GPU (RX series, not Radeon integrated)
            echo %%G | findstr /i "Radeon RX" >nul 2>&1
            if not errorlevel 1 (
                if "!GPU_PLATFORM!"=="cpu" (
                    set GPU_PLATFORM=amd
                    set "GPU_NAME=%%G"
                )
            )
            REM -- Track iGPU name for info display
            echo %%G | findstr /i "AMD Radeon Intel UHD Intel Iris" >nul 2>&1
            if not errorlevel 1 (
                set "IGPU_NAME=%%G"
            )
        )
    )
) else (
    REM -- NVIDIA found via nvidia-smi; still scan WMI briefly for iGPU name (info only)
    for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"`) do (
        echo %%G | findstr /i "AMD Radeon Intel UHD Intel Iris" >nul 2>&1
        if not errorlevel 1 set "IGPU_NAME=%%G"
    )
)

if "!GPU_PLATFORM!"=="nvidia" (
    echo [OK] NVIDIA dGPU detected: !GPU_NAME! ^(!NVIDIA_VRAM_MB! MB VRAM^)
    if defined IGPU_NAME echo [INFO] iGPU also present: !IGPU_NAME! ^(not used for LLM^)
) else if "!GPU_PLATFORM!"=="intel_arc" (
    echo [OK] Intel Arc GPU detected: !GPU_NAME!
    if defined IGPU_NAME echo [INFO] iGPU also present: !IGPU_NAME!
) else if "!GPU_PLATFORM!"=="amd" (
    echo [WARN] AMD dGPU detected: !GPU_NAME! ^(ROCm not supported on Windows -- CPU fallback^)
) else (
    if defined IGPU_NAME (
        echo [INFO] Only iGPU found: !IGPU_NAME! ^(CPU inference mode^)
    ) else (
        echo [WARN] No GPU found. CPU-only mode.
    )
)

REM =============================================================
REM  STEP 2 -- Locate conda
REM =============================================================
echo.
echo [2/5] Locating conda...
set "CONDA_EXE="

if exist "%USERPROFILE%\.conda\Scripts\conda.exe"      set "CONDA_EXE=%USERPROFILE%\.conda\Scripts\conda.exe"
if not defined CONDA_EXE (
if exist "C:\ProgramData\miniforge3\Scripts\conda.exe" set "CONDA_EXE=C:\ProgramData\miniforge3\Scripts\conda.exe"
)
if not defined CONDA_EXE (
if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe"  set "CONDA_EXE=%USERPROFILE%\miniforge3\Scripts\conda.exe"
)
if not defined CONDA_EXE (
if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set "CONDA_EXE=C:\ProgramData\miniconda3\Scripts\conda.exe"
)
if not defined CONDA_EXE (
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe"  set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
)
if not defined CONDA_EXE (
if exist "C:\ProgramData\anaconda3\Scripts\conda.exe"  set "CONDA_EXE=C:\ProgramData\anaconda3\Scripts\conda.exe"
)
if not defined CONDA_EXE (
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe"   set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
)
if not defined CONDA_EXE (
    where conda >nul 2>&1
    if not errorlevel 1 set "CONDA_EXE=conda"
)

if not defined CONDA_EXE (
    echo [ERROR] Conda not found. Please install Miniforge:
    echo         https://conda-forge.org/download/
    pause
    exit /b 1
)
echo [OK] Conda: %CONDA_EXE%

REM =============================================================
REM  STEP 3 -- Create / verify conda env "llm" + install packages
REM =============================================================
echo.
echo [3/5] Checking conda env "llm"...

set "CONDA_ENV_PYTHON="
set "CONDA_ENV_SCRIPTS="

if exist "%USERPROFILE%\.conda\envs\llm\python.exe" (
    set "CONDA_ENV_PYTHON=%USERPROFILE%\.conda\envs\llm\python.exe"
    set "CONDA_ENV_SCRIPTS=%USERPROFILE%\.conda\envs\llm\Scripts"
)
if not defined CONDA_ENV_PYTHON (
if exist "C:\ProgramData\miniforge3\envs\llm\python.exe" (
    set "CONDA_ENV_PYTHON=C:\ProgramData\miniforge3\envs\llm\python.exe"
    set "CONDA_ENV_SCRIPTS=C:\ProgramData\miniforge3\envs\llm\Scripts"
))
if not defined CONDA_ENV_PYTHON (
if exist "%USERPROFILE%\miniforge3\envs\llm\python.exe" (
    set "CONDA_ENV_PYTHON=%USERPROFILE%\miniforge3\envs\llm\python.exe"
    set "CONDA_ENV_SCRIPTS=%USERPROFILE%\miniforge3\envs\llm\Scripts"
))
if not defined CONDA_ENV_PYTHON (
if exist "C:\ProgramData\miniconda3\envs\llm\python.exe" (
    set "CONDA_ENV_PYTHON=C:\ProgramData\miniconda3\envs\llm\python.exe"
    set "CONDA_ENV_SCRIPTS=C:\ProgramData\miniconda3\envs\llm\Scripts"
))
if not defined CONDA_ENV_PYTHON (
if exist "%USERPROFILE%\miniconda3\envs\llm\python.exe" (
    set "CONDA_ENV_PYTHON=%USERPROFILE%\miniconda3\envs\llm\python.exe"
    set "CONDA_ENV_SCRIPTS=%USERPROFILE%\miniconda3\envs\llm\Scripts"
))

if not defined CONDA_ENV_PYTHON (
    echo [INFO] Env "llm" not found. Creating with Python 3.11...
    "%CONDA_EXE%" create -n llm python=3.11 -y
    if errorlevel 1 ( echo [ERROR] Failed to create conda env. & pause & exit /b 1 )
    if exist "%USERPROFILE%\.conda\envs\llm\python.exe" (
        set "CONDA_ENV_PYTHON=%USERPROFILE%\.conda\envs\llm\python.exe"
        set "CONDA_ENV_SCRIPTS=%USERPROFILE%\.conda\envs\llm\Scripts"
    )
    if not defined CONDA_ENV_PYTHON (
        echo [ERROR] Env created but python.exe not found. Check conda config.
        pause & exit /b 1
    )
)
echo [OK] Env python: %CONDA_ENV_PYTHON%

REM -- Install / verify PyTorch backend (must match GPU platform)
set "NEED_TORCH=0"
"%CONDA_ENV_PYTHON%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    set "NEED_TORCH=1"
    echo [INFO] PyTorch not found.
) else (
    REM -- Check that installed torch matches GPU platform
    if "!GPU_PLATFORM!"=="nvidia" (
        "%CONDA_ENV_PYTHON%" -c "import torch; assert '+cu' in torch.__version__" >nul 2>&1
        if errorlevel 1 (
            set "NEED_TORCH=1"
            echo [INFO] PyTorch found but is CPU-only build. Reinstalling with CUDA...
        )
    )
    if "!GPU_PLATFORM!"=="intel_arc" (
        "%CONDA_ENV_PYTHON%" -c "import torch; assert hasattr(torch,'xpu')" >nul 2>&1
        if errorlevel 1 (
            set "NEED_TORCH=1"
            echo [INFO] PyTorch found but missing XPU support. Reinstalling...
        )
    )
)
if "!NEED_TORCH!"=="1" (
    echo [INFO] Installing PyTorch for platform: !GPU_PLATFORM!...
    if "!GPU_PLATFORM!"=="intel_arc" (
        "%CONDA_ENV_PYTHON%" -m pip install --pre --upgrade "ipex-llm[xpu_2.6]" --extra-index-url https://download.pytorch.org/whl/xpu
    ) else if "!GPU_PLATFORM!"=="nvidia" (
        "%CONDA_ENV_PYTHON%" -m pip install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ) else (
        "%CONDA_ENV_PYTHON%" -m pip install torch torchvision torchaudio
    )
    if errorlevel 1 ( echo [ERROR] PyTorch install failed. & pause & exit /b 1 )
    echo [OK] PyTorch installed.
) else (
    echo [OK] PyTorch already present with correct backend.
)

REM -- Install / verify project requirements
"%CONDA_ENV_PYTHON%" -c "import fastapi, streamlit, sqlalchemy" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Project packages missing. Installing requirements...
    "%CONDA_ENV_PYTHON%" -m pip install -q -r requirements.txt -r requirements-llm.txt
    if errorlevel 1 ( echo [ERROR] requirements install failed. & pause & exit /b 1 )
    echo [OK] Requirements installed.
) else (
    echo [OK] Project requirements already satisfied.
)

REM =============================================================
REM  STEP 4 -- Download GGUF + launch llama-server (llama.cpp)
REM  Uses OpenAI-compatible API on port 8080
REM  Supports all GGUF architectures (qwen35, etc.) natively
REM =============================================================

REM -- check system memory and GPU VRAM to decide whether to enable LLM features
set "SKIP_LLM=0"
set "SYS_RAM_GB=0"
set "VRAM_GB=0"
REM -- Use temp files to avoid 'for /f' backtick parsing issues in cmd.
REM -- Use 1073741824 (not 1GB) so the token 'GB' never appears in this script.
powershell -NoProfile -Command "try{[math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1073741824)}catch{0}" >"%TEMP%\_ramt.tmp" 2>nul
if exist "%TEMP%\_ramt.tmp" ( set /p SYS_RAM_GB=<"%TEMP%\_ramt.tmp" & del "%TEMP%\_ramt.tmp" >nul 2>&1 )

REM -- VRAM: prefer nvidia-smi (accurate) over WMI AdapterRAM (uint32, caps at 4 GB)
if "!GPU_PLATFORM!"=="nvidia" if !NVIDIA_VRAM_MB! GTR 0 (
    set /a VRAM_GB=(!NVIDIA_VRAM_MB! + 1023^) / 1024
) else (
    REM -- Fallback: WMI AdapterRAM (unreliable for dedicated GPUs with >4 GB VRAM)
    powershell -NoProfile -Command "try{$v=(Get-CimInstance Win32_VideoController|Measure-Object -Property AdapterRAM -Maximum).Maximum;[math]::Floor($v/1073741824)}catch{0}" >"%TEMP%\_vramt.tmp" 2>nul
    if exist "%TEMP%\_vramt.tmp" ( set /p VRAM_GB=<"%TEMP%\_vramt.tmp" & del "%TEMP%\_vramt.tmp" >nul 2>&1 )
)

echo [INFO] System RAM: !SYS_RAM_GB! GB, dGPU VRAM: !VRAM_GB! GB

REM -- Ensure values are purely numeric; reset to 0 if not
echo %SYS_RAM_GB%| findstr /r "^[0-9][0-9]*$" >nul 2>&1 || set "SYS_RAM_GB=0"
echo %VRAM_GB%| findstr /r "^[0-9][0-9]*$" >nul 2>&1 || set "VRAM_GB=0"

REM -- Skip LLM only when truly insufficient: no usable GPU AND low RAM
REM    With 4+ GB VRAM partial offload works; with 32+ GB RAM CPU-only works
if %VRAM_GB% LSS 4 if %SYS_RAM_GB% LSS 16 (
    set "SKIP_LLM=1"
    echo [WARN] Insufficient resources -- LLM features disabled ^(need VRAM 4+ GB or RAM 16+ GB^)
)

echo.
echo [4/5] Setting up llama-server (llama.cpp)...

REM -- Common GGUF paths
set "GGUF_DIR=%USERPROFILE%\ollama-models"
set "GGUF_FILE=!GGUF_DIR!\Qwen3.5-9B-heretic-v2.Q4_K_M.gguf"
set "GGUF_URL=https://huggingface.co/crownelius/Crow-9B-Opus-4.6-Distill-Heretic_Qwen3.5/resolve/main/Qwen3.5-9B-heretic-v2.Q4_K_M.gguf"
set "MODEL_ALIAS=crow-9b-opus"
set "LLAMA_DEVICE_ARG="

REM -- Select llama.cpp binary variant by GPU platform
if "!GPU_PLATFORM!"=="intel_arc" (
    set "LLAMA_FILTER=*sycl*x64*"
    set "LLAMA_DIR=%USERPROFILE%\llama-cpp-sycl"
    set "LLAMA_NGL=99"
    set "LLAMA_CTX=32768"
    set "LLAMA_NP=1"
) else if "!GPU_PLATFORM!"=="nvidia" (
    set "LLAMA_FILTER=llama-*-bin-win-cuda-12*x64*"
    set "LLAMA_DIR=%USERPROFILE%\llama-cpp-cuda"
    set "LLAMA_DEVICE_ARG= --device CUDA0"
    REM -- 16 GB VRAM: model ~5.2 GB + KV-cache @ 65536 ctx ~8 GB = 13.2 GB total (safe)
    REM -- np=1 keeps KV cache at one slot; academic analysis is single-user / sequential
    set "LLAMA_NGL=99"
    set "LLAMA_CTX=65536"
    set "LLAMA_NP=1"
    if !VRAM_GB! LSS 8 (
        set "LLAMA_NGL=28"
        set "LLAMA_CTX=16384"
        set "LLAMA_NP=1"
    )
    if !VRAM_GB! LSS 6 set "LLAMA_NGL=18"
    if !VRAM_GB! LSS 4 set "LLAMA_NGL=8"
    if !VRAM_GB! LSS 2 set "LLAMA_NGL=0"
) else (
    set "LLAMA_FILTER=*avx2*x64*"
    set "LLAMA_DIR=%USERPROFILE%\llama-cpp-avx2"
    set "LLAMA_NGL=0"
    set "LLAMA_CTX=8192"
    set "LLAMA_NP=4"
)
set "LLAMA_EXE=!LLAMA_DIR!\llama-server.exe"
set "LLAMA_ZIP=%TEMP%\llama-cpp.zip"

REM -- Intel Arc: verify XPU and set Level Zero env vars
if "!GPU_PLATFORM!"=="intel_arc" (
    "%CONDA_ENV_PYTHON%" -c "import torch; assert torch.xpu.is_available()" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] XPU not accessible. Check driver at:
        echo        https://www.intel.com/content/www/us/en/download/785597/
    ) else (
        echo [OK] Intel Arc XPU backend ready.
    )
    set ONEAPI_DEVICE_SELECTOR=level_zero:0
    set ZES_ENABLE_SYSMAN=1
    set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
) else if "!GPU_PLATFORM!"=="nvidia" (
    "%CONDA_ENV_PYTHON%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] NVIDIA CUDA not accessible. Check CUDA drivers.
    ) else (
        echo [OK] NVIDIA CUDA ready.
    )
) else (
    echo [INFO] CPU-only mode.
)

REM -- Auto-download llama.cpp binary if llama-server.exe is missing
if not exist "!LLAMA_EXE!" (
    echo [INFO] Fetching latest llama.cpp release URL for !LLAMA_FILTER!...
    powershell -NoProfile -Command "$r=Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest'; $a=$r.assets|Where-Object{$_.name -like '!LLAMA_FILTER!'}; if($a){$a[0].browser_download_url}else{''}" > "%TEMP%\llama_url.txt" 2>nul
    set /p LLAMA_DL_URL=<"%TEMP%\llama_url.txt"
    del "%TEMP%\llama_url.txt" >nul 2>&1
    if "!LLAMA_DL_URL!"=="" (
        echo [WARN] Could not find a llama.cpp release matching !LLAMA_FILTER!
        echo        Download manually from https://github.com/ggml-org/llama.cpp/releases
        echo        and place llama-server.exe in: !LLAMA_DIR!
    ) else (
        echo [INFO] Downloading llama.cpp: !LLAMA_DL_URL!
        curl -L --progress-bar -o "!LLAMA_ZIP!" "!LLAMA_DL_URL!"
        if errorlevel 1 (
            echo [WARN] Download failed. LLM features degraded.
        ) else (
            if not exist "!LLAMA_DIR!" mkdir "!LLAMA_DIR!"
            powershell -NoProfile -Command "Expand-Archive -Path '!LLAMA_ZIP!' -DestinationPath '!LLAMA_DIR!' -Force; Remove-Item '!LLAMA_ZIP!'"
            REM -- Flatten sub-folder when zip extracts into a named sub-directory
            if not exist "!LLAMA_EXE!" (
                powershell -NoProfile -Command "$f=Get-ChildItem '!LLAMA_DIR!' -Recurse -Filter 'llama-server.exe'|Select-Object -First 1; if($f){Get-ChildItem $f.DirectoryName|Copy-Item -Destination '!LLAMA_DIR!' -Force}"
            )
            echo [OK] llama.cpp installed to !LLAMA_DIR!
        )
    )
) else (
    echo [OK] llama-server already installed: !LLAMA_EXE!
)

REM -- NVIDIA CUDA runtime DLL package (cudart/cublas) is required for ggml-cuda backend
if "!GPU_PLATFORM!"=="nvidia" if exist "!LLAMA_EXE!" (
    if not exist "!LLAMA_DIR!\cudart64_12.dll" (
        echo [INFO] CUDA runtime DLLs not found. Downloading cudart package...
        set "CUDART_ZIP=%TEMP%\llama-cudart.zip"
        set "CUDART_URL=https://github.com/ggml-org/llama.cpp/releases/latest/download/cudart-llama-bin-win-cuda-12.4-x64.zip"
        echo [INFO] Downloading CUDA runtime DLLs: !CUDART_URL!
        curl -fL --progress-bar -o "!CUDART_ZIP!" "!CUDART_URL!"
        if errorlevel 1 (
            echo [WARN] cudart package download failed.
        ) else (
            powershell -NoProfile -Command "Expand-Archive -Path '!CUDART_ZIP!' -DestinationPath '!LLAMA_DIR!' -Force; Remove-Item '!CUDART_ZIP!'"
            echo [OK] CUDA runtime DLLs installed.
        )
    ) else (
        echo [OK] CUDA runtime DLLs already present.
    )

    REM -- Verify that llama.cpp can see GPU devices for offload
    set "PATH=!LLAMA_DIR!;%PATH%"
    "!LLAMA_EXE!" --list-devices > "%TEMP%\llama_devices.txt" 2>nul
    findstr /r /c:"CUDA[0-9][0-9]*:" "%TEMP%\llama_devices.txt" >nul
    if errorlevel 1 (
        echo [WARN] llama.cpp still cannot see a CUDA device. Model may run on CPU only.
        echo [WARN] Check NVIDIA driver and ensure this process can access the dGPU.
    ) else (
        echo [OK] llama.cpp detected CUDA device^(s^) for offload.
    )
    del "%TEMP%\llama_devices.txt" >nul 2>&1
)

REM -- Probe optional web-search flag support (not available in all llama.cpp builds)
set "LLAMA_WEB_SEARCH_ARG="
if exist "!LLAMA_EXE!" (
    "!LLAMA_EXE!" --help 2>nul | findstr /i /c:"--web-search" >nul
    if not errorlevel 1 (
        set "LLAMA_WEB_SEARCH_ARG= --web-search"
        echo [INFO] llama-server supports --web-search.
    ) else (
        echo [INFO] llama-server build has no --web-search support. Starting without it.
    )
)

REM -- Download GGUF model if not on disk (skip if memory insufficient)
if "!SKIP_LLM!"=="0" (
    if not exist "!GGUF_FILE!" (
        echo [INFO] Downloading Crow-9B-Opus-4.6 Q4_K_M GGUF ^(~6 GB^)...
        echo [INFO] Source: !GGUF_URL!
        if not exist "!GGUF_DIR!" mkdir "!GGUF_DIR!"
        curl -L --progress-bar -C - -o "!GGUF_FILE!" "!GGUF_URL!"
        if errorlevel 1 (
            echo [WARN] GGUF download failed. Run manually:
            echo         curl -L -o "!GGUF_FILE!" "!GGUF_URL!"
        ) else (
            echo [OK] GGUF saved to !GGUF_FILE!
        )
    ) else (
        echo [OK] GGUF already on disk: !GGUF_FILE!
    )
) else (
    echo [INFO] LLM download skipped due to insufficient RAM/VRAM (see earlier warning)
)

REM -- Start llama-server if not already running on port 8080 (skip if LLM disabled)
curl -sf http://localhost:8080/health >nul 2>&1
if errorlevel 1 (
    if "!SKIP_LLM!"=="0" (
        if exist "!LLAMA_EXE!" if exist "!GGUF_FILE!" (
            if defined LLAMA_WEB_SEARCH_ARG (
                echo [INFO] Starting llama-server on http://127.0.0.1:8080 ^(ngl=!LLAMA_NGL!, ctx=!LLAMA_CTX!, np=!LLAMA_NP!, web-search^)...
            ) else (
                echo [INFO] Starting llama-server on http://127.0.0.1:8080 ^(ngl=!LLAMA_NGL!, ctx=!LLAMA_CTX!, np=!LLAMA_NP!^)...
            )
            if "!GPU_PLATFORM!"=="intel_arc" (
                start "llama-server (Intel Arc SYCL)" cmd /k "set PATH=!LLAMA_DIR!;%PATH% && set ONEAPI_DEVICE_SELECTOR=level_zero:0 && set ZES_ENABLE_SYSMAN=1 && set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 && "!LLAMA_EXE!" -m "!GGUF_FILE!" -c !LLAMA_CTX! -np !LLAMA_NP! --alias !MODEL_ALIAS! --port 8080 --host 127.0.0.1!LLAMA_WEB_SEARCH_ARG!"
            ) else (
                start "llama-server" cmd /k "set PATH=!LLAMA_DIR!;%PATH% && "!LLAMA_EXE!" -m "!GGUF_FILE!"!LLAMA_DEVICE_ARG! -ngl !LLAMA_NGL! -c !LLAMA_CTX! -np !LLAMA_NP! --alias !MODEL_ALIAS! --port 8080 --host 127.0.0.1!LLAMA_WEB_SEARCH_ARG!"
            )
            echo [INFO] Waiting for llama-server to become ready ^(up to 60 s^)...
            call :_wait_for_llama
        ) else (
            echo [WARN] llama-server or GGUF not found -- LLM features degraded.
            if not exist "!LLAMA_EXE!" echo [WARN]   Missing: !LLAMA_EXE!
            if not exist "!GGUF_FILE!" echo [WARN]   Missing: !GGUF_FILE!
        )
    ) else (
        echo [INFO] llama-server startup skipped due to insufficient RAM/VRAM
    )
) else (
    echo [OK] llama-server already running on http://localhost:8080
)

REM =============================================================
REM  STEP 5 -- Start application services
REM =============================================================
echo.
echo [5/5] Starting application services...
echo.

echo Starting FastAPI backend on http://localhost:8000 ...
start "FastAPI Backend" cmd /c ""%CONDA_ENV_PYTHON%" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

echo Starting Streamlit UI on http://localhost:8501 ...
start "Streamlit UI" cmd /c ""%CONDA_ENV_SCRIPTS%\streamlit.exe" run src/ui/app.py --server.port 8501"

echo.
echo  Both services are starting:
echo    Backend:  http://localhost:8000/docs
echo    UI:       http://localhost:8501
echo.
echo  Close this window to stop.
pause
endlocal
goto :eof

REM =============================================================
REM  Subroutine: poll llama-server /health up to 60 s
REM =============================================================
:_wait_for_llama
set /a _WAIT_TRIES=0
:_wfl_loop
timeout /t 5 /nobreak >nul
curl -sf http://localhost:8080/health >nul 2>&1
if not errorlevel 1 (
    echo [OK] llama-server ready on http://localhost:8080
    goto :eof
)
set /a _WAIT_TRIES+=1
if !_WAIT_TRIES! LSS 12 (
    echo [INFO]   Still loading... ^(!_WAIT_TRIES! x 5 s^)
    goto :_wfl_loop
)
echo [WARN] llama-server did not respond within 60 s.
echo [WARN] The model may still be loading -- check the server window.
echo [WARN] The app will start anyway and retry on first LLM use.
goto :eof