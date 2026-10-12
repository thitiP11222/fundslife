$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed. Dashboard not started." }
python -m streamlit run final_app.py
