#!/usr/bin/env bash
# ===========================================================================
# RMPP classification support system - local launcher (Linux/macOS)
# Requires conda env rmpp-pack (fallbacks: rmpp-app, Pyradiomics)
# ===========================================================================
set -e
cd "$(dirname "$0")/.."

if conda env list | grep -q "rmpp-pack"; then
  ENV="rmpp-pack"
elif conda env list | grep -q "rmpp-app"; then
  ENV="rmpp-app"
elif conda env list | grep -q "Pyradiomics"; then
  ENV="Pyradiomics"
else
  echo "[Error] No conda environment found. Run: conda env create -f environment.yml"
  exit 1
fi

echo "Using environment: $ENV"
echo "Open: http://localhost:8501"
conda run -n "$ENV" streamlit run app.py \
  --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats=false
