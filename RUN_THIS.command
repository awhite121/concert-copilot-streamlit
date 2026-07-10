#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Starting Encore AI..."
python3 -m streamlit run app.py --server.port 8517
