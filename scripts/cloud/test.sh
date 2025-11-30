#!/bin/bash

set -e

cd $(dirname "$0")
source load_env.sh
cd "../.."

./scripts/cloud/clean_instance.sh
./scripts/cloud/compile_and_run.sh


gcloud compute ssh --zone=$ZONE cuda-gpu --command "export STORAGE_BUCKET='${STORAGE_BUCKET}' OUTPUT_FILE='${OUTPUT_FILE}'; bash -s" <<'EOF'
        set -e  # Exit immediately if any command fails
        
        cd build_source;

        export PATH="$HOME/.cargo/bin:$PATH"

        if command -v uv &> /dev/null; then
            echo "uv is already installed. Skipping installation."
        else
            echo "uv not found. Installing..."
            if ! command -v curl &> /dev/null; then
                echo "curl not found. Installing..."
                sudo apt install -y curl
            fi
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi

        source $HOME/.local/bin/env

        echo "Creating virtual environment..."
        uv venv --python 3.11 --clear

        source .venv/bin/activate

        if [ -f "requirements.txt" ]; then
            echo "Installing requirements from requirements.txt..."
            uv pip install -r requirements.txt
        else
            echo "Warning: requirements.txt not found. Skipping package installation."
        fi

        pytest -v
EOF
