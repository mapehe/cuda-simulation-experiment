#!/bin/bash

cd $(dirname "$0")
source load_env.sh
cd "../.."

HAS_UPLOAD=false
HAS_UPLOAD_VIDEO=false

for arg in "$@"; do
    if [[ "$arg" == "--upload" ]]; then
        HAS_UPLOAD=true
    fi
    if [[ "$arg" == "--upload-video" ]]; then
        HAS_UPLOAD_VIDEO=true
    fi
done

rsync -avz \
  -e ./scripts/cloud/gcloud_rsync_wrapper.sh \
  --exclude '.git' \
  --exclude 'build_source' \
  --exclude '*.o' \
  ./src ./include Makefile config.json configOverrides.json requirements.txt ./scripts \
  cuda-gpu:~/build_source/

if $HAS_UPLOAD; then
    gcloud compute ssh --zone=$ZONE cuda-gpu \
        --command "bash -lc 'cd build_source && make run && gcloud storage cp simulation_result gs://${STORAGE_BUCKET}/simulation_result_\$(date +%Y%m%d_%H%M%S)'"
else
    gcloud compute ssh --zone=$ZONE cuda-gpu \
        --command "bash -lc 'cd build_source && make run'"
fi

if $HAS_UPLOAD_VIDEO; then
    gcloud compute ssh --zone=$ZONE cuda-gpu --command "export STORAGE_BUCKET='${STORAGE_BUCKET}'; bash -s" <<'EOF'
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

            echo "Setup complete."
            ./scripts/postprocess/postprocess.py simulation_result
            gcloud storage cp simulation_result.mp4 gs://${STORAGE_BUCKET}/simulation_result_$(date +%Y%m%d_%H%M%S).mp4
EOF
fi
