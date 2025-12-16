#!/bin/bash

set -e

cd $(dirname "$0")
source load_env.sh
cd "../.."

HAS_UPLOAD=false
HAS_UPLOAD_VIDEO=false
OUTPUT_FILE=""
SIMULATION_MODE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --upload) 
            HAS_UPLOAD=true 
            ;;
        --upload-video) 
            HAS_UPLOAD_VIDEO=true 
            ;;
        --output) 
            OUTPUT_FILE="$2" 
            shift
            ;;
        --mode) 
            SIMULATION_MODE="$2" 
            shift
            ;;
        *) 
            echo "Unknown parameter: $1" 
            ;;
    esac
    shift 
done

if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="simulation_result_$(date +%Y%m%d_%H%M%S)"
    echo "Using a generated output filename ${OUTPUT_FILE}"
fi

rsync -avz \
  -e ./scripts/cloud/gcloud_rsync_wrapper.sh \
  --exclude '.git' \
  --exclude 'build_source' \
  --exclude '*.o' \
  --exclude '__pycache__' \
  ./src ./src-test ./include Makefile config*.json requirements.txt ./scripts ./tests \
  cuda-gpu:~/build_source/

if $HAS_UPLOAD; then
    gcloud compute ssh --zone=$ZONE cuda-gpu \
        --command "bash -lc 'cd build_source && make && ./bin/main -o ${OUTPUT_FILE} -m ${SIMULATION_MODE} && gcloud storage cp ${OUTPUT_FILE} gs://${STORAGE_BUCKET}/${OUTPUT_FILE}'"
else
    gcloud compute ssh --zone=$ZONE cuda-gpu \
        --command "bash -lc 'cd build_source && make && ./bin/main -o ${OUTPUT_FILE} -m ${SIMULATION_MODE}'"
fi

if $HAS_UPLOAD_VIDEO; then
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

            echo "Setup complete."
            ./scripts/postprocess/postprocess.py ${OUTPUT_FILE}
            gcloud storage cp ${OUTPUT_FILE}.mp4 gs://${STORAGE_BUCKET}/${OUTPUT_FILE}.mp4
EOF
fi
