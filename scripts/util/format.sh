#!/bin/bash

cd "$(dirname "$0")/../.."
source .venv/bin/activate
find . -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.h" -o -name "*.json" \) -exec clang-format -i {} +
black ./**/*.py
