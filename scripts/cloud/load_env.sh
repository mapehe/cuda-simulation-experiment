#!/bin/bash

FILE="../../.env"

if [ ! -f "$FILE" ]; then
  echo "Error: .env does not exist in the project root (see CLOUD.md)."
    exit 1
fi

source $FILE

REQUIRED_VARS=(
  "USER"
  "PROJECT_ID"
  "SERVICE_ACCOUNT"
  "ZONE"
  "REGION"
  "STORAGE_BUCKET"
)

MISSING_FOUND=false

for VAR_NAME in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!VAR_NAME}" ]]; then
    echo "Error: variable '$VAR_NAME' is not set or is empty. Make sure your .env is set correctly (see CLOUD.md)."
    MISSING_FOUND=true
  fi
done

if [ "$MISSING_FOUND" = true ]; then
  echo "Validation failed: Missing required environment variables."
  exit 1
fi
