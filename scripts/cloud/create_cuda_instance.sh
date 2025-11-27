#!/bin/bash

cd $(dirname "$0")
source load_env.sh

gcloud compute instances create cuda-gpu --project=$PROJECT_ID \
  --zone=$ZONE --machine-type=n1-standard-1 \
  --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
  --maintenance-policy=TERMINATE --provisioning-model=STANDARD \
  --service-account=$SERVICE_ACCOUNT \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --accelerator=count=1,type=nvidia-tesla-t4 \
  --create-disk=auto-delete=yes,boot=yes,device-name=cuda-gpu,image=projects/ml-images/global/images/c0-deeplearning-common-cu124-v20250325-debian-11-py310-conda,mode=rw,size=50,type=pd-balanced \
  --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring \
  --labels=goog-ec-src=vm_add-gcloud --reservation-affinity=any
