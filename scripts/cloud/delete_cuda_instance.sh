#!/bin/bash

cd $(dirname "$0")
source load_env.sh
cd "../.."

gcloud compute instances delete cuda-gpu --zone=$ZONE
