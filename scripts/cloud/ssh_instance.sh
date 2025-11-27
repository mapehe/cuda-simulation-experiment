#!/bin/bash

cd $(dirname "$0")
source load_env.sh

gcloud compute ssh cuda-gpu --zone=$ZONE
