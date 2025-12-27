#!/bin/bash
# FunClip 启动脚本

export UPLOAD_DIR="/home/sunxiaomeng/Large_Model_Registration/backend/uploads"
export CUDA_VISIBLE_DEVICES=4

python funclip_api.py --host 0.0.0.0 --port 5000
