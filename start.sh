#!/bin/bash
# Render 启动脚本: 同时启动 Python 微服务和 Node 服务
cd "$(dirname "$0")"

# 安装 Node 依赖
npm install

# 安装 Python 3 和 ffmpeg (Render Node 环境可能没有)
apt-get update -qq && apt-get install -y -qq python3 python3-pip python3-dev ffmpeg 2>/dev/null

# 安装 Python 依赖
pip3 install librosa pretty_midi flask dtw-python SpeechRecognition pydub --break-system-packages

# 启动 Python 微服务 (后台)
python3 analyze.py &
PYTHON_PID=$!
echo "Python微服务启动, PID=$PYTHON_PID"

# 等待 Python 就绪
sleep 8

# 启动 Node 服务 (前台)
exec NODE_OPTIONS="" node server.js
