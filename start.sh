#!/bin/bash
# Render 启动脚本: 同时启动 Python 微服务和 Node 服务
cd "$(dirname "$0")"

# 安装 Python 依赖
pip3 install librosa pretty_midi flask dtw-python speech_recognition pydub --break-system-packages 2>/dev/null

# 安装 Node 依赖
npm install 2>/dev/null

# 启动 Python 微服务 (后台)
python3 analyze.py &
PYTHON_PID=$!
echo "Python微服务启动, PID=$PYTHON_PID"

# 等待 Python 就绪
sleep 5

# 启动 Node 服务 (前台)
exec NODE_OPTIONS="" node server.js
