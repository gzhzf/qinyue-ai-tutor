#!/bin/bash
cd "$(dirname "$0")"

# 启动 Python 微服务 (后台)
python3 analyze.py &
PYTHON_PID=$!
echo "Python微服务启动, PID=$PYTHON_PID"

# 等待 Python 就绪
sleep 8

# 启动 Node 服务 (前台)
exec NODE_OPTIONS="" node server.js
