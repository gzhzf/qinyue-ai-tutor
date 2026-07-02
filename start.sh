#!/bin/bash
cd /app

# 启动 Python 微服务 (后台)
python3 analyze.py &
sleep 8

# 启动 Node 服务 (前台)
node server.js
