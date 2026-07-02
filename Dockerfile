FROM node:18-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg \
    bash \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制 package.json 并安装 Node 依赖
COPY package.json ./
RUN npm install

# 安装 Python 依赖
RUN pip3 install librosa pretty_midi flask dtw-python SpeechRecognition pydub --break-system-packages

# 复制所有文件
COPY . .

# 暴露端口
EXPOSE $PORT

# 直接用 sh 启动 (不依赖 bash)
CMD ["sh", "-c", "cd /app && python3 analyze.py & sleep 8 && node server.js"]
