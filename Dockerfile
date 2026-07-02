FROM node:18-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install

# 只安装轻量Python依赖 (不需要librosa, 用numpy实现)
RUN pip3 install numpy flask SpeechRecognition pydub --break-system-packages

COPY . .

EXPOSE $PORT

CMD ["sh", "-c", "cd /app && python3 analyze.py > /tmp/py.log 2>&1 & sleep 10 && node server.js"]
