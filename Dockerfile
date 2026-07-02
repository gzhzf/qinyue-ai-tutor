FROM node:18-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json ./
RUN npm install

# 分步安装Python依赖, 避免内存不足
RUN pip3 install numpy --break-system-packages --no-cache-dir
RUN pip3 install scipy --break-system-packages --no-cache-dir
RUN pip3 install flask --break-system-packages --no-cache-dir
RUN pip3 install six --break-system-packages --no-cache-dir
RUN pip3 install librosa --break-system-packages --no-cache-dir
RUN pip3 install pretty_midi --break-system-packages --no-cache-dir
RUN pip3 install dtw-python --break-system-packages --no-cache-dir
RUN pip3 install SpeechRecognition pydub --break-system-packages --no-cache-dir

COPY . .

EXPOSE $PORT

# 启动: 先Python后Node
CMD ["sh", "-c", "cd /app && python3 analyze.py > /tmp/py.log 2>&1 & sleep 15 && node server.js"]
