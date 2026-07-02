FROM node:18-slim

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json ./
RUN npm install

RUN pip3 install numpy flask SpeechRecognition pydub --break-system-packages --no-cache-dir

COPY . .
EXPOSE $PORT

CMD ["sh", "-c", "cd /app && python3 analyze.py > /tmp/py.log 2>&1 & sleep 20 && node server.js"]
