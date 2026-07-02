# Render 部署指南

## 方式一：GitHub + Render（推荐）

### 第1步：在本地电脑把代码推到 GitHub

```bash
# 在你的电脑上
cd qinyue-voice-demo
git init
git add .
git commit -m "琴乐启蒙AI导师语音版"
```

然后在 GitHub 创建一个新仓库（比如 `qinyue-ai-tutor`），推送：

```bash
git remote add origin https://github.com/你的用户名/qinyue-ai-tutor.git
git branch -M main
git push -u origin main
```

### 第2步：在 Render 创建 Web Service

1. 打开 https://render.com → 注册/登录
2. 点击 **New +** → **Web Service**
3. 连接你的 GitHub 仓库 `qinyue-ai-tutor`
4. 填写配置：

| 配置项 | 值 |
|--------|-----|
| Name | qinyue-ai-tutor |
| Runtime | Node |
| Build Command | `npm install && pip3 install librosa pretty_midi flask dtw-python SpeechRecognition pydub --break-system-packages` |
| Start Command | `bash start.sh` |
| Plan | Free |

5. 添加环境变量（Environment → Add Environment Variable）：

| Key | Value |
|-----|-------|
| YUANQI_API_KEY | XAG5rI40SS5zgvmVCr8RAuG9YAAkcJeT |
| YUANQI_API_URL | https://yuanqi.tencent.com/openapi/v1/agent/chat/completions |
| YUANQI_ASSISTANT_ID | 2070835229588909120 |
| NODE_OPTIONS | （留空） |

6. 点击 **Create Web Service**
7. 等待构建完成（约3-5分钟）
8. 部署成功后会得到固定地址：`https://qinyue-ai-tutor.onrender.com`

### 第3步：验证

打开 `https://qinyue-ai-tutor.onrender.com`，应该能看到完整页面。

---

## 注意事项

- Render Free Plan 会在 **15分钟无访问后休眠**，首次唤醒需约30秒
- Free Plan 内存 512MB，librosa 分析可能偏慢但能用
- 环境变量在 Render 后台设置，不要把 .env 推到 GitHub
- .gitignore 已经排除了 .env 和 node_modules

## 如果 pip install 失败

Render 的 Node 环境可能没有 Python 3。备选方案：

把 Build Command 改成：
```
npm install && apt-get update && apt-get install -y python3 python3-pip ffmpeg && pip3 install librosa pretty_midi flask dtw-python SpeechRecognition pydub --break-system-packages
```
