/**
 * 琴乐启蒙AI导师 语音版 - 后端代理
 * 1. 调用元器官方OpenAPI
 * 2. 智能降级: 检测到智能体固定回复时, 自动切换本地知识库
 */

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import multer from "multer";
import { spawn } from "child_process";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));
// 静态文件带no-cache头, 防止浏览器缓存旧版本
app.use(express.static("public", {
  setHeaders: function(res) {
    res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
    res.setHeader("Pragma", "no-cache");
  },
}));

const PORT = process.env.PORT || 3000;
const YUANQI_API_URL = process.env.YUANQI_API_URL || "https://yuanqi.tencent.com/openapi/v1/agent/chat/completions";
const YUANQI_API_KEY = process.env.YUANQI_API_KEY;
const YUANQI_ASSISTANT_ID = process.env.YUANQI_ASSISTANT_ID;

// 加载本地知识库 (降级用)
let localKB = [];
try {
  const kbPath = path.join(__dirname, "knowledge.json");
  localKB = JSON.parse(fs.readFileSync(kbPath, "utf-8"));
  console.log("[kb] 本地知识库已加载, QA数:", localKB.length);
} catch(e) {
  console.warn("[kb] 本地知识库加载失败:", e.message);
}

// 智能体固定回复的特征关键词 (检测到这些词说明智能体未正常工作)
const STUCK_PATTERNS = ["请上传学生演奏", "建议时长30", "五个维度进行AI辅助测评"];

/* 本地知识库匹配 */
function matchLocalKB(message) {
  const msg = message.toLowerCase();
  let bestMatch = null;
  let bestScore = 0;
  for (const item of localKB) {
    let score = 0;
    for (const kw of item.keywords) {
      if (msg.includes(kw.toLowerCase())) score += kw.length;
    }
    if (score > bestScore) { bestScore = score; bestMatch = item; }
  }
  return bestScore > 0 ? bestMatch : null;
}

/* 检测是否为固定回复 */
function isStuckReply(text) {
  return STUCK_PATTERNS.some(function(p) { return text.includes(p); });
}

/* 健康检查 */
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    apiKeyConfigured: !!YUANQI_API_KEY && YUANQI_API_KEY !== "在此填写你的API密钥",
    assistantId: YUANQI_ASSISTANT_ID || "未配置",
    localKBSize: localKB.length,
  });
});

/* 对话接口 */
app.post("/api/chat", async (req, res) => {
  try {
    const { messages, userId } = req.body;
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "messages 不能为空" });
    }

    // 提取最后一条用户消息文本
    const lastMsg = messages[messages.length - 1];
    let userText = "";
    if (lastMsg && lastMsg.content && Array.isArray(lastMsg.content)) {
      userText = lastMsg.content.filter(function(c) { return c.type === "text"; }).map(function(c) { return c.text; }).join("");
    } else if (lastMsg && typeof lastMsg.content === "string") {
      userText = lastMsg.content;
    }

    // 先尝试调用元器API
    let yuanqiAnswer = "";
    let yuanqiFailed = false;

    if (YUANQI_API_KEY && YUANQI_API_KEY !== "在此填写你的API密钥") {
      try {
        const payload = {
          assistant_id: YUANQI_ASSISTANT_ID,
          user_id: userId || "qinyue-voice-demo-user",
          stream: false,
          messages: messages,
        };
        console.log("[chat] 调用元器API, 消息数:", messages.length);
        const response = await fetch(YUANQI_API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: "Bearer " + YUANQI_API_KEY,
            "X-Source": "openapi",
          },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (response.ok && data.choices && data.choices[0]) {
          const content = data.choices[0].message.content;
          if (typeof content === "string") yuanqiAnswer = content;
          else if (Array.isArray(content)) yuanqiAnswer = content.filter(function(c) { return c.type === "text"; }).map(function(c) { return c.text; }).join("");
        }
        // 检测是否为固定回复
        if (yuanqiAnswer && isStuckReply(yuanqiAnswer)) {
          console.log("[chat] 检测到智能体固定回复, 切换本地知识库");
          yuanqiFailed = true;
        }
      } catch(apiErr) {
        console.warn("[chat] 元器API调用失败:", apiErr.message);
        yuanqiFailed = true;
      }
    } else {
      yuanqiFailed = true;
    }

    // 如果元器API失败或固定回复, 使用本地知识库
    if (yuanqiFailed || !yuanqiAnswer) {
      const match = matchLocalKB(userText);
      if (match) {
        console.log("[chat] 本地知识库匹配:", match.question);
        return res.json({
          answer: match.answer,
          source: "local_kb",
          matched: match.question,
        });
      } else {
        // 本地也没匹配到, 返回默认提示
        return res.json({
          answer: "这个问题我还不太会回答, 试试问我关于节拍、音符、指法、Tomplay使用或巴赫小步舞曲的问题吧!",
          source: "fallback",
        });
      }
    }

    // 元器API正常返回
    res.json({
      answer: yuanqiAnswer,
      source: "yuanqi_api",
    });
  } catch (err) {
    console.error("[chat] 服务器错误:", err);
    res.status(500).json({ error: "服务器内部错误", detail: err.message });
  }
});

/* ===================== 文件上传中间件 (测评+语音识别共用) ===================== */
const upload = multer({ storage: multer.memoryStorage() });

/* ===================== 服务端语音识别 (iOS兼容) ===================== */
app.post("/api/speech-to-text", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "未收到音频文件" });
    }
    console.log("[stt] 收到录音:", req.file.originalname, "大小:", req.file.size);

    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
    formData.append("audio", blob, req.file.originalname || "audio.webm");

    const pyResponse = await fetch("http://localhost:5001/speech_to_text", {
      method: "POST",
      body: formData,
    });
    const result = await pyResponse.json();

    if (!pyResponse.ok) {
      return res.status(500).json(result);
    }
    res.json(result);
  } catch (err) {
    console.error("[stt] 服务器错误:", err);
    res.status(500).json({ error: "服务器内部错误", detail: err.message });
  }
});

/* ===================== 演奏测评接口 ===================== */

app.post("/api/assess", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "未收到音频文件" });
    }

    console.log("[assess] 收到音频:", req.file.originalname, "大小:", req.file.size);

    // 转发音频到Python分析微服务 (用 FormData)
    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
    formData.append("audio", blob, req.file.originalname || "audio.wav");

    const pyResponse = await fetch("http://localhost:5001/analyze", {
      method: "POST",
      body: formData,
    });

    const analysis = await pyResponse.json();

    if (!pyResponse.ok) {
      console.error("[assess] Python分析失败:", analysis);
      return res.status(500).json({ error: "音频分析失败", detail: analysis });
    }

    // 如果曲目不匹配, 直接返回
    if (!analysis.isCorrectSong) {
      return res.json(analysis);
    }

    // 曲目正确, 生成中文评语 (先调元器, 失败则用本地模板)
    let comment = generateAssessComment(analysis);

    // 尝试调元器API生成更丰富的评语
    if (YUANQI_API_KEY && YUANQI_API_KEY !== "在此填写你的API密钥") {
      try {
        const scoreSummary = [
          "节奏稳定: " + analysis.scores.rhythm.score + "分",
          "音高准确: " + analysis.scores.pitch.score + "分",
          "完整流畅: " + analysis.scores.fluency.score + "分",
          "力度层次: " + analysis.scores.dynamics.score + "分",
          "音乐表现: " + analysis.scores.expression.score + "分",
          "总分: " + analysis.totalScore + "分",
        ].join("\n");

        const assessMessage = "以下是学生演奏巴赫G大调小步舞曲Anh.114的AI测评结果, 请用亲切鼓励的语气生成一段课后讲评(200字以内):\n" + scoreSummary;

        const payload = {
          assistant_id: YUANQI_ASSISTANT_ID,
          user_id: "assess-user",
          stream: false,
          messages: [{ role: "user", content: [{ type: "text", text: assessMessage }] }],
        };

        const yuanqiResp = await fetch(YUANQI_API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: "Bearer " + YUANQI_API_KEY,
            "X-Source": "openapi",
          },
          body: JSON.stringify(payload),
        });

        const yuanqiData = await yuanqiResp.json();
        if (yuanqiResp.ok && yuanqiData.choices && yuanqiData.choices[0]) {
          const content = yuanqiData.choices[0].message.content;
          let yuanqiAnswer = "";
          if (typeof content === "string") yuanqiAnswer = content;
          else if (Array.isArray(content)) yuanqiAnswer = content.filter(function(c) { return c.type === "text"; }).map(function(c) { return c.text; }).join("");
          if (yuanqiAnswer && !isStuckReply(yuanqiAnswer)) {
            comment = yuanqiAnswer;
            console.log("[assess] 元器评语生成成功");
          }
        }
      } catch(e) {
        console.warn("[assess] 元器评语生成失败, 使用本地模板:", e.message);
      }
    }

    analysis.comment = comment;
    res.json(analysis);
  } catch (err) {
    console.error("[assess] 服务器错误:", err);
    res.status(500).json({ error: "服务器内部错误", detail: err.message });
  }
});

/* 生成本地测评评语 (降级方案) */
function generateAssessComment(a) {
  const s = a.scores;
  const total = a.totalScore;
  let level = "";
  if (total >= 85) level = "优秀";
  else if (total >= 70) level = "良好";
  else if (total >= 60) level = "及格";
  else level = "需要加油";

  let praises = [];
  let suggestions = [];

  if (s.rhythm.score >= 80) praises.push("节奏感很好");
  else suggestions.push("多用节拍器练习节奏稳定性");

  if (s.pitch.score >= 80) praises.push("音高很准确");
  else suggestions.push("注意看谱上的升降号");

  if (s.fluency.score >= 80) praises.push("演奏很流畅");
  else suggestions.push("分段练习后尝试连贯演奏");

  if (s.dynamics.score >= 80) praises.push("力度变化丰富");
  else suggestions.push("加强强弱拍的对比");

  if (s.expression.score >= 80) praises.push("音乐表现力强");
  else suggestions.push("多听示范录音感受音乐性");

  let comment = "演奏评级: " + level + " (" + total + "分)\n\n";
  if (praises.length > 0) comment += "值得表扬: " + praises.join(", ") + "!\n";
  if (suggestions.length > 0) comment += "练习建议: " + suggestions.join("; ") + "。\n";
  comment += "\n继续加油, 你会越来越棒!";
  return comment;
}

app.listen(PORT, () => {
  console.log("琴乐启蒙AI导师语音版已启动: http://localhost:" + PORT);
  console.log("本地知识库QA数:", localKB.length);
  if (!YUANQI_API_KEY || YUANQI_API_KEY === "在此填写你的API密钥") {
    console.log("提示: 未配置YUANQI_API_KEY, 将使用本地知识库模式");
  }
});
