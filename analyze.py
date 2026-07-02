#!/usr/bin/env python3
"""
琴乐启蒙AI导师 - 钢琴演奏音频分析微服务 (音乐体检版) - 轻量版
使用纯numpy+ffmpeg替代librosa, 适合Render Free 512MB内存
7层分析: 基础特征 → 风格判断 → 旋律伴奏 → 节奏对比 → 技术难度 → 情绪表现 → 演奏诊断
"""

import numpy as np
import subprocess
import os
import sys
import tempfile
import json
import functools
import pretty_midi
from flask import Flask, request, jsonify

# 强制stdout不缓冲
print = functools.partial(print, flush=True)

app = Flask(__name__)

REFERENCE_MIDI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.mid")
REFERENCE_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_audio.wav")

# ==================== 参考数据缓存 ====================
_ref_chroma = None
_ref_scores = None
_ref_features = None
_last_best_shift = 0
_ref_notes_set = None

# ==================== 纯numpy音频工具 (替代librosa) ====================
N_FFT = 2048
HOP_LENGTH = 512

def _load_audio(path, sr=22050):
    """替代 librosa.load(path, sr=22050, mono=True) — 用ffmpeg解码为float32"""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-f", "s16le", "-ar", str(sr), "-ac", "1", "-"],
        capture_output=True, timeout=300,
    )
    if len(proc.stdout) == 0:
        raise RuntimeError(
            "ffmpeg解码失败: " + proc.stderr.decode("utf-8", errors="ignore")[:300]
        )
    y = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return y, sr

def _rms(y, frame_length=512, hop_length=512):
    """替代 librosa.feature.rms(y=y)[0]
    按规格: np.array([np.sqrt(np.mean(y[i:i+512]**2)) for i in range(0, len(y)-512, 512)])
    对极短音频做了保底"""
    if len(y) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(y) <= frame_length:
        return np.array([float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))], dtype=np.float32)
    return np.array(
        [float(np.sqrt(np.mean(y[i:i + frame_length].astype(np.float64) ** 2)))
         for i in range(0, len(y) - frame_length, hop_length)],
        dtype=np.float32,
    )

def _chroma_cqt(y, sr=22050, hop_length=HOP_LENGTH, n_fft=N_FFT):
    """替代 librosa.feature.chroma_cqt — FFT实现12音级色谱
    对每帧做FFT, 把频率>80Hz的bin映射到pc=midi%12, 累加频谱能量"""
    if len(y) == 0:
        return np.zeros((12, 1), dtype=np.float32)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n_frames = 1 + (len(y) - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    valid_mask = freqs > 80
    valid_idx = np.where(valid_mask)[0]
    valid_freqs = freqs[valid_mask]
    midi = 69.0 + 12.0 * np.log2(valid_freqs / 440.0)
    pc = np.mod(np.round(midi).astype(int), 12)

    n_bins = len(freqs)
    mapping = np.zeros((12, n_bins), dtype=np.float32)
    for p, idx in zip(pc, valid_idx):
        mapping[p, idx] += 1.0

    chroma = np.zeros((12, n_frames), dtype=np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    for i in range(n_frames):
        frame = y[i * hop_length: i * hop_length + n_fft] * win
        spec = np.abs(np.fft.rfft(frame))
        chroma[:, i] = mapping @ (spec.astype(np.float32) ** 2)

    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma = chroma / norms
    return chroma

def _beat_track(y, sr=22050, hop_length=HOP_LENGTH):
    """替代 librosa.beat.beat_track — 找RMS峰值, 由峰值间隔得到tempo
    返回 (tempo: float, beat_frames: np.ndarray[int])"""
    rms = _rms(y, hop_length=hop_length)
    if len(rms) < 4:
        return 120.0, np.array([0, max(1, len(rms) // 2)], dtype=int)
    threshold = float(np.mean(rms))
    peaks = []
    for i in range(1, len(rms) - 1):
        v = rms[i]
        if v > threshold and v >= rms[i - 1] and v >= rms[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        beat_frames = np.array(
            list(range(0, len(rms), max(1, len(rms) // 4))), dtype=int
        )
        return 120.0, beat_frames
    peak_frames = np.array(peaks, dtype=int)
    intervals = np.diff(peak_frames)
    avg_interval_sec = float(np.mean(intervals)) * hop_length / sr
    if avg_interval_sec <= 0:
        return 120.0, peak_frames
    tempo = 60.0 / avg_interval_sec
    return float(tempo), peak_frames

def _onset_detect(y, sr=22050, hop_length=HOP_LENGTH):
    """替代 librosa.onset.onset_detect — 找RMS突增点"""
    rms = _rms(y, hop_length=hop_length)
    if len(rms) < 2:
        return np.array([], dtype=int)
    diff = np.diff(rms)
    mean_diff = float(np.mean(diff)) + 1e-9
    onsets = np.where(diff > mean_diff * 3.0)[0] + 1
    return onsets.astype(int)

def _trim(y, top_db=30):
    """替代 librosa.effects.trim — 找首个/末个 |y|>max*0.03, 截取中间
    返回 (y_trimmed, np.array([start, end]))"""
    if len(y) == 0:
        return y, np.array([0, 0])
    threshold = float(np.max(np.abs(y))) * 0.03
    if threshold <= 0:
        return y, np.array([0, len(y) - 1])
    indices = np.where(np.abs(y) > threshold)[0]
    if len(indices) == 0:
        return y, np.array([0, len(y) - 1])
    start = int(indices[0])
    end = int(indices[-1])
    return y[start:end + 1], np.array([start, end])

def _hpss(y):
    """替代 librosa.effects.hpss — 跳过谐波分离, 返回 (y, zeros)"""
    return y, np.zeros_like(y)

def _dtw(X, Y, metric='cosine'):
    """替代 librosa.sequence.dtw — 纯numpy实现DTW
    X: (12, n), Y: (12, m), 返回 (D: (n, m), wp: (path_len, 2))
    D[-1,-1] 为总累积距离, len(wp) 为路径长度
    内层用 numpy 向量化 (cumsum + minimum.accumulate), 外层逐行 Python 循环"""
    n = X.shape[1]
    m = Y.shape[1]
    if n == 0 or m == 0:
        D = np.zeros((max(1, n), max(1, m)), dtype=np.float64)
        wp = np.array([[max(0, n - 1), max(0, m - 1)]], dtype=int)
        return D, wp

    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    Yn = Y / (np.linalg.norm(Y, axis=0, keepdims=True) + 1e-9)
    sim = Xn.T @ Yn  # (n, m) 余弦相似度
    dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)

    D = np.full((n, m), np.inf, dtype=np.float64)
    D[0, 0] = dist[0, 0]
    D[0, :] = np.cumsum(dist[0, :])
    for i in range(1, n):
        D[i, 0] = D[i - 1, 0] + dist[i, 0]

    # 逐行用向量化推递推 D[i,j] = dist[i,j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    # 令 tmp[j] = dist[i,j] + min(D[i-1,j], D[i-1,j-1])  (j=0时 prev_padded=inf)
    # 则 D[i,j] = min(tmp[j], D[i,j-1] + dist[i,j])
    # 通过 cumsum(C) + minimum.accumulate 一次算完整行
    for i in range(1, n):
        prev_row = D[i - 1, :]
        prev_padded = np.empty_like(prev_row)
        prev_padded[0] = np.inf
        prev_padded[1:] = prev_row[:-1]
        tmp = dist[i, :] + np.minimum(prev_row, prev_padded)
        C = np.cumsum(dist[i, :])
        G = tmp - C
        E = np.minimum.accumulate(G)
        D[i, :] = E + C

    # 回溯路径: 从 (n-1, m-1) 走回 (0, 0)
    wp = []
    i, j = n - 1, m - 1
    wp.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            choices = (D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            c = int(np.argmin(choices))
            if c == 0:
                i -= 1
            elif c == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        wp.append((i, j))
    wp = np.array(wp[::-1], dtype=int)
    return D, wp

def _normalize(x, axis=0):
    """替代 librosa.util.normalize(x, axis=0)"""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)

def _spectral_centroid(y, sr=22050, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """替代 librosa.feature.spectral_centroid — FFT频谱质心, 返回 (1, n_frames)"""
    if len(y) == 0:
        return np.zeros((1, 1), dtype=np.float32)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n_frames = 1 + (len(y) - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    win = np.hanning(n_fft).astype(np.float32)
    centroids = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = y[i * hop_length: i * hop_length + n_fft] * win
        spec = np.abs(np.fft.rfft(frame))
        total = float(np.sum(spec)) + 1e-9
        centroids[i] = float(np.sum(freqs * spec)) / total
    return centroids.reshape(1, -1)

def _spectral_bandwidth(y, sr=22050, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """替代 librosa.feature.spectral_bandwidth — FFT频谱带宽, 返回 (1, n_frames)"""
    if len(y) == 0:
        return np.zeros((1, 1), dtype=np.float32)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n_frames = 1 + (len(y) - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    win = np.hanning(n_fft).astype(np.float32)
    bws = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = y[i * hop_length: i * hop_length + n_fft] * win
        spec = np.abs(np.fft.rfft(frame))
        total = float(np.sum(spec)) + 1e-9
        centroid = float(np.sum(freqs * spec)) / total
        bws[i] = float(np.sum(np.abs(freqs - centroid) * spec)) / total
    return bws.reshape(1, -1)

def _stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """替代 librosa.stft — 逐帧 np.fft.rfft(y[i:i+2048] * np.hanning(2048))
    返回 shape (n_fft//2+1, n_frames) complex"""
    n_bins = n_fft // 2 + 1
    if len(y) == 0:
        return np.zeros((n_bins, 1), dtype=complex)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n_frames = 1 + (len(y) - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1
    win = np.hanning(n_fft).astype(np.float32)
    out = np.zeros((n_bins, n_frames), dtype=complex)
    for i in range(n_frames):
        out[:, i] = np.fft.rfft(y[i * hop_length: i * hop_length + n_fft] * win)
    return out

def _fft_frequencies(sr=22050, n_fft=N_FFT):
    """替代 librosa.fft_frequencies(sr=sr)"""
    return np.fft.rfftfreq(n_fft, 1.0 / sr)

def _frames_to_time(frames, sr=22050, hop_length=HOP_LENGTH):
    """替代 librosa.frames_to_time(frames, sr=sr)"""
    return np.array(frames, dtype=float) * hop_length / sr

def _sync(data, frames, aggregate=np.mean):
    """替代 librosa.util.sync(chroma, beats_frames, aggregate=np.mean) — 对每个区间取mean"""
    frames = np.asarray(frames)
    if len(frames) < 2:
        return data[:, :1] if data.shape[1] > 0 else data
    out = np.zeros((data.shape[0], len(frames) - 1), dtype=data.dtype)
    for i in range(len(frames) - 1):
        s, e = int(frames[i]), int(frames[i + 1])
        if e <= s:
            e = s + 1
        if e > data.shape[1]:
            e = data.shape[1]
        if e <= s:
            out[:, i] = 0
        else:
            out[:, i] = aggregate(data[:, s:e], axis=1)
    return out


# ==================== 以下为原 analyze.py 业务逻辑 (librosa 调用全部替换为 helper) ====================

def synth_midi_simple(pm, fs=22050):
    duration = pm.get_end_time()
    n = int(duration * fs)
    audio = np.zeros(n, dtype=np.float32)
    for instrument in pm.instruments:
        for note in instrument.notes:
            s = int(note.start * fs); e = min(int(note.end * fs), n)
            if s >= e: continue
            t = np.arange(e - s) / fs
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            envelope = np.exp(-3.0 * t)
            audio[s:e] += 0.3 * envelope * np.sin(2 * np.pi * freq * t)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    return audio

def get_reference_chroma():
    global _ref_chroma
    if _ref_chroma is not None: return _ref_chroma
    try:
        if os.path.exists(REFERENCE_AUDIO):
            y, sr = _load_audio(REFERENCE_AUDIO, sr=22050)
            _ref_chroma = _chroma_cqt(y=y, sr=22050, hop_length=512)
            print(f"[ref] chroma loaded, shape={_ref_chroma.shape}")
            return _ref_chroma
        pm = pretty_midi.PrettyMIDI(REFERENCE_MIDI)
        try:
            audio = pm.fluidsynth(fs=22050)
            if len(audio) == 0: audio = synth_midi_simple(pm)
        except Exception:
            audio = synth_midi_simple(pm)
        _ref_chroma = _chroma_cqt(y=audio, sr=22050, hop_length=512)
        return _ref_chroma
    except Exception as e:
        print(f"[ref] chroma load failed: {e}")
        return None

def extract_basic_features(y, sr):
    """第1层: 基础音频特征"""
    try:
        tempo, beat_frames = _beat_track(y=y, sr=sr)
        tempo = float(tempo)
        beat_times = _frames_to_time(beat_frames, sr=sr)
        if len(beat_times) >= 2:
            intervals = np.diff(beat_times)
            tempo_stability = 1.0 - min(1.0, float(np.std(intervals) / (np.mean(intervals) + 1e-6)))
        else:
            tempo_stability = 0.5

        # 节拍判断 (3/4 vs 4/4 vs 2/4)
        if len(beat_frames) >= 8:
            rms = _rms(y=y)
            beat_rms = [rms[min(f, len(rms)-1)] for f in beat_frames]
            group3 = [np.mean(beat_rms[i:i+3]) - np.mean(beat_rms[i+1:i+4]) if i+4 <= len(beat_rms) else 0 for i in range(0, len(beat_rms)-3, 1)]
            group4 = [np.mean(beat_rms[i:i+4]) - np.mean(beat_rms[i+1:i+5]) if i+5 <= len(beat_rms) else 0 for i in range(0, len(beat_rms)-4, 1)]
            if abs(np.mean(group3)) > abs(np.mean(group4)):
                meter = "3/4"
            else:
                meter = "4/4"
        else:
            meter = "未知"

        # 音区 (频谱质心)
        spectral_centroid = float(np.mean(_spectral_centroid(y=y, sr=sr)))
        if spectral_centroid > 3000: register = "偏高音区"
        elif spectral_centroid > 1500: register = "中音区"
        else: register = "偏低音区"

        # 音量
        rms = _rms(y=y)
        avg_volume = float(np.mean(rms))
        volume_range = float(np.std(rms) / (np.mean(rms) + 1e-6))
        if volume_range > 0.4: volume_desc = "强弱变化大"
        elif volume_range > 0.2: volume_desc = "有一定强弱变化"
        else: volume_desc = "音量较平"

        # 音符密度 (onset检测)
        onset_frames = _onset_detect(y=y, sr=sr)
        onset_rate = float(len(onset_frames) / (len(y) / sr))
        if onset_rate > 4: density = "密集"
        elif onset_rate > 2: density = "适中"
        else: density = "稀疏"

        # 声音厚度 (频谱带宽)
        bandwidth = float(np.mean(_spectral_bandwidth(y=y, sr=sr)))
        if bandwidth > 2500: texture = "厚重和弦/复杂织体"
        elif bandwidth > 1500: texture = "简单伴奏"
        else: texture = "单旋律线条"

        return {
            "tempo": round(tempo, 0), "tempoStability": round(tempo_stability, 2),
            "meter": meter, "register": register, "registerValue": round(spectral_centroid, 0),
            "avgVolume": round(avg_volume, 3), "volumeRange": round(volume_range, 2), "volumeDesc": volume_desc,
            "onsetRate": round(onset_rate, 1), "density": density,
            "bandwidth": round(bandwidth, 0), "texture": texture,
        }
    except Exception as e:
        print(f"[features] 基础特征提取失败: {e}")
        return {"tempo": 120, "tempoStability": 0.5, "meter": "3/4", "register": "中音区", "volumeDesc": "适中", "density": "适中", "texture": "简单伴奏"}

def classify_style(f):
    """第2层: 音乐风格判断"""
    score = {"古典启蒙": 0, "巴赫复调": 0, "浪漫派": 0, "炫技练习曲": 0, "现代派": 0}
    if 100 <= f["tempo"] <= 140: score["古典启蒙"] += 2
    if f["texture"] == "简单伴奏": score["古典启蒙"] += 2
    if f["density"] == "适中": score["古典启蒙"] += 1
    if f["texture"] in ["厚重和弦/复杂织体", "简单伴奏"]: score["巴赫复调"] += 1
    if f["tempoStability"] > 0.7: score["巴赫复调"] += 2
    if f["meter"] == "3/4": score["巴赫复调"] += 2
    if f["tempoStability"] < 0.6: score["浪漫派"] += 1
    if f["volumeRange"] > 0.4: score["浪漫派"] += 2
    if f["density"] == "密集": score["炫技练习曲"] += 2
    if f["tempo"] > 150: score["炫技练习曲"] += 2
    best = max(score, key=score.get)
    return {"style": best, "scores": score, "desc": {
        "古典启蒙": "结构清楚、旋律规整、伴奏简单", "巴赫复调": "复调线条、多声部进行",
        "浪漫派": "旋律抒情、速度自由", "炫技练习曲": "音符密集、技术要求高", "现代派": "节奏尖锐、不协和音多"
    }.get(best, "")}

def analyze_melody_accompaniment(y, sr):
    """第3层: 旋律与伴奏关系"""
    try:
        chroma = _chroma_cqt(y=y, sr=sr, hop_length=512)
        dominant = np.argmax(chroma, axis=0)
        changes = np.sum(np.diff(dominant) != 0)
        melody_clarity = 1.0 - min(1.0, changes / max(len(dominant), 1) * 3)
        spec = np.abs(_stft(y))
        freqs = _fft_frequencies(sr=sr)
        high_mask = freqs > 1000
        low_mask = (freqs > 100) & (freqs <= 1000)
        high_energy = float(np.mean(spec[high_mask, :]))
        low_energy = float(np.mean(spec[low_mask, :]))
        balance = high_energy / (low_energy + 1e-6)
        if balance > 0.8:
            relation = "旋律突出, 伴奏适度"
            left_right = "右手旋律清晰, 左手伴奏配合"
        elif balance > 0.4:
            relation = "旋律与伴奏较为均衡"
            left_right = "左右手关系较平衡"
        else:
            relation = "伴奏偏重, 旋律被掩盖"
            left_right = "左手可能盖住右手, 需调整力度"
        return {
            "melodyClarity": round(melody_clarity, 2), "relation": relation,
            "leftRight": left_right, "balance": round(balance, 2),
            "isPolyphonic": bool(melody_clarity < 0.5),
        }
    except:
        return {"melodyClarity": 0.7, "relation": "旋律清晰", "leftRight": "右手旋律为主", "balance": 0.6, "isPolyphonic": False}

def compare_rhythm(user_feat, ref_feat):
    """第4层: 节奏和速度对比"""
    tempo_diff = user_feat["tempo"] - ref_feat["tempo"]
    if abs(tempo_diff) < 10: tempo_compare = "速度基本一致"
    elif tempo_diff > 0: tempo_compare = f"偏快(比标准快{abs(tempo_diff):.0f}BPM)"
    else: tempo_compare = f"偏慢(比标准慢{abs(tempo_diff):.0f}BPM)"
    stab_diff = user_feat["tempoStability"] - ref_feat["tempoStability"]
    if stab_diff > 0.1: stab_compare = "节奏更稳定"
    elif stab_diff > -0.1: stab_compare = "节奏稳定性相近"
    else: stab_compare = "节奏不够稳定, 有拖拍或抢拍"
    if user_feat["meter"] == ref_feat["meter"]: meter_match = True
    else: meter_match = False
    return {
        "tempoCompare": tempo_compare, "tempoDiff": round(tempo_diff, 0),
        "stabilityCompare": stab_compare, "meterMatch": bool(meter_match),
        "hasDance": bool(user_feat["meter"] == "3/4"), "danceDesc": "有舞曲感(3/4拍小步舞曲)" if user_feat["meter"] == "3/4" else "无舞曲感",
    }

def assess_difficulty(f, chroma=None):
    """第5层: 技术难度评估"""
    level = 0; reasons = []
    if f["tempo"] > 150: level += 3; reasons.append("速度快, 控制难度高")
    elif f["tempo"] > 120: level += 1
    if f["density"] == "密集": level += 3; reasons.append("音符密集, 技术要求高")
    elif f["density"] == "适中": level += 1
    if f["volumeRange"] > 0.4: level += 2; reasons.append("强弱变化大, 音色控制难")
    if f["texture"] == "厚重和弦/复杂织体": level += 2; reasons.append("多声部复杂, 听觉和控制难")
    try:
        if chroma is not None:
            jumps = np.sum(np.abs(np.diff(np.argmax(chroma, axis=0))) > 5)
            if jumps > len(chroma[0]) * 0.15: level += 1; reasons.append("大跳较多, 手位转换难")
    except: pass
    if level >= 6: diff = {"level": "高", "grade": "5-6级", "reasons": reasons}
    elif level >= 3: diff = {"level": "中", "grade": "2-4级", "reasons": reasons if reasons else ["适中难度"]}
    else: diff = {"level": "低", "grade": "1-2级", "reasons": reasons if reasons else ["入门难度, 适合启蒙"]}
    return diff

def assess_emotion(y, sr):
    """第6层: 情绪和表现"""
    try:
        centroid = float(np.mean(_spectral_centroid(y=y, sr=sr)))
        rms = _rms(y=y)
        energy_var = float(np.std(rms))
        chroma = _chroma_cqt(y=y, sr=sr, hop_length=512)
        major_energy = float(chroma[0].mean() + chroma[2].mean() + chroma[4].mean() + chroma[7].mean() + chroma[9].mean() + chroma[11].mean()) if chroma.shape[0] > 11 else 0.5
        minor_energy = float(chroma[1].mean() + chroma[3].mean() + chroma[5].mean() + chroma[6].mean() + chroma[8].mean() + chroma[10].mean()) if chroma.shape[0] > 11 else 0.5
        is_bright = major_energy > minor_energy
        if centroid > 2500: brightness = "明亮"
        elif centroid > 1500: brightness = "温暖"
        else: brightness = "暗淡"
        if energy_var > 0.15: energy = "戏剧化"
        elif energy_var > 0.08: energy = "有起伏"
        else: energy = "平稳"
        if is_bright and energy_var < 0.12: mood = "轻巧优雅, 儿童化舞曲感"
        elif is_bright and energy_var >= 0.12: mood = "活泼开朗, 有表现力"
        elif not is_bright and energy_var < 0.12: mood = "沉思内敛, 较成熟"
        else: mood = "深沉戏剧化, 复杂情绪"
        return {"brightness": brightness, "isBright": bool(is_bright), "energy": energy, "mood": mood, "centroid": round(centroid, 0)}
    except:
        return {"brightness": "温暖", "isBright": True, "energy": "平稳", "mood": "轻巧优雅", "centroid": 2000}

def diagnose_issues(user_feat, ref_feat, y, sr):
    """第7层: 演奏问题诊断"""
    issues = []
    if user_feat["tempoStability"] < 0.6:
        issues.append({"type": "节奏", "level": "明显", "desc": "节奏不够稳定, 可能有拖拍或抢拍, 建议用节拍器练习"})
    elif user_feat["tempoStability"] < 0.8:
        issues.append({"type": "节奏", "level": "轻微", "desc": "节奏偶有波动, 整体可控"})
    else:
        issues.append({"type": "节奏", "level": "良好", "desc": "节奏稳定, 拍点清晰"})
    tempo_diff = abs(user_feat["tempo"] - ref_feat["tempo"])
    if tempo_diff > 20:
        issues.append({"type": "速度", "level": "明显", "desc": f"速度与标准差异大({'偏快' if user_feat['tempo'] > ref_feat['tempo'] else '偏慢'}{tempo_diff:.0f}BPM)"})
    rms = _rms(y=y)
    silence_ratio = float(np.sum(rms < np.max(rms) * 0.12) / max(len(rms), 1))
    if silence_ratio > 0.25:
        issues.append({"type": "流畅性", "level": "明显", "desc": f"有明显停顿(静音{silence_ratio:.0%}), 建议分段熟练后连贯演奏"})
    elif silence_ratio > 0.10:
        issues.append({"type": "流畅性", "level": "轻微", "desc": "偶有停顿, 整体尚可"})
    else:
        issues.append({"type": "流畅性", "level": "良好", "desc": "演奏连贯流畅"})
    ma = analyze_melody_accompaniment(y, sr)
    if ma["balance"] < 0.4:
        issues.append({"type": "力度平衡", "level": "明显", "desc": "左手伴奏偏重, 盖住了右手旋律, 建议右手加强、左手放松"})
    elif ma["balance"] < 0.6:
        issues.append({"type": "力度平衡", "level": "轻微", "desc": "左右手力度基本平衡, 可更突出旋律"})
    else:
        issues.append({"type": "力度平衡", "level": "良好", "desc": "旋律突出, 伴奏适度"})
    if user_feat["volumeRange"] < 0.15:
        issues.append({"type": "音乐表现", "level": "需改进", "desc": "力度变化少, 缺乏强弱对比, 建议加强音乐表现力"})
    elif user_feat["volumeRange"] < 0.3:
        issues.append({"type": "音乐表现", "level": "一般", "desc": "有基本力度变化, 可更丰富"})
    else:
        issues.append({"type": "音乐表现", "level": "良好", "desc": "力度层次丰富, 音乐表现力好"})
    return issues

# ==================== 保留原有的评分函数 ====================
STUCK_PATTERNS = ["请上传学生演奏", "建议时长30", "五个维度进行AI辅助测评"]

def analyze_rhythm(y, sr):
    try:
        tempo, beat_frames = _beat_track(y=y, sr=sr)
        if len(beat_frames) < 3: return 60, "节拍点太少"
        beat_times = _frames_to_time(beat_frames, sr=sr)
        intervals = np.diff(beat_times)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
        score = max(40, min(100, int(100 - cv * 133)))
        comment = f" tempo约{float(tempo):.0f}BPM"
        if cv < 0.1: comment += ", 节奏非常稳定"
        elif cv < 0.2: comment += ", 节奏基本稳定, 偶有波动"
        else: comment += ", 节奏不够稳定, 需要数拍子练习"
        return score, comment
    except: return 65, "节奏分析受限"

def analyze_pitch(y, sr):
    try:
        ref_chroma = get_reference_chroma()
        perf_chroma = _chroma_cqt(y=y, sr=sr, hop_length=512)
        if ref_chroma is None: return 65, "参考数据不可用"
        ref_prof = ref_chroma.mean(axis=1); perf_prof = perf_chroma.mean(axis=1)
        ref_norm = ref_prof / (np.linalg.norm(ref_prof) + 1e-6)
        perf_norm = perf_prof / (np.linalg.norm(perf_prof) + 1e-6)
        cos_sim = float(np.dot(ref_norm, perf_norm))
        ref_frames = min(ref_chroma.shape[1], perf_chroma.shape[1], 600)
        ref_dom = np.argmax(ref_chroma[:, :ref_frames], axis=0)
        perf_dom = np.argmax(perf_chroma[:, :ref_frames], axis=0)
        matches = np.sum(np.abs(ref_dom - perf_dom) <= 1)
        frame_match = float(matches / ref_frames)
        pitch_score = cos_sim * 0.6 + frame_match * 0.4
        score = max(30, min(100, int(pitch_score * 100)))
        if cos_sim > 0.85: comment = f"音高分布高度匹配({cos_sim:.0%})"
        elif cos_sim > 0.70: comment = f"音高分布基本匹配({cos_sim:.0%})"
        elif cos_sim > 0.55: comment = f"音高匹配度一般({cos_sim:.0%}), 部分音符可能不准"
        else: comment = f"音高匹配度偏低({cos_sim:.0%}), 可能弹的不是这首曲子"
        print(f"[pitch] cos={cos_sim:.3f} frame={frame_match:.3f} score={score}")
        return score, comment
    except Exception as e:
        print(f"[pitch] error: {e}")
        return 65, "音高分析受限"

def analyze_fluency(y, sr):
    try:
        rms = _rms(y=y)
        threshold = np.max(rms) * 0.12
        silent = rms < threshold
        silence_ratio = float(np.sum(silent) / max(len(silent), 1))
        rms_diff = np.abs(np.diff(rms))
        mean_diff = float(np.mean(rms_diff)) + 1e-6
        jump_ratio = float(np.mean(rms_diff > mean_diff * 4))
        score = max(30, min(100, int(100 - silence_ratio * 150 - jump_ratio * 50)))
        if silence_ratio > 0.25: comment = f"有明显停顿(静音{silence_ratio:.0%})"
        elif silence_ratio > 0.10: comment = f"偶有停顿(静音{silence_ratio:.0%})"
        elif jump_ratio > 0.15: comment = "音符衔接不够连贯"
        else: comment = "演奏连贯流畅"
        return score, comment
    except: return 65, "流畅性分析受限"

def analyze_dynamics(rms):
    try:
        dynamic_range = float(np.std(rms) / (np.mean(rms) + 1e-6))
        changes = int(np.sum(np.diff(rms) != 0))
        change_rate = float(changes / max(len(rms), 1))
        score = max(30, min(100, int(dynamic_range * 60 + change_rate * 50)))
        if dynamic_range > 0.4: comment = f"力度变化丰富(动态{dynamic_range:.2f})"
        elif dynamic_range > 0.25: comment = f"有基本力度变化(动态{dynamic_range:.2f})"
        elif dynamic_range > 0.15: comment = f"力度变化偏少(动态{dynamic_range:.2f})"
        else: comment = f"力度较平(动态{dynamic_range:.2f})"
        return score, comment
    except: return 65, "力度分析受限"

def analyze_expression(y, sr, tempo):
    try:
        rms = _rms(y=y)
        mid = len(rms) // 2
        strong = np.mean(rms[:mid]) if mid > 0 else 0
        weak = np.mean(rms[mid:]) if mid > 0 else 0
        contrast = abs(strong - weak) / (strong + weak + 1e-6)
        tempo_score = max(30, min(100, 100 - abs(tempo - 120) * 2))
        score = int(tempo_score * 0.5 + min(contrast * 200, 100) * 0.5)
        score = max(40, min(100, score))
        comment = f"速度{'适中' if tempo_score > 70 else '偏快或偏慢'}, "
        comment += "强弱拍有区分" if contrast > 0.1 else "强弱拍对比不足"
        return score, comment
    except: return 65, "表现力分析受限"

def identify_song(y, sr):
    """曲目识别 — 用 DTW + 12次转调偏移找最佳匹配
    纯numpy实现的DTW对色谱做比对, 12个半音偏移各试一次, 取最低cost"""
    try:
        ref_chroma = get_reference_chroma()
        if ref_chroma is None: return True, 0.7, "参考数据不可用"

        # 提取用户音频色谱 (跳过谐波分离, 直接用y)
        y_h, _ = _hpss(y)
        perf_chroma = _chroma_cqt(y=y_h, sr=sr, hop_length=512)

        # 归一化色谱列
        ref_norm = _normalize(ref_chroma, axis=0)
        perf_norm = _normalize(perf_chroma, axis=0)

        # 12次转调偏移DTW — 找最佳匹配
        best_cost = 999
        best_shift = 0
        costs = []
        for shift in range(12):
            perf_shifted = np.roll(perf_norm, shift, axis=0)
            D, wp = _dtw(X=ref_norm, Y=perf_shifted, metric='cosine')
            cost = float(D[-1, -1] / len(wp))
            costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_shift = shift

        # cost越低越相似, 0=完全相同, 0.5=不相关
        # 转换为相似度: 1 - cost*2 (cost 0→100%, cost 0.25→50%, cost 0.5→0%)
        similarity = max(0, min(1, 1 - best_cost * 2))

        # 音高分布余弦相似度 (辅助判断)
        ref_prof = ref_chroma.mean(axis=1)
        perf_prof = perf_chroma.mean(axis=1)
        cos_sim = float(np.dot(ref_prof/(np.linalg.norm(ref_prof)+1e-6), perf_prof/(np.linalg.norm(perf_prof)+1e-6)))

        # 调性判断
        pcs = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        user_top3 = sorted([(pcs[i], float(perf_prof[i])) for i in range(12)], key=lambda x:-x[1])[:3]
        ref_top3 = sorted([(pcs[i], float(ref_prof[i])) for i in range(12)], key=lambda x:-x[1])[:3]

        print(f"[identify] DTW best_cost={best_cost:.4f} (shift={best_shift}), similarity={similarity:.3f}, cos_sim={cos_sim:.3f}")
        print(f"[identify] 用户top3: {user_top3}, 参考 top3: {ref_top3}")

        # 把best_shift存到全局变量供闸门使用
        global _last_best_shift
        _last_best_shift = best_shift

        if similarity > 0.60:
            return True, round(similarity, 2), f"曲目匹配度{similarity:.0%} (DTW cost={best_cost:.3f}, shift={best_shift})"
        elif similarity > 0.40:
            return True, round(similarity, 2), f"曲目匹配度偏低({similarity:.0%}), 可能不是Anh.114"
        else:
            return True, round(similarity, 2), f"曲目匹配度很低({similarity:.0%}), 很可能不是Anh.114"
    except Exception as e:
        print(f"[identify] error: {e}")
        return True, 0.7, "识别受限"

def get_reference_scores():
    global _ref_scores
    if _ref_scores is not None: return _ref_scores
    try:
        if not os.path.exists(REFERENCE_AUDIO): return None
        y, sr = _load_audio(REFERENCE_AUDIO, sr=22050)
        r_rhythm, _ = analyze_rhythm(y, sr)
        r_pitch, _ = analyze_pitch(y, sr)
        r_fluency, _ = analyze_fluency(y, sr)
        rms = _rms(y=y)
        r_dynamics, _ = analyze_dynamics(rms)
        tempo, _ = _beat_track(y=y, sr=sr)
        r_expression, _ = analyze_expression(y, sr, float(tempo))
        _ref_scores = {"rhythm": r_rhythm, "pitch": r_pitch, "fluency": r_fluency, "dynamics": r_dynamics, "expression": r_expression}
        print(f"[ref] 基准分数: {_ref_scores}")
        return _ref_scores
    except Exception as e:
        print(f"[ref] 基准分数计算失败: {e}")
        return None

def get_reference_features():
    global _ref_features
    if _ref_features is not None: return _ref_features
    try:
        if os.path.exists(REFERENCE_AUDIO):
            y, sr = _load_audio(REFERENCE_AUDIO, sr=22050)
            _ref_features = extract_basic_features(y, sr)
            print(f"[ref] 基准特征: {_ref_features}")
            return _ref_features
    except Exception as e:
        print(f"[ref] 基准特征计算失败: {e}")
    return {"tempo": 120, "tempoStability": 0.8, "meter": "3/4", "register": "中音区", "volumeDesc": "适中", "density": "适中", "texture": "简单伴奏"}

def normalize_to_ref(score, ref_score):
    if ref_score is None or ref_score <= 0: return score
    return max(30, min(100, int(score / ref_score * 100)))

def extract_notes_set(y, sr):
    try:
        chroma = _chroma_cqt(y=y, sr=sr, hop_length=512)
        profile = chroma.mean(axis=1)
        result = {}
        for midi in range(48, 75):
            pc = midi % 12
            if profile[pc] > np.mean(profile) * 0.6:
                result[str(midi)] = True
        return result
    except: return {}

def get_reference_notes_set():
    global _ref_notes_set
    if _ref_notes_set is not None: return _ref_notes_set
    try:
        if os.path.exists(REFERENCE_AUDIO):
            y, sr = _load_audio(REFERENCE_AUDIO, sr=22050)
            _ref_notes_set = extract_notes_set(y, sr)
            return _ref_notes_set
    except: pass
    return {}

# ==================== 主分析接口 ====================

def generate_student_feedback(total, issues):
    """改进8: 学生版反馈 — 温柔鼓励"""
    if total >= 85:
        prefix = "你弹得真棒！"
    elif total >= 70:
        prefix = "你弹得不错, 继续加油！"
    else:
        prefix = "别灰心, 多练几次一定会更好！"
    tips = []
    for i in issues:
        if i["level"] not in ("良好",) and i["type"] == "节奏":
            tips.append("跟着节拍器慢慢练, 数清楚拍子")
        if i["level"] not in ("良好",) and i["type"] == "力度平衡":
            tips.append("让右手像小歌唱家一样唱出来, 左手轻轻配合")
        if i["level"] not in ("良好",) and i["type"] == "流畅性":
            tips.append("把难弹的地方单独多练几遍")
    if not tips:
        tips.append("保持现在的水平, 可以挑战更有表现力的弹法")
    return prefix + " " + "；".join(tips) + "。"


def generate_teacher_feedback(total, issues, rhythm_compare, difficulty):
    """改进8: 老师版反馈 — 专业可执行"""
    parts = []
    if total >= 85:
        parts.append(f"学生完整度较好(总分{total}), ")
    elif total >= 70:
        parts.append(f"学生基本掌握曲目(总分{total}), ")
    else:
        parts.append(f"学生需加强练习(总分{total}), ")
    problems = [i for i in issues if i["level"] not in ("良好",)]
    if problems:
        parts.append("主要问题: " + "; ".join(f"{p['type']}({p['desc']})" for p in problems) + "。")
    else:
        parts.append("各维度表现均衡。")
    # 建议
    parts.append(" 建议: 使用Tomplay 60%速度片段循环练习, ")
    if difficulty.get("level") == "高":
        parts.append("先分手练熟再合手。")
    else:
        parts.append("重点改善音乐表现力。")
    return "".join(parts)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "未收到音频文件"}), 400
        audio_file = request.files["audio"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        try:
            y, sr = _load_audio(tmp_path, sr=22050)
            original_duration = len(y) / sr
            print(f"[analyze] 音频: {original_duration:.1f}秒")
            MAX_DURATION = 60
            if original_duration > MAX_DURATION:
                y = y[:int(MAX_DURATION * sr)]

            # ===== 先裁剪静音/噪音, 提取有效演奏片段 =====
            try:
                yt_trimmed, trim_idx = _trim(y, top_db=30)
                if len(yt_trimmed) > sr * 3:  # 裁剪后至少3秒
                    trimmed_duration = len(yt_trimmed) / sr
                    lead_silence = trim_idx[0] / sr
                    if lead_silence > 0.5:
                        print(f"[trim] 裁掉前{lead_silence:.1f}秒静音/噪音, 有效{trimmed_duration:.1f}秒")
                    y = yt_trimmed
            except:
                pass

            # ===== 改进1: 音频质量检测 =====
            rms_full = _rms(y=y)
            avg_vol = float(np.mean(rms_full))
            clipping_ratio = float(np.sum(np.abs(y) > 0.95) / len(y))
            noise_floor = float(np.percentile(rms_full, 10))
            snr = avg_vol / (noise_floor + 1e-6)

            # 改进3: 有效演奏片段检测
            yt, trim_idx = _trim(y, top_db=30)
            effective_duration = len(yt) / sr
            lead_silence = trim_idx[0] / sr
            silent_frames = rms_full < np.max(rms_full) * 0.12
            pauses = 0; in_pause = False; pause_len = 0
            for s in silent_frames:
                if s:
                    pause_len += 1
                    if not in_pause and pause_len > int(0.5 * sr / 512):
                        pauses += 1; in_pause = True
                else:
                    in_pause = False; pause_len = 0

            print(f"[quality] vol={avg_vol:.3f}, clip={clipping_ratio:.1%}, noise={noise_floor:.4f}, eff={effective_duration:.1f}s, pauses={pauses}")

            audio_quality = "good"; quality_issues = []
            if avg_vol < 0.02: audio_quality = "poor"; quality_issues.append("音量过小")
            if clipping_ratio > 0.05: audio_quality = "poor"; quality_issues.append("有明显爆音/削波")
            # 噪声检测: 如果噪声底线很高(>0.1), 说明背景噪声大
            if noise_floor > 0.1 and avg_vol < 0.15: audio_quality = "poor"; quality_issues.append("噪声较大")
            if effective_duration < 10: audio_quality = "poor"; quality_issues.append(f"有效演奏仅{effective_duration:.1f}秒")

            if audio_quality == "poor":
                return jsonify({
                    "isCorrectSong": None, "audioQuality": audio_quality,
                    "qualityIssues": quality_issues, "effectiveDuration": round(effective_duration, 1),
                    "message": "音频质量不足, 暂不评分",
                    "suggestion": "请靠近钢琴重新录制, 建议时长30-60秒, 环境安静, 避免说话声和杂音。",
                })

            # ===== 曲目匹配闸门 =====
            user_feat = extract_basic_features(y, sr)
            user_chroma = _chroma_cqt(y=y, sr=sr, hop_length=512)
            ref_feat = get_reference_features()
            is_correct, similarity, sim_comment = identify_song(y, sr)

            veto_reasons = []; veto_triggered = False
            pcs = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            g_major_pcs = {0, 2, 4, 6, 7, 9, 11}
            core_notes = {7, 9, 11, 2}
            ref_chroma_gate = get_reference_chroma()
            perf_prof = user_chroma.mean(axis=1)
            ref_prof = ref_chroma_gate.mean(axis=1) if ref_chroma_gate is not None else np.zeros(12)

            # 闸门1
            g_coverage = float(sum(perf_prof[i] for i in g_major_pcs) / (sum(perf_prof) + 1e-6))
            if g_coverage < 0.60:
                veto_triggered = True
                top_notes = [pcs[i] for i in np.argsort(perf_prof)[-3:][::-1]]
                veto_reasons.append(f"调性不匹配: 音高覆盖G大调仅{g_coverage:.0%}(主要音为{','.join(top_notes)}), Anh.114是G大调小步舞曲")

            # 闸门2
            core_present = sum(1 for i in core_notes if perf_prof[i] > np.mean(perf_prof) * 0.8)
            if core_present < 2 and not veto_triggered:
                veto_triggered = True
                veto_reasons.append(f"核心音组不匹配: G/A/B/D中仅{core_present}个明显出现")

            # 闸门2.5: B音能量+G大调覆盖率组合判断
            # B音偏低+G大调覆盖率也低 → 才拦截; B音偏低但G大调覆盖率高 → 可能是学生演奏问题, 放过
            ref_b_energy = ref_prof[11]
            user_b_energy = perf_prof[11]
            if user_b_energy < ref_b_energy * 0.50 and g_coverage < 0.65:
                veto_triggered = True
                veto_reasons.append(f"B音偏低且G大调覆盖率不足({g_coverage:.0%}), 不符合Anh.114特征")

            # 闸门2.6: D大调倾向检测 (C#偏高+D偏高+G大调覆盖率低)
            if perf_prof[1] > 0.15 and perf_prof[2] > 0.35 and g_coverage < 0.65:
                veto_triggered = True
                veto_reasons.append(f"疑似D大调(C#={perf_prof[1]:.2f}, D={perf_prof[2]:.2f}), 非G大调Anh.114")

            # 闸门2.7: 转调检测 — DTW需要转调才匹配说明可能不是G大调原调
            # 但如果cos_sim很高(>0.90), 音高分布已经证明是G大调, 不拦截
            cos_sim_val = float(np.dot(ref_prof/(np.linalg.norm(ref_prof)+1e-6), perf_prof/(np.linalg.norm(perf_prof)+1e-6)))
            if _last_best_shift != 0 and similarity < 0.60 and cos_sim_val < 0.92:
                veto_triggered = True
                veto_reasons.append(f"DTW需转调{_last_best_shift}个半音才匹配且音高相似度{cos_sim_val:.0%}, 非G大调原调Anh.114")

            # 闸门3
            dtw_sim_noshift = similarity
            try:
                ref_norm_gate = _normalize(ref_chroma_gate, axis=0) if ref_chroma_gate is not None else None
                y_h_gate, _ = _hpss(y)
                perf_chroma_gate = _chroma_cqt(y=y_h_gate, sr=sr, hop_length=512)
                perf_norm_gate = _normalize(perf_chroma_gate, axis=0)
                if ref_norm_gate is not None:
                    D_noshift, wp_noshift = _dtw(X=ref_norm_gate, Y=perf_norm_gate, metric='cosine')
                    dtw_cost_noshift = float(D_noshift[-1, -1] / len(wp_noshift))
                    dtw_sim_noshift = max(0, min(1, 1 - dtw_cost_noshift * 2))
                    if dtw_sim_noshift < 0.25:
                        veto_triggered = True
                        veto_reasons.append(f"旋律匹配度仅{dtw_sim_noshift:.0%}(无转调), 主旋律轮廓与Anh.114不一致")
            except: pass

            # 闸门4
            try:
                onset_frames = _onset_detect(y=y, sr=sr, hop_length=512)
                if len(onset_frames) >= 10:
                    user_fp = np.argmax(user_chroma[:, onset_frames[:20]], axis=0)
                    user_int = np.where(np.diff(user_fp) % 12 > 6, np.diff(user_fp) % 12 - 12, np.diff(user_fp) % 12)
                    ref_y_fp, _ = _load_audio(REFERENCE_AUDIO, sr=22050)
                    ref_of = _onset_detect(y=ref_y_fp, sr=22050, hop_length=512)
                    if len(ref_of) >= 10:
                        ref_fp = np.argmax(_chroma_cqt(y=ref_y_fp, sr=22050, hop_length=512)[:, ref_of[:20]], axis=0)
                        ref_int = np.where(np.diff(ref_fp) % 12 > 6, np.diff(ref_fp) % 12 - 12, np.diff(ref_fp) % 12)
                        ml = min(len(user_int), len(ref_int))
                        dm = float(np.sum(np.sign(user_int[:ml]) == np.sign(ref_int[:ml])) / ml)
                        if dm < 0.25 and similarity < 0.60:
                            veto_triggered = True
                            veto_reasons.append(f"开头旋律轮廓匹配率仅{dm:.0%}且旋律相似度{similarity:.0%}, 旋律走向与Anh.114不一致")
            except: pass

            user_meter = user_feat.get("meter", "未知")
            ref_meter = ref_feat.get("meter", "3/4")

            if veto_triggered:
                return jsonify({
                    "isCorrectSong": False, "similarity": round(similarity, 2), "totalScore": None,
                    "message": "曲目不匹配, 暂不评分", "reasons": veto_reasons,
                    "suggestion": "请上传巴赫《G大调小步舞曲 Anh.114》的演奏录音(3/4拍, G大调), 建议从曲目开头开始录制, 时长30-60秒。",
                    "userFeatures": {"tempo": user_feat.get("tempo"), "meter": user_meter},
                    "refFeatures": {"tempo": ref_feat.get("tempo"), "meter": ref_meter},
                    "audioQuality": audio_quality,
                })

            # ===== 改进2: 匹配可信度等级 =====
            # 通过了四道闸门说明调性+核心音+旋律轮廓都匹配, 最低给medium
            if similarity > 0.70:
                confidence = "high"
            else:
                confidence = "medium"

            # 低可信度只在闸门勉强通过时出现 (目前四道闸门已足够严格, 不再单独判low)

            if confidence == "low":
                return jsonify({
                    "isCorrectSong": True, "matchConfidence": "low", "totalScore": None,
                    "similarity": round(similarity, 2),
                    "message": "曲目匹配可信度较低, 暂不评分",
                    "suggestion": "系统无法确认是否为Anh.114, 请尝试从曲目开头录制, 确保环境安静、时长30秒以上。",
                    "audioQuality": audio_quality,
                })

            print(f"[gate] 通过, 可信度={confidence}")

            # ===== 改进4: 扣分制评分 =====
            rhythm_raw, rhythm_comment = analyze_rhythm(y, sr)
            pitch_raw, pitch_comment = analyze_pitch(y, sr)
            fluency_raw, fluency_comment = analyze_fluency(y, sr)
            rms = _rms(y=y)
            dynamics_raw, dynamics_comment = analyze_dynamics(rms)
            tempo, _ = _beat_track(y=y, sr=sr)
            expression_raw, expression_comment = analyze_expression(y, sr, float(tempo))

            ref_scores = get_reference_scores()
            rhythm_score = normalize_to_ref(rhythm_raw, ref_scores["rhythm"]) if ref_scores else rhythm_raw
            pitch_score = normalize_to_ref(pitch_raw, ref_scores["pitch"]) if ref_scores else pitch_raw
            fluency_score = normalize_to_ref(fluency_raw, ref_scores["fluency"]) if ref_scores else fluency_raw
            dynamics_score = normalize_to_ref(dynamics_raw, ref_scores["dynamics"]) if ref_scores else dynamics_raw
            expression_score = normalize_to_ref(expression_raw, ref_scores["expression"]) if ref_scores else expression_raw

            # ===== 改进5: 限制异常高分 =====
            score_cap = 100
            if audio_quality != "good": score_cap = min(score_cap, 90)
            if confidence == "medium": score_cap = min(score_cap, 85)
            if similarity < 0.95: score_cap = min(score_cap, 95)
            expression_cap = min(score_cap, 90) if similarity < 0.95 else 95

            rhythm_score = min(rhythm_score, score_cap)
            pitch_score = min(pitch_score, score_cap)
            fluency_score = min(fluency_score, score_cap)
            dynamics_score = min(dynamics_score, score_cap)
            expression_score = min(expression_score, expression_cap)

            total_score = round(rhythm_score * 0.15 + pitch_score * 0.40 + fluency_score * 0.10 + dynamics_score * 0.15 + expression_score * 0.20)

            # 扣分明细
            deductions = []
            if rhythm_score < 95: deductions.append({"dim": "节奏稳定", "deduction": 95 - rhythm_score, "reason": rhythm_comment})
            if pitch_score < 95: deductions.append({"dim": "音高准确", "deduction": 95 - pitch_score, "reason": pitch_comment})
            if fluency_score < 95: deductions.append({"dim": "完整流畅", "deduction": 95 - fluency_score, "reason": fluency_comment})
            if dynamics_score < 95: deductions.append({"dim": "力度层次", "deduction": 95 - dynamics_score, "reason": dynamics_comment})
            if expression_score < 90: deductions.append({"dim": "音乐表现", "deduction": 90 - expression_score, "reason": expression_comment})

            # 音乐体检
            style = classify_style(user_feat)
            melody = analyze_melody_accompaniment(y, sr)
            rhythm_compare = compare_rhythm(user_feat, ref_feat)
            difficulty = assess_difficulty(user_feat, user_chroma)
            emotion = assess_emotion(y, sr)
            issues = diagnose_issues(user_feat, ref_feat, y, sr)
            notes_played = extract_notes_set(y, sr)
            notes_expected = get_reference_notes_set()

            # ===== 改进6: 老师复核点 =====
            review_points = []
            if pitch_score < 85: review_points.append("个别音高是否为错音, AI识别可能受录音质量影响, 建议老师结合现场听辨")
            if fluency_score < 85: review_points.append("乐句连接处的停顿是技术问题还是读谱停顿, 需要老师现场判断")
            if dynamics_score < 85: review_points.append("左手是否真正盖住右手, 需结合钢琴现场音响判断")
            if expression_score < 85: review_points.append("音乐表现力的评分较为主观, 建议老师结合整体课堂表现综合判断")
            if confidence == "medium": review_points.append("本次曲目匹配可信度为中等, 建议老师确认学生确实在弹Anh.114")
            if not review_points: review_points.append("各维度表现较好, 无需特别复核")

            # ===== 改进8: 双反馈 =====
            student_feedback = generate_student_feedback(total_score, issues)
            teacher_feedback = generate_teacher_feedback(total_score, issues, rhythm_compare, difficulty)

            result = {
                "isCorrectSong": True, "matchConfidence": confidence,
                "similarity": round(similarity, 2), "totalScore": total_score,
                "audioQuality": audio_quality,
                "effectiveDuration": round(effective_duration, 1),
                "leadSilence": round(lead_silence, 1), "pauses": pauses,
                "tempo": round(float(tempo), 0),
                "duration": round(len(y) / sr, 1),
                "originalDuration": round(original_duration, 1),
                "truncated": original_duration > MAX_DURATION,
                "scores": {
                    "rhythm": {"score": rhythm_score, "comment": rhythm_comment, "max": score_cap},
                    "pitch": {"score": pitch_score, "comment": pitch_comment, "max": score_cap},
                    "fluency": {"score": fluency_score, "comment": fluency_comment, "max": score_cap},
                    "dynamics": {"score": dynamics_score, "comment": dynamics_comment, "max": score_cap},
                    "expression": {"score": expression_score, "comment": expression_comment, "max": expression_cap},
                },
                "deductions": deductions,
                "reviewPoints": review_points,
                "studentFeedback": student_feedback,
                "teacherFeedback": teacher_feedback,
                "report": {
                    "basicFeatures": user_feat, "style": style, "melody": melody,
                    "rhythmCompare": rhythm_compare, "difficulty": difficulty,
                    "emotion": emotion, "issues": issues,
                },
                "notesPlayed": notes_played, "notesExpected": notes_expected,
            }
            print(f"[analyze] 完成: 总分={total_score}, 可信度={confidence}, 质量={audio_quality}")
            return jsonify(result)

        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"[analyze] 错误: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "referenceLoaded": get_reference_chroma() is not None})

@app.route("/speech_to_text", methods=["POST"])
def speech_to_text():
    try:
        if "audio" not in request.files: return jsonify({"error": "未收到音频文件"}), 400
        audio_file = request.files["audio"]
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            audio_file.save(tmp.name); tmp_path = tmp.name
        try:
            wav_path = tmp_path.replace(".webm", ".wav")
            result = subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path], capture_output=True, timeout=30)
            if result.returncode != 0: return jsonify({"error": "音频转换失败"}), 500
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source: audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language="zh-CN")
                return jsonify({"text": text, "engine": "google"})
            except sr.UnknownValueError: return jsonify({"text": "", "error": "未能识别语音内容"})
            except sr.RequestError as e: return jsonify({"text": "", "error": f"识别服务不可用: {e}"})
        finally:
            for p in [tmp_path, wav_path]:
                if os.path.exists(p): os.unlink(p)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[启动] 预加载参考数据...")
    get_reference_chroma()
    get_reference_scores()
    get_reference_features()
    get_reference_notes_set()
    print("[启动] 音频分析微服务(轻量版)启动: http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
