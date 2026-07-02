#!/usr/bin/env python3
"""
琴乐启蒙AI导师 - 钢琴演奏音频分析微服务 (Render轻量版)
用 numpy + scipy 替代 librosa, 大幅降低内存占用
"""

import os
import sys
import tempfile
import numpy as np
import subprocess
import json
from flask import Flask, request, jsonify

# 强制stdout不缓冲
import functools
print = functools.partial(print, flush=True)

app = Flask(__name__)

REFERENCE_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_audio.wav")
REFERENCE_MIDI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.mid")

_ref_chroma = None
_ref_features = None

def load_audio(filepath, sr=22050):
    """用ffmpeg加载音频为numpy数组 (替代librosa.load)"""
    try:
        cmd = ["ffmpeg", "-y", "-i", filepath, "-f", "s16le", "-ar", str(sr), "-ac", "1", "-"]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sr
    except Exception as e:
        print(f"[audio] 加载失败: {e}")
        return np.zeros(sr * 2), sr

def compute_chroma(y, sr=22050):
    """用numpy计算chroma特征 (替代librosa.feature.chroma_cqt)
    用FFT频谱映射到12个音级"""
    try:
        hop = 512
        n_fft = 2048
        frames = range(0, len(y) - n_fft, hop)
        chroma = np.zeros((12, len(frames)))
        
        for i, start in enumerate(frames):
            frame = y[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            # 加窗FFT
            windowed = frame * np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            
            # 只取有意义的频率范围 (80Hz - 5000Hz)
            valid = (freqs > 80) & (freqs < 5000)
            for j, f in enumerate(freqs[valid]):
                if f > 0:
                    # 频率转MIDI音高
                    midi = 69 + 12 * np.log2(f / 440.0)
                    pc = int(midi) % 12
                    chroma[pc, i] += spectrum[valid][j]
        
        # 归一化
        col_sums = chroma.sum(axis=0, keepdims=True) + 1e-9
        chroma = chroma / col_sums
        return chroma
    except Exception as e:
        print(f"[chroma] 计算失败: {e}")
        return np.zeros((12, 100))

def compute_rms(y, hop=512):
    """计算RMS能量"""
    frames = range(0, len(y) - hop, hop)
    rms = np.array([np.sqrt(np.mean(y[i:i+hop]**2)) for i in frames])
    return rms

def compute_spectral_centroid(y, sr=22050, n_fft=2048, hop=512):
    """计算频谱质心"""
    try:
        frames = range(0, len(y) - n_fft, hop)
        centroids = []
        for start in frames:
            frame = y[start:start+n_fft] * np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
            if spectrum.sum() > 0:
                centroids.append(np.sum(freqs * spectrum) / (spectrum.sum() + 1e-9))
            else:
                centroids.append(0)
        return np.mean(centroids) if centroids else 0
    except:
        return 2000

def get_reference_chroma():
    global _ref_chroma
    if _ref_chroma is not None:
        return _ref_chroma
    try:
        if os.path.exists(REFERENCE_AUDIO):
            y, sr = load_audio(REFERENCE_AUDIO, 22050)
            _ref_chroma = compute_chroma(y, sr)
            print(f"[ref] chroma loaded, shape={_ref_chroma.shape}")
            return _ref_chroma
    except Exception as e:
        print(f"[ref] 加载失败: {e}")
    return None

def get_reference_features():
    global _ref_features
    if _ref_features is not None:
        return _ref_features
    try:
        if os.path.exists(REFERENCE_AUDIO):
            y, sr = load_audio(REFERENCE_AUDIO, 22050)
            rms = compute_rms(y)
            centroid = compute_spectral_centroid(y, sr)
            _ref_features = {
                "tempo": 123, "tempoStability": 0.96, "meter": "3/4",
                "register": "中音区" if 1500 < centroid < 3000 else "偏低音区",
                "volumeRange": float(np.std(rms) / (np.mean(rms) + 1e-6)),
                "volumeDesc": "强弱变化适度",
                "density": "适中", "texture": "简单伴奏",
            }
            return _ref_features
    except:
        pass
    return {"tempo": 123, "tempoStability": 0.96, "meter": "3/4", "register": "中音区",
            "volumeRange": 0.3, "volumeDesc": "适中", "density": "适中", "texture": "简单伴奏"}

def extract_basic_features(y, sr):
    """基础音频特征"""
    try:
        rms = compute_rms(y)
        avg_vol = float(np.mean(rms))
        centroid = compute_spectral_centroid(y, sr)
        onset_threshold = np.max(rms) * 0.3
        onset_count = np.sum(np.diff(rms) > onset_threshold)
        onset_rate = float(onset_count / (len(y) / sr)) if len(y) > 0 else 0
        
        # 节拍估计 (简化版)
        # 找RMS峰值作为节拍
        peaks = []
        threshold = np.max(rms) * 0.5
        for i in range(1, len(rms)-1):
            if rms[i] > threshold and rms[i] > rms[i-1] and rms[i] > rms[i+1]:
                peaks.append(i)
        if len(peaks) >= 3:
            intervals = np.diff(peaks) * 512 / sr
            tempo = 60.0 / np.mean(intervals) if np.mean(intervals) > 0 else 120
            cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
            stability = max(0, min(1, 1 - cv))
        else:
            tempo = 120; stability = 0.5
        
        # 节拍判断 (简化: 看RMS周期性)
        meter = "3/4" if len(peaks) > 0 and len(peaks) % 3 == 0 else "4/4"
        
        return {
            "tempo": round(tempo, 0), "tempoStability": round(stability, 2),
            "meter": meter,
            "register": "偏高音区" if centroid > 3000 else ("中音区" if centroid > 1500 else "偏低音区"),
            "avgVolume": round(avg_vol, 3),
            "volumeRange": round(float(np.std(rms) / (np.mean(rms) + 1e-6)), 2),
            "volumeDesc": "强弱变化大" if np.std(rms) / (np.mean(rms) + 1e-6) > 0.4 else ("有一定强弱变化" if np.std(rms) / (np.mean(rms) + 1e-6) > 0.2 else "音量较平"),
            "onsetRate": round(onset_rate, 1),
            "density": "密集" if onset_rate > 4 else ("适中" if onset_rate > 2 else "稀疏"),
            "texture": "简单伴奏",
            "bandwidth": round(centroid, 0),
        }
    except Exception as e:
        print(f"[features] 错误: {e}")
        return {"tempo": 120, "tempoStability": 0.5, "meter": "3/4", "register": "中音区",
                "volumeDesc": "适中", "volumeRange": 0.3, "density": "适中", "texture": "简单伴奏"}

def dtw_chroma(ref_chroma, perf_chroma):
    """简化DTW: 用numpy实现, 不依赖dtw库"""
    try:
        # 降采样到最多200帧
        step_ref = max(1, ref_chroma.shape[1] // 200)
        step_perf = max(1, perf_chroma.shape[1] // 200)
        ref = ref_chroma[:, ::step_ref]
        perf = perf_chroma[:, ::step_perf]
        
        # 计算代价矩阵 (余弦距离)
        ref_norm = ref / (np.linalg.norm(ref, axis=0) + 1e-9)
        perf_norm = perf / (np.linalg.norm(perf, axis=0) + 1e-9)
        
        # 简化DTW: 用累积距离
        n, m = ref.shape[1], perf.shape[1]
        D = np.full((n+1, m+1), np.inf)
        D[0, 0] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 1 - np.dot(ref_norm[:, i-1], perf_norm[:, j-1])
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        
        cost = D[n, m] / (n + m)
        similarity = max(0, min(1, 1 - cost * 2))
        return similarity, cost
    except Exception as e:
        print(f"[dtw] 错误: {e}")
        return 0.5, 0.25

def analyze_rhythm(y, sr):
    try:
        rms = compute_rms(y)
        peaks = []
        threshold = np.max(rms) * 0.5
        for i in range(1, len(rms)-1):
            if rms[i] > threshold and rms[i] > rms[i-1] and rms[i] > rms[i+1]:
                peaks.append(i)
        if len(peaks) < 3:
            return 60, "节拍点太少"
        intervals = np.diff(peaks) * 512 / sr
        cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
        tempo = 60.0 / np.mean(intervals)
        score = max(40, min(100, int(100 - cv * 133)))
        if cv < 0.1: comment = f"tempo约{tempo:.0f}BPM, 节奏非常稳定"
        elif cv < 0.2: comment = f"tempo约{tempo:.0f}BPM, 节奏基本稳定"
        else: comment = f"tempo约{tempo:.0f}BPM, 节奏不够稳定"
        return score, comment
    except:
        return 65, "节奏分析受限"

def analyze_pitch(y, sr):
    try:
        ref_chroma = get_reference_chroma()
        perf_chroma = compute_chroma(y, sr)
        if ref_chroma is None:
            return 65, "参考数据不可用"
        ref_prof = ref_chroma.mean(axis=1)
        perf_prof = perf_chroma.mean(axis=1)
        cos_sim = float(np.dot(ref_prof / (np.linalg.norm(ref_prof) + 1e-9),
                               perf_prof / (np.linalg.norm(perf_prof) + 1e-9)))
        dtw_sim, dtw_cost = dtw_chroma(ref_chroma, perf_chroma)
        pitch_score = cos_sim * 0.6 + dtw_sim * 0.4
        score = max(30, min(100, int(pitch_score * 100)))
        if cos_sim > 0.85: comment = f"音高分布高度匹配({cos_sim:.0%})"
        elif cos_sim > 0.70: comment = f"音高分布基本匹配({cos_sim:.0%})"
        else: comment = f"音高匹配度偏低({cos_sim:.0%})"
        return score, comment
    except:
        return 65, "音高分析受限"

def analyze_fluency(y, sr):
    try:
        rms = compute_rms(y)
        threshold = np.max(rms) * 0.12
        silent = rms < threshold
        silence_ratio = float(np.sum(silent) / max(len(silent), 1))
        score = max(30, min(100, int(100 - silence_ratio * 150)))
        if silence_ratio > 0.25: comment = f"有明显停顿(静音{silence_ratio:.0%})"
        elif silence_ratio > 0.10: comment = f"偶有停顿(静音{silence_ratio:.0%})"
        else: comment = "演奏连贯流畅"
        return score, comment
    except:
        return 65, "流畅性分析受限"

def analyze_dynamics(rms):
    try:
        dr = float(np.std(rms) / (np.mean(rms) + 1e-6))
        score = max(30, min(100, int(dr * 60 + 50)))
        if dr > 0.4: comment = f"力度变化丰富(动态{dr:.2f})"
        elif dr > 0.25: comment = f"有基本力度变化(动态{dr:.2f})"
        else: comment = f"力度较平(动态{dr:.2f})"
        return score, comment
    except:
        return 65, "力度分析受限"

def analyze_expression(y, sr, tempo):
    try:
        rms = compute_rms(y)
        mid = len(rms) // 2
        contrast = abs(np.mean(rms[:mid]) - np.mean(rms[mid:])) / (np.mean(rms) + 1e-6)
        tempo_score = max(30, min(100, 100 - abs(tempo - 120) * 2))
        score = int(tempo_score * 0.5 + min(contrast * 200, 100) * 0.5)
        score = max(40, min(100, score))
        comment = f"速度{'适中' if tempo_score > 70 else '偏快或偏慢'}, "
        comment += "强弱拍有区分" if contrast > 0.1 else "强弱拍对比不足"
        return score, comment
    except:
        return 65, "表现力分析受限"

def identify_song(y, sr):
    try:
        ref_chroma = get_reference_chroma()
        if ref_chroma is None: return True, 0.7, "参考数据不可用"
        perf_chroma = compute_chroma(y, sr)
        dtw_sim, dtw_cost = dtw_chroma(ref_chroma, perf_chroma)
        ref_prof = ref_chroma.mean(axis=1)
        perf_prof = perf_chroma.mean(axis=1)
        cos_sim = float(np.dot(ref_prof/(np.linalg.norm(ref_prof)+1e-9), perf_prof/(np.linalg.norm(perf_prof)+1e-9)))
        g_major = {0,2,4,6,7,9,11}
        g_ratio = float(sum(perf_prof[i] for i in g_major) / (sum(perf_prof)+1e-6))
        similarity = (cos_sim + g_ratio) / 2
        return True, similarity, f"匹配度{similarity:.0%}"
    except:
        return True, 0.7, "识别受限"

def extract_notes_set(y, sr):
    try:
        chroma = compute_chroma(y, sr)
        profile = chroma.mean(axis=1)
        result = {}
        for midi in range(48, 75):
            pc = midi % 12
            if profile[pc] > np.mean(profile) * 0.6:
                result[str(midi)] = True
        return result
    except:
        return {}

_ref_notes = None
def get_reference_notes_set():
    global _ref_notes
    if _ref_notes is not None: return _ref_notes
    try:
        if os.path.exists(REFERENCE_AUDIO):
            y, sr = load_audio(REFERENCE_AUDIO, 22050)
            _ref_notes = extract_notes_set(y, sr)
            return _ref_notes
    except: pass
    return {}

_ref_scores = None
def get_reference_scores():
    global _ref_scores
    if _ref_scores is not None: return _ref_scores
    try:
        if not os.path.exists(REFERENCE_AUDIO): return None
        y, sr = load_audio(REFERENCE_AUDIO, 22050)
        r_rhythm, _ = analyze_rhythm(y, sr)
        r_pitch, _ = analyze_pitch(y, sr)
        r_fluency, _ = analyze_fluency(y, sr)
        rms = compute_rms(y)
        r_dynamics, _ = analyze_dynamics(rms)
        r_expression, _ = analyze_expression(y, sr, 123)
        _ref_scores = {"rhythm": r_rhythm, "pitch": r_pitch, "fluency": r_fluency, "dynamics": r_dynamics, "expression": r_expression}
        print(f"[ref] 基准分数: {_ref_scores}")
        return _ref_scores
    except:
        return None

def normalize_to_ref(score, ref_score):
    if ref_score is None or ref_score <= 0: return score
    return max(30, min(100, int(score / ref_score * 100)))

def generate_student_feedback(total, issues):
    if total >= 85: prefix = "你弹得真棒！"
    elif total >= 70: prefix = "你弹得不错, 继续加油！"
    else: prefix = "别灰心, 多练几次一定会更好！"
    tips = []
    for i in issues:
        if i["level"] not in ("良好",) and i["type"] == "节奏": tips.append("跟着节拍器慢慢练, 数清楚拍子")
        if i["level"] not in ("良好",) and i["type"] == "力度平衡": tips.append("让右手像小歌唱家一样唱出来, 左手轻轻配合")
        if i["level"] not in ("良好",) and i["type"] == "流畅性": tips.append("把难弹的地方单独多练几遍")
    if not tips: tips.append("保持现在的水平, 可以挑战更有表现力的弹法")
    return prefix + " " + "；".join(tips) + "。"

def generate_teacher_feedback(total, issues, rhythm_compare, difficulty):
    parts = []
    parts.append(f"学生{'完整度较好' if total >= 85 else '基本掌握曲目'}(总分{total}), ")
    problems = [i for i in issues if i["level"] not in ("良好",)]
    if problems:
        parts.append("主要问题: " + "; ".join(f"{p['type']}({p['desc']})" for p in problems) + "。")
    else:
        parts.append("各维度表现均衡。")
    parts.append("建议: 使用Tomplay 60%速度片段循环练习。")
    return "".join(parts)

def diagnose_issues(user_feat, ref_feat, y, sr):
    issues = []
    if user_feat.get("tempoStability", 0) < 0.6:
        issues.append({"type": "节奏", "level": "明显", "desc": "节奏不够稳定, 可能有拖拍或抢拍"})
    else:
        issues.append({"type": "节奏", "level": "良好", "desc": "节奏稳定, 拍点清晰"})
    rms = compute_rms(y)
    silence_ratio = float(np.sum(rms < np.max(rms) * 0.12) / max(len(rms), 1))
    if silence_ratio > 0.25:
        issues.append({"type": "流畅性", "level": "明显", "desc": f"有明显停顿(静音{silence_ratio:.0%})"})
    elif silence_ratio > 0.10:
        issues.append({"type": "流畅性", "level": "轻微", "desc": "偶有停顿, 整体尚可"})
    else:
        issues.append({"type": "流畅性", "level": "良好", "desc": "演奏连贯流畅"})
    dr = user_feat.get("volumeRange", 0)
    if dr < 0.15:
        issues.append({"type": "力度平衡", "level": "明显", "desc": "力度变化少, 缺乏强弱对比"})
    elif dr < 0.3:
        issues.append({"type": "力度平衡", "level": "轻微", "desc": "有基本力度变化, 可更突出旋律"})
    else:
        issues.append({"type": "力度平衡", "level": "良好", "desc": "力度层次丰富"})
    return issues

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
            # 如果不是wav, 用ffmpeg转
            if not tmp_path.endswith(".wav"):
                wav_path = tmp_path + ".wav"
                subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-ar", "22050", "-ac", "1", wav_path],
                             capture_output=True, timeout=30)
                os.unlink(tmp_path)
                tmp_path = wav_path
            
            y, sr = load_audio(tmp_path, 22050)
            original_duration = len(y) / sr
            print(f"[analyze] 音频: {original_duration:.1f}秒")
            
            if original_duration > 60:
                y = y[:int(60 * sr)]
            
            if len(y) < sr * 2:
                return jsonify({"error": "音频太短, 至少需要2秒"}), 400

            # 音频质量检测
            rms_full = compute_rms(y)
            avg_vol = float(np.mean(rms_full))
            clipping = float(np.sum(np.abs(y) > 0.95) / len(y))
            yt_trim = y[np.abs(y) > np.max(np.abs(y)) * 0.05]
            effective_duration = len(yt_trim) / sr if len(yt_trim) > 0 else original_duration
            audio_quality = "good"
            quality_issues = []
            if avg_vol < 0.02: audio_quality = "poor"; quality_issues.append("音量过小")
            if clipping > 0.05: audio_quality = "poor"; quality_issues.append("有明显爆音/削波")
            if effective_duration < 5: audio_quality = "poor"; quality_issues.append(f"有效演奏仅{effective_duration:.1f}秒")
            
            if audio_quality == "poor":
                return jsonify({"isCorrectSong": None, "audioQuality": audio_quality,
                    "qualityIssues": quality_issues, "effectiveDuration": round(effective_duration, 1),
                    "message": "音频质量不足, 暂不评分",
                    "suggestion": "请靠近钢琴重新录制, 建议时长30-60秒, 环境安静。"})

            # 曲目闸门
            user_feat = extract_basic_features(y, sr)
            user_chroma = compute_chroma(y, sr)
            ref_feat = get_reference_features()
            is_correct, similarity, sim_comment = identify_song(y, sr)
            
            veto_reasons = []; veto_triggered = False
            pcs = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            g_major_pcs = {0,2,4,6,7,9,11}
            core_notes = {7,9,11,2}
            ref_chroma_gate = get_reference_chroma()
            perf_prof = user_chroma.mean(axis=1)
            ref_prof = ref_chroma_gate.mean(axis=1) if ref_chroma_gate is not None else np.zeros(12)
            
            # 闸门1: G大调覆盖率
            g_coverage = float(sum(perf_prof[i] for i in g_major_pcs) / (sum(perf_prof)+1e-6))
            if g_coverage < 0.60:
                veto_triggered = True
                top_notes = [pcs[i] for i in np.argsort(perf_prof)[-3:][::-1]]
                veto_reasons.append(f"调性不匹配: 音高覆盖G大调仅{g_coverage:.0%}(主要音为{','.join(top_notes)}), Anh.114是G大调小步舞曲")
            
            # 闸门2: 核心音组
            core_present = sum(1 for i in core_notes if perf_prof[i] > np.mean(perf_prof) * 0.8)
            if core_present < 2 and not veto_triggered:
                veto_triggered = True
                veto_reasons.append(f"核心音组不匹配: G/A/B/D中仅{core_present}个明显出现")
            
            # 闸门3: 无转调DTW
            dtw_sim_noshift = similarity
            if ref_chroma_gate is not None:
                ref_norm = ref_chroma_gate / (np.linalg.norm(ref_chroma_gate, axis=0) + 1e-9)
                perf_norm = user_chroma / (np.linalg.norm(user_chroma, axis=0) + 1e-9)
                dtw_sim_noshift, dtw_cost = dtw_chroma(ref_chroma_gate, user_chroma)
                if dtw_sim_noshift < 0.25:
                    veto_triggered = True
                    veto_reasons.append(f"旋律匹配度仅{dtw_sim_noshift:.0%}(无转调), 主旋律轮廓与Anh.114不一致")
            
            # 闸门4: 旋律轮廓指纹
            try:
                # 用RMS峰值代替onset检测
                rms_fp = compute_rms(y)
                peaks_fp = []
                thresh_fp = np.max(rms_fp) * 0.5
                for i in range(1, len(rms_fp)-1):
                    if rms_fp[i] > thresh_fp and rms_fp[i] > rms_fp[i-1] and rms_fp[i] > rms_fp[i+1]:
                        peaks_fp.append(i)
                if len(peaks_fp) >= 10:
                    user_fp = np.argmax(user_chroma[:, peaks_fp[:20]], axis=0)
                    user_int = np.diff(user_fp) % 12
                    user_int = np.where(user_int > 6, user_int - 12, user_int)
                    ref_y_fp, _ = load_audio(REFERENCE_AUDIO, 22050)
                    rms_ref = compute_rms(ref_y_fp)
                    peaks_ref = []
                    thresh_ref = np.max(rms_ref) * 0.5
                    for i in range(1, len(rms_ref)-1):
                        if rms_ref[i] > thresh_ref and rms_ref[i] > rms_ref[i-1] and rms_ref[i] > rms_ref[i+1]:
                            peaks_ref.append(i)
                    if len(peaks_ref) >= 10:
                        ref_chroma_fp = compute_chroma(ref_y_fp, 22050)
                        ref_fp = np.argmax(ref_chroma_fp[:, peaks_ref[:20]], axis=0)
                        ref_int = np.diff(ref_fp) % 12
                        ref_int = np.where(ref_int > 6, ref_int - 12, ref_int)
                        ml = min(len(user_int), len(ref_int))
                        dm = float(np.sum(np.sign(user_int[:ml]) == np.sign(ref_int[:ml])) / ml)
                        if dm < 0.35:
                            veto_triggered = True
                            veto_reasons.append(f"开头旋律轮廓匹配率仅{dm:.0%}, 旋律走向与Anh.114不一致")
            except: pass
            
            if veto_triggered:
                return jsonify({"isCorrectSong": False, "similarity": round(float(similarity), 2),
                    "totalScore": None, "message": "曲目不匹配, 暂不评分", "reasons": veto_reasons,
                    "suggestion": "请上传巴赫《G大调小步舞曲Anh.114》的演奏录音(3/4拍, G大调), 建议从曲目开头录制, 时长30-60秒。",
                    "userFeatures": {"tempo": user_feat.get("tempo"), "meter": user_feat.get("meter")},
                    "refFeatures": {"tempo": ref_feat.get("tempo"), "meter": ref_feat.get("meter")},
                    "audioQuality": audio_quality})

            # 可信度
            if similarity > 0.70: confidence = "high"
            else: confidence = "medium"

            # 五维评分
            rhythm_raw, rhythm_comment = analyze_rhythm(y, sr)
            pitch_raw, pitch_comment = analyze_pitch(y, sr)
            fluency_raw, fluency_comment = analyze_fluency(y, sr)
            rms = compute_rms(y)
            dynamics_raw, dynamics_comment = analyze_dynamics(rms)
            tempo = user_feat.get("tempo", 120)
            expression_raw, expression_comment = analyze_expression(y, sr, float(tempo))

            ref_scores = get_reference_scores()
            rhythm_score = normalize_to_ref(rhythm_raw, ref_scores["rhythm"]) if ref_scores else rhythm_raw
            pitch_score = normalize_to_ref(pitch_raw, ref_scores["pitch"]) if ref_scores else pitch_raw
            fluency_score = normalize_to_ref(fluency_raw, ref_scores["fluency"]) if ref_scores else fluency_raw
            dynamics_score = normalize_to_ref(dynamics_raw, ref_scores["dynamics"]) if ref_scores else dynamics_raw
            expression_score = normalize_to_ref(expression_raw, ref_scores["expression"]) if ref_scores else expression_raw

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

            deductions = []
            if rhythm_score < 95: deductions.append({"dim": "节奏稳定", "deduction": 95 - rhythm_score, "reason": rhythm_comment})
            if pitch_score < 95: deductions.append({"dim": "音高准确", "deduction": 95 - pitch_score, "reason": pitch_comment})
            if fluency_score < 95: deductions.append({"dim": "完整流畅", "deduction": 95 - fluency_score, "reason": fluency_comment})
            if dynamics_score < 95: deductions.append({"dim": "力度层次", "deduction": 95 - dynamics_score, "reason": dynamics_comment})
            if expression_score < 90: deductions.append({"dim": "音乐表现", "deduction": 90 - expression_score, "reason": expression_comment})

            issues = diagnose_issues(user_feat, ref_feat, y, sr)
            notes_played = extract_notes_set(y, sr)
            notes_expected = get_reference_notes_set()

            review_points = []
            if pitch_score < 85: review_points.append("个别音高是否为错音, 建议老师结合现场听辨")
            if fluency_score < 85: review_points.append("乐句连接处的停顿需要老师现场判断")
            if dynamics_score < 85: review_points.append("左手是否真正盖住右手, 需结合现场音响判断")
            if confidence == "medium": review_points.append("本次曲目匹配可信度为中等, 建议老师确认")
            if not review_points: review_points.append("各维度表现较好, 无需特别复核")

            student_feedback = generate_student_feedback(total_score, issues)
            teacher_feedback = generate_teacher_feedback(total_score, issues, {}, {"level": "中"})

            result = {
                "isCorrectSong": True, "matchConfidence": confidence,
                "similarity": round(float(similarity), 2), "totalScore": total_score,
                "audioQuality": audio_quality,
                "effectiveDuration": round(effective_duration, 1),
                "leadSilence": 0, "pauses": 0,
                "tempo": round(float(tempo), 0),
                "duration": round(len(y) / sr, 1),
                "originalDuration": round(original_duration, 1),
                "truncated": original_duration > 60,
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
                "report": {"basicFeatures": user_feat, "issues": issues},
                "notesPlayed": notes_played, "notesExpected": notes_expected,
            }
            print(f"[analyze] 完成: 总分={total_score}, 可信度={confidence}")
            return jsonify(result)
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)
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
            result = subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                                   capture_output=True, timeout=30)
            if result.returncode != 0: return jsonify({"error": "音频转换失败"}), 500
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source: audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language="zh-CN")
                return jsonify({"text": text, "engine": "google"})
            except: return jsonify({"text": "", "error": "未能识别语音内容"})
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
    print("[启动] 轻量版音频分析微服务: http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
