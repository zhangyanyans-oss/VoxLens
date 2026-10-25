#!/usr/bin/env python3
"""火山引擎流式识别（带说话人分离） + DeepSeek 翻译 + 百度多音色 TTS。"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import logging
import os
import queue
import struct
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import requests
import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 3200  # 200ms；官方建议双向流式每包 200ms
CHUNK_SECONDS = CHUNK_SAMPLES / SAMPLE_RATE
LOGGER = logging.getLogger("volc-translator")

# 双向流式优化版：尽快返回 partial，并用 nostream 二遍结果给出 definite。
# 说话人分离需要同时开启 enable_nonstream 和 ssd_version=200。
VOLC_WS_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
VOLC_RESOURCE_ID = "volc.seedasr.sauc.duration"  # 2.0 小时版

# 默认 fast 与原版本完全相同；长句模式仅由用户手动选择。
SPEAKER_PROFILES = {
    "fast": {"end_window_size": 800, "force_to_speech_time": 1000},
    "long": {"end_window_size": 1400, "force_to_speech_time": 1500},
}


def load_env_file(path: Optional[Path] = None) -> None:
    env_path = path or Path(__file__).resolve().with_name(".env")
    if not env_path.is_file():
        return
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("export "): line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator: continue
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _header(msg_type: int, flags: int = 0, ser: int = 1, comp: int = 1) -> bytes:
    return bytes([(1 << 4) | 1, (msg_type << 4) | flags, (ser << 4) | comp, 0])

def _full_client_request(payload: dict) -> bytes:
    data = gzip.compress(json.dumps(payload).encode())
    return _header(0b0001, 0, 1, 1) + struct.pack(">I", len(data)) + data

def _audio_request(pcm: bytes, is_last: bool = False) -> bytes:
    data = gzip.compress(pcm) if pcm else b""
    flags = 0b0010 if is_last else 0b0000
    return _header(0b0010, flags, 0, 1) + struct.pack(">I", len(data)) + data

def _parse_response(raw: bytes) -> Optional[dict]:
    """解析服务端二进制包，并保留负序号用于判断最终结果已经到达。"""
    if len(raw) < 4:
        return None
    msg_type = (raw[1] >> 4) & 0x0F
    flags = raw[1] & 0x0F
    comp = raw[2] & 0x0F
    offset = 4
    sequence = None
    if flags & 0x01:
        if len(raw) < 8:
            raise ValueError("响应缺少 sequence")
        sequence = struct.unpack(">i", raw[4:8])[0]
        offset = 8
    if len(raw) < offset + 4:
        raise ValueError("响应缺少 payload size")
    size = struct.unpack(">I", raw[offset : offset + 4])[0]
    payload = raw[offset + 4 : offset + 4 + size]
    if len(payload) != size:
        raise ValueError("响应 payload 不完整")
    if msg_type == 0b1001:
        if comp == 0b0001:
            payload = gzip.decompress(payload)
        data = json.loads(payload.decode())
        data["_sequence"] = sequence
        return data
    if msg_type == 0b1111:
        err_msg = payload.decode(errors="ignore")
        raise RuntimeError(f"火山错误响应: {err_msg}")
    return None


class SegmentState(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"

@dataclass(frozen=True)
class TranscriptSegment:
    id: str
    text: str
    speaker_id: str
    state: SegmentState
    revision: int


class VolcStreamingASR:
    def __init__(
        self, app_id: str, access_key: str, on_segment: Callable,
        audio_queue: "LatestQueue", speaker_mode: str = "fast",
    ):
        self.app_id = app_id
        self.access_key = access_key
        self.on_segment = on_segment
        self.audio_queue = audio_queue
        self.speaker_mode = speaker_mode
        self._revisions: dict[str, int] = {}
        self._texts: dict[str, str] = {}
        self._final_ids: set[str] = set()
        self._session = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        # capture.stop() 已先放入结束标记；这里等待尾包和最后一句返回。
        self._stop.set()
        self._thread.join(timeout=15)
        if self._thread.is_alive():
            LOGGER.warning("火山连接停止超时；最后一句可能尚未返回")

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while not self._stop.is_set():
                try:
                    loop.run_until_complete(self._main())
                except Exception as exc:
                    if not self._stop.is_set():
                        LOGGER.warning("与火山的连接断开，1 秒后重连：%s", exc)
                        self._stop.wait(1)
        finally:
            loop.close()

    async def _main(self) -> None:
        self._session += 1
        self._revisions.clear()
        self._texts.clear()
        self._final_ids.clear()
        headers = {
            "X-Api-App-Key": self.app_id,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": VOLC_RESOURCE_ID,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        async with websockets.connect(
            VOLC_WS_URL, additional_headers=headers, max_size=16 * 1024 * 1024,
            ping_interval=20, ping_timeout=20,
        ) as ws:
            req = {
                "user": {"uid": "realtime-translator"},
                "audio": {"format": "pcm", "rate": SAMPLE_RATE, "bits": 16, "channel": 1},
                "request": {
                    "model_name": "bigmodel",
                    "enable_itn": True,
                    "enable_punc": True,
                    "enable_speaker_info": True,
                    "show_utterances": True,
                    "enable_nonstream": True,
                    "ssd_version": "200",
                    "result_type": "full",
                    **SPEAKER_PROFILES[self.speaker_mode],
                },
            }
            await ws.send(_full_client_request(req))
            sender_done = asyncio.Event()

            async def send_audio():
                next_send = time.monotonic()
                try:
                    while True:
                        try:
                            pcm = await asyncio.to_thread(self.audio_queue.get, 0.2)
                        except queue.Empty:
                            continue
                        if pcm is None:
                            await ws.send(_audio_request(b"", is_last=True))
                            return
                        # 按音频时长发送，避免把 100ms 音频以 10ms 间隔突发推给服务端。
                        delay = next_send - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        await ws.send(_audio_request(pcm))
                        next_send = max(next_send + CHUNK_SECONDS, time.monotonic())
                finally:
                    sender_done.set()

            async def recv_results():
                final_wait_started = None
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if sender_done.is_set():
                            final_wait_started = final_wait_started or time.monotonic()
                            if time.monotonic() - final_wait_started > 12:
                                raise TimeoutError("等待火山最终结果超时")
                        continue
                    data = _parse_response(raw)
                    if data:
                        self._handle_response(data)
                        sequence = data.get("_sequence")
                        if sequence is not None and sequence < 0:
                            return

            sender = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(recv_results())
            try:
                await sender
                await receiver
            finally:
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)

    def _handle_response(self, data: dict) -> None:
        result = data.get("result") or {}
        if isinstance(result, list):
            result = result[0] if result else {}
        utterances = result.get("utterances") or []
        for index, utterance in enumerate(utterances):
            text = (utterance.get("text") or "").strip()
            if not text:
                continue
            additions = utterance.get("additions") or {}
            spk = str(
                utterance.get("speaker_id")
                or utterance.get("speaker")
                or additions.get("speaker_id")
                or additions.get("speaker")
                or "0"
            )
            is_final = bool(utterance.get("definite", utterance.get("is_final", False)))
            start = utterance.get("start_time")
            # 时间轴标识比“说话人+文本前缀”可靠：相同内容的两句话不会被误删。
            seg_key = str(start) if start is not None else f"index-{index}"
            seg_id = f"{self._session}:{seg_key}"
            if seg_id in self._final_ids:
                continue
            previous = self._revisions.get(seg_id, 0)
            state = SegmentState.FINAL if is_final else SegmentState.PARTIAL
            if not is_final and previous and self._texts.get(seg_id) == text:
                continue
            revision = previous + 1
            self._revisions[seg_id] = revision
            self._texts[seg_id] = text
            if is_final:
                self._final_ids.add(seg_id)
            self.on_segment(TranscriptSegment(
                id=seg_id, text=text, speaker_id=spk, state=state, revision=revision,
            ))


class AudioCapture:
    def __init__(self, audio_queue: "LatestQueue"):
        self.audio_queue = audio_queue
        self._stop = threading.Event()
        self._stream = None
        self._audio = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        import pyaudio
        self._audio = pyaudio.PyAudio()
        idx = None
        for i in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(i)
            if "CABLE Output" in str(info.get("name", "")):
                idx = i
                LOGGER.info("使用虚拟声卡: %s", info.get("name"))
                break
        self._stream = self._audio.open(
            format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True,
            frames_per_buffer=CHUNK_SAMPLES, input_device_index=idx,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                self.audio_queue.put_latest(data)
            except Exception as exc:
                if not self._stop.is_set(): LOGGER.error("采集异常: %s", exc)
                break
        self.audio_queue.close()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive(): self._thread.join(3)
        if self._stream: self._stream.stop_stream(); self._stream.close()
        if self._audio: self._audio.terminate()


class LatestQueue:
    """只用于实时音频；溢出时丢最旧音频，但必须显式计数和报警。"""

    def __init__(self, maxsize: int, name: str):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.name = name
        self._lock = threading.Lock()
        self.dropped = 0

    def put_latest(self, item) -> None:
        with self._lock:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                pass
            try:
                self._q.get_nowait()
                self.dropped += 1
                if self.dropped == 1 or self.dropped % 25 == 0:
                    LOGGER.warning(
                        "%s 队列积压，已丢 %d 个旧音频块；请检查网络或识别连接",
                        self.name, self.dropped,
                    )
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    def close(self) -> None:
        """结束标记不能像普通音频一样被静默丢弃。"""
        try:
            self._q.put(None, timeout=5)
        except queue.Full:
            LOGGER.error("音频结束标记写入失败，无法冲刷最后一句")

    def get(self, timeout: float = 0.2):
        return self._q.get(timeout=timeout)

    def qsize(self) -> int:
        return self._q.qsize()


class DeepSeekTranslator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.deepseek.com/chat/completions"
        self._local = threading.local()
        self.system_prompt = (
            "你是科技播客同声传译。把用户输入的英文翻译成自然中文。"
            "deep dive='深入探讨'，crystal clear='非常清楚'，generic 在工厂语境下='通用版'。"
            "只输出译文，不要解释。"
        )

    def translate(self, text: str) -> Optional[str]:
        if not text.strip(): return None
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}],
            "temperature": 0.2, "max_tokens": 300,
        }
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self._local.session = session
        for attempt in range(2):
            try:
                r = session.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=(5, 20),
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                if attempt == 1:
                    LOGGER.warning("DeepSeek 失败：%s；原文：%s", exc, text)
                else:
                    time.sleep(0.5)
        return None


class BaiduTTS:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self._token: Optional[str] = None
        self._exp = 0.0
        self._lock = threading.Lock()

    def _get_token(self) -> Optional[str]:
        with self._lock:
            if self._token and time.time() < self._exp: return self._token
            try:
                r = requests.post(
                    "https://aip.baidubce.com/oauth/2.0/token", 
                    params={"grant_type": "client_credentials", "client_id": self.api_key, "client_secret": self.secret_key},
                    timeout=8, proxies={"http": None, "https": None}
                ).json()
                if "access_token" in r:
                    self._token = r["access_token"]
                    self._exp = time.time() + int(r["expires_in"]) - 300
                    return self._token
            except Exception: pass
            return None

    def synthesize(self, text: str, speaker_id: str) -> Optional[bytes]:
        token = self._get_token()
        if not token or not text.strip(): return None
        per_map = {"0": "3", "1": "4", "2": "106", "3": "110"}
        per = per_map.get(speaker_id, "3")
        try:
            r = requests.post(
                "https://tsn.baidu.com/text2audio", 
                data={
                    "tex": text, "tok": token, "cuid": "volc_translator", "ctp": 1,
                    "lan": "zh", "spd": 7, "pit": 5, "vol": 7, "per": per, "aue": 6,
                }, 
                timeout=12, proxies={"http": None, "https": None}
            )
            if "audio" in r.headers.get("Content-Type", "").lower() or r.content.startswith(b"RIFF"):
                return r.content
        except Exception: pass
        return None


class TTSProcessor:
    """并行合成、按提交顺序播放；任何队列都不再主动扔句子。"""

    def __init__(self, api_key: str, secret_key: str, workers: int = 2):
        self.client = BaiduTTS(api_key, secret_key)
        self.in_q: queue.Queue = queue.Queue()
        self.workers = max(1, workers)
        self._stop = threading.Event()
        self._pa = None
        self._threads: list[threading.Thread] = []
        self._cv = threading.Condition()
        self._next_submit = 0
        self._next_play = 0
        self._results: dict[int, Optional[bytes]] = {}

    def start(self) -> None:
        import pyaudio
        self._pa = pyaudio.PyAudio()
        for index in range(self.workers):
            thread = threading.Thread(target=self._synth_loop, args=(index + 1,), daemon=True)
            thread.start()
            self._threads.append(thread)
        thread = threading.Thread(target=self._play_loop, daemon=True)
        thread.start()
        self._threads.append(thread)

    def submit(self, text: str, speaker_id: str) -> None:
        if not text.strip():
            return
        with self._cv:
            sequence = self._next_submit
            self._next_submit += 1
        self.in_q.put((sequence, text.strip(), speaker_id))
        backlog = self.in_q.qsize()
        if backlog in (10, 30, 60):
            LOGGER.warning("TTS 待合成 %d 句；不会跳句，但播报会逐渐滞后", backlog)

    def _synth_loop(self, worker_id: int) -> None:
        while not self._stop.is_set():
            try:
                sequence, text, speaker_id = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            started = time.monotonic()
            audio = self.client.synthesize(text, speaker_id)
            elapsed = time.monotonic() - started
            if audio is None:
                LOGGER.warning("第 %d 句 TTS 合成失败，保留该句错误并继续后续播报", sequence + 1)
            elif elapsed > 3:
                LOGGER.info("TTS 合成 %.1fs（工作线程 %d）", elapsed, worker_id)
            with self._cv:
                self._results[sequence] = audio
                self._cv.notify_all()
            self.in_q.task_done()

    def _play_loop(self) -> None:
        import pyaudio
        while not self._stop.is_set():
            with self._cv:
                self._cv.wait_for(
                    lambda: self._next_play in self._results or self._stop.is_set(), timeout=0.5,
                )
                if self._next_play not in self._results:
                    continue
                audio = self._results.pop(self._next_play)
                sequence = self._next_play
                self._next_play += 1
            if audio is None:
                continue
            stream = None
            try:
                with wave.open(io.BytesIO(audio), "rb") as wav:
                    stream = self._pa.open(
                        format=pyaudio.paInt16, channels=wav.getnchannels(),
                        rate=wav.getframerate(), output=True,
                    )
                    data = wav.readframes(2048)
                    while data and not self._stop.is_set():
                        stream.write(data)
                        data = wav.readframes(2048)
            except Exception as exc:
                if not self._stop.is_set():
                    LOGGER.warning("第 %d 句播放失败：%s", sequence + 1, exc)
            finally:
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._pa:
            self._pa.terminate()


class App:
    def __init__(self, args, on_event: Optional[Callable[[str, dict], None]] = None):
        self.args = args
        self.on_event = on_event
        self.audio_q = LatestQueue(25, "audio")  # 最多缓存 5 秒，而不是原来的 50 秒
        self.trans_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._partial_active = False
        self._trans_cv = threading.Condition()
        self._next_translate = 0
        self._next_publish = 0
        self._translated: dict[int, tuple[TranscriptSegment, Optional[str]]] = {}
        self.translator = None if args.no_translate else DeepSeekTranslator(os.environ["DEEPSEEK_API_KEY"])
        self.tts = None
        if self.translator and not args.no_tts:
            self.tts = TTSProcessor(
                os.environ["BAIDU_TTS_API_KEY"], os.environ["BAIDU_TTS_SECRET_KEY"],
                workers=args.tts_workers,
            )
        self.asr = VolcStreamingASR(
            os.environ["VOLC_APP_ID"], os.environ["VOLC_ACCESS_KEY"], self._on_segment, self.audio_q,
            speaker_mode=args.speaker_mode,
        )
        self.capture = AudioCapture(self.audio_q)

    def _emit(self, kind: str, **payload) -> None:
        if self.on_event:
            try:
                self.on_event(kind, payload)
            except Exception as exc:
                LOGGER.debug("界面事件回调失败：%s", exc)

    def _on_segment(self, seg: TranscriptSegment) -> None:
        self._emit(
            "partial" if seg.state == SegmentState.PARTIAL else "final",
            id=seg.id, text=seg.text, speaker_id=seg.speaker_id,
        )
        if seg.state == SegmentState.PARTIAL:
            if not self.args.no_partial:
                print(f"\r[识别中][说话人{seg.speaker_id}] {seg.text}\033[K", end="", flush=True)
                self._partial_active = True
            return
        if self._partial_active:
            print("\r\033[K", end="")
            self._partial_active = False
        print(f"[原文][说话人{seg.speaker_id}] {seg.text}", flush=True)
        if not self.translator:
            return
        with self._trans_cv:
            sequence = self._next_translate
            self._next_translate += 1
        self.trans_q.put((sequence, seg))
        backlog = self.trans_q.qsize()
        if backlog in (5, 15, 30):
            LOGGER.warning("待翻译 %d 句；不会跳句，可适当增加 --translate-workers", backlog)

    def _translate_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sequence, seg = self.trans_q.get(timeout=0.5)
            except queue.Empty:
                continue
            started = time.monotonic()
            translated = self.translator.translate(seg.text)
            elapsed = time.monotonic() - started
            if elapsed > 3:
                LOGGER.info("DeepSeek 翻译 %.1fs：%s", elapsed, seg.text[:30])
            with self._trans_cv:
                self._translated[sequence] = (seg, translated)
                self._trans_cv.notify_all()
            self.trans_q.task_done()

    def _publish_loop(self) -> None:
        """慢句允许后句先算完，但译文和 TTS 仍严格按原文顺序发布。"""
        while not self._stop.is_set():
            with self._trans_cv:
                self._trans_cv.wait_for(
                    lambda: self._next_publish in self._translated or self._stop.is_set(), timeout=0.5,
                )
                if self._next_publish not in self._translated:
                    continue
                seg, translated = self._translated.pop(self._next_publish)
                self._next_publish += 1
            if not translated:
                self._emit(
                    "translation_failed", id=seg.id, text=seg.text, speaker_id=seg.speaker_id,
                )
                print(f"[翻译失败][说话人{seg.speaker_id}] {seg.text}", flush=True)
                continue
            self._emit(
                "translation", id=seg.id, text=translated,
                original=seg.text, speaker_id=seg.speaker_id,
            )
            print(f"[译文][说话人{seg.speaker_id}] {translated}", flush=True)
            if self.tts:
                self.tts.submit(translated, seg.speaker_id)

    def start(self) -> None:
        if self.tts:
            self.tts.start()
        if self.translator:
            for _ in range(self.args.translate_workers):
                thread = threading.Thread(target=self._translate_loop, daemon=True)
                thread.start()
                self._threads.append(thread)
            thread = threading.Thread(target=self._publish_loop, daemon=True)
            thread.start()
            self._threads.append(thread)
        self.asr.start()
        self.capture.start()
        print("已启动：火山双向流式二遍识别（说话人分离）+ DeepSeek + 百度多音色")
        LOGGER.info("说话人分句模式：%s（fast 为原速）", self.args.speaker_mode)
        if self.audio_q.dropped:
            LOGGER.warning("启动前已有音频块被丢弃")

    def stop(self) -> None:
        self.capture.stop()
        self.asr.stop()
        self._stop.set()
        with self._trans_cv:
            self._trans_cv.notify_all()
        if self.tts:
            self.tts.stop()


class TranslatorPanel:
    """轻量桌面悬浮面板；识别在后台运行，隐藏窗口不会停止任务。"""

    def __init__(self, args):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.args = args
        self.app: Optional[App] = None
        self.running = False
        self.busy = False
        self.closing = False
        self.event_q: queue.Queue = queue.Queue()
        self._speaker_aliases: dict[str, str] = {}
        self._speaker_tags: set[str] = set()
        self._translation_marks: dict[str, str] = {}
        self._mark_counter = 0
        self.root = tk.Tk()
        self.root.title("实时转写翻译")
        self.root.geometry("720x620+80+60")
        self.root.minsize(520, 400)
        self.root.configure(bg="#f1f5f9")
        self.root.attributes("-topmost", True)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f1f5f9")
        style.configure("TLabel", background="#f1f5f9", foreground="#0f172a")
        style.configure("TButton", padding=(12, 8), font=("Microsoft YaHei UI", 10))
        style.configure("Accent.TButton", foreground="#ffffff", background="#2563eb")
        style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")],
            foreground=[("disabled", "#f8fafc")],
        )

        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)
        header = ttk.Frame(container)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(
            header, text="实时转写翻译", font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(side="left")
        self.status_label = tk.Label(
            header, text="● 尚未开始", bg="#f1f5f9", fg="#64748b",
            font=("Microsoft YaHei UI", 10),
        )
        self.status_label.pack(side="right")

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 12))
        self.toggle_button = ttk.Button(
            controls, text="开始转写翻译", style="Accent.TButton", command=self.toggle,
        )
        self.toggle_button.pack(side="left")
        ttk.Button(controls, text="隐藏悬浮窗", command=self.hide_panel).pack(side="left", padx=8)
        ttk.Button(controls, text="清空对话", command=self.clear).pack(side="left")
        self.speaker_mode = tk.StringVar(value="速度优先")
        self.speaker_box = ttk.Combobox(
            controls, textvariable=self.speaker_mode, state="readonly", width=10,
            values=("速度优先", "长句优先"),
        )
        self.speaker_box.pack(side="right", padx=(0, 10))
        self.topmost = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls, text="窗口置顶", variable=self.topmost,
            command=lambda: self.root.attributes("-topmost", self.topmost.get()),
        ).pack(side="right")

        conversation_frame = ttk.Frame(container)
        conversation_frame.pack(fill="both", expand=True)
        self.conversation = tk.Text(
            conversation_frame, wrap="word", state="disabled", cursor="arrow", height=10,
            bg="#ffffff", fg="#0f172a", relief="flat", bd=0,
            padx=18, pady=14, spacing1=4, spacing3=8,
            font=("Microsoft YaHei UI", 11), selectbackground="#bfdbfe",
        )
        scrollbar = ttk.Scrollbar(conversation_frame, command=self.conversation.yview)
        self.conversation.configure(yscrollcommand=scrollbar.set)
        self.conversation.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.conversation.tag_configure("meta", foreground="#64748b", font=("Microsoft YaHei UI", 9))
        self.conversation.tag_configure(
            "english", foreground="#1e3a8a", font=("Microsoft YaHei UI", 12, "bold"),
            lmargin1=8, lmargin2=8,
        )
        self.conversation.tag_configure(
            "chinese", foreground="#047857", font=("Microsoft YaHei UI", 12),
            lmargin1=8, lmargin2=8, spacing3=14,
        )
        self.conversation.tag_configure(
            "failed", foreground="#b91c1c", font=("Microsoft YaHei UI", 10),
            lmargin1=8, lmargin2=8, spacing3=14,
        )

        self.partial_label = tk.Label(
            container, text="点击“开始转写翻译”后，这里会显示正在识别的英文。",
            anchor="w", justify="left", bg="#e2e8f0", fg="#475569",
            padx=12, pady=10, font=("Microsoft YaHei UI", 10),
        )
        self.partial_label.pack(fill="x", pady=(12, 6))
        ttk.Label(
            container, text="隐藏后任务仍在后台运行，可点击任务栏图标恢复；Esc 也可隐藏。",
            foreground="#64748b", font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w")
        self.root.bind("<Escape>", lambda _event: self.hide_panel())
        self.root.bind("<Configure>", self._resize_wrap)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _resize_wrap(self, event) -> None:
        if event.widget == self.root:
            self.partial_label.configure(wraplength=max(300, event.width - 60))

    def post_event(self, kind: str, payload: dict) -> None:
        # 所有后台事件只进入队列，由 Tk 主线程统一更新控件。
        self.event_q.put((kind, payload))

    def toggle(self) -> None:
        if self.busy:
            return
        if self.running:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        self.args.speaker_mode = {"速度优先": "fast", "长句优先": "long"}[self.speaker_mode.get()]
        self.busy = True
        self.speaker_box.state(["disabled"])
        self.toggle_button.configure(text="正在启动…", state="disabled")
        self._set_status("正在连接音频和云服务", "#b45309")

        def worker() -> None:
            try:
                app = App(self.args, on_event=self.post_event)
                self.app = app
                app.start()
            except Exception as exc:
                LOGGER.exception("启动失败")
                self.post_event("start_failed", {"message": str(exc)})
            else:
                self.post_event("started", {})

        threading.Thread(target=worker, daemon=True).start()

    def stop(self) -> None:
        self.busy = True
        self.toggle_button.configure(text="正在停止…", state="disabled")
        self._set_status("正在停止并等待最后一句", "#b45309")

        def worker() -> None:
            try:
                if self.app:
                    self.app.stop()
            except Exception as exc:
                LOGGER.warning("停止异常：%s", exc)
            finally:
                self.app = None
                self.post_event("stopped", {})

        threading.Thread(target=worker, daemon=True).start()

    def hide_panel(self) -> None:
        # 最小化而不是销毁窗口：后台识别、翻译和播放继续进行。
        self.root.iconify()

    def clear(self) -> None:
        self.conversation.configure(state="normal")
        self.conversation.delete("1.0", "end")
        self.conversation.configure(state="disabled")
        self._translation_marks.clear()

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.configure(text=f"● {text}", fg=color)

    def _append(self, text: str, tag: str) -> None:
        self.conversation.configure(state="normal")
        self.conversation.insert("end", text, tag)
        self.conversation.see("end")
        self.conversation.configure(state="disabled")

    def _speaker_display(self, raw_id) -> tuple[str, str, str]:
        """火山从 0 编号，界面从 1 编号；同一原始编号始终使用同一颜色。"""
        raw = str(raw_id).strip() or "0"
        if raw not in self._speaker_aliases:
            self._speaker_aliases[raw] = str(int(raw) + 1) if raw.isdigit() else str(len(self._speaker_aliases) + 1)
        label = self._speaker_aliases[raw]
        colors = ("#2563eb", "#7c3aed", "#c2410c", "#0f766e", "#be123c", "#4d7c0f")
        color = colors[(int(label) - 1) % len(colors)]
        tag = f"speaker_{label}"
        if tag not in self._speaker_tags:
            self.conversation.tag_configure(tag, foreground=color, font=("Microsoft YaHei UI", 9, "bold"))
            self._speaker_tags.add(tag)
        return f"说话人 {label}", tag, color

    def _insert_translation(self, data: dict, text: str, tag: str) -> None:
        """插回对应英文的末尾，不受其他说话人后续消息影响。"""
        mark = self._translation_marks.pop(str(data.get("id")), None)
        if not mark:
            return  # 清空对话后才到达的旧译文不再重新出现。
        self.conversation.configure(state="normal")
        self.conversation.insert(mark, f"{text}\n", tag)
        self.conversation.mark_unset(mark)
        self.conversation.see("end")
        self.conversation.configure(state="disabled")

    def _handle_event(self, kind: str, data: dict) -> None:
        if kind == "partial":
            speaker, _tag, color = self._speaker_display(data["speaker_id"])
            self.partial_label.configure(text=f"{speaker}正在识别：{data['text']}", fg=color)
            return
        if kind == "final":
            self.partial_label.configure(text="等待下一句…", fg="#475569")
            segment_id = str(data["id"])
            if segment_id in self._translation_marks:
                return
            speaker, speaker_tag, _color = self._speaker_display(data["speaker_id"])
            stamp = time.strftime("%H:%M:%S")
            self._append(f"{speaker}  ·  {stamp}\n", speaker_tag)
            self._append(f"{data['text']}\n", "english")
            # 左重力标记停在当前英文之后；后续其他说话人的消息不会把它挤走。
            self._mark_counter += 1
            mark = f"translation_{self._mark_counter}"
            self.conversation.mark_set(mark, "end-1c")
            self.conversation.mark_gravity(mark, "left")
            self._translation_marks[segment_id] = mark
            return
        if kind == "translation":
            self._insert_translation(data, data["text"], "chinese")
            return
        if kind == "translation_failed":
            self._insert_translation(data, "翻译失败，请查看终端日志。", "failed")
            return
        if kind == "started":
            self.running = True
            self.busy = False
            self.toggle_button.configure(text="停止转写翻译", state="normal")
            self._set_status("正在转写翻译", "#059669")
            return
        if kind == "start_failed":
            self.running = False
            self.busy = False
            self.app = None
            self.speaker_box.state(["!disabled"])
            self.toggle_button.configure(text="开始转写翻译", state="normal")
            self._set_status("启动失败", "#dc2626")
            self.partial_label.configure(text=f"启动失败：{data['message']}", fg="#b91c1c")
            return
        if kind == "stopped":
            self.running = False
            self.busy = False
            self.speaker_box.state(["!disabled"])
            self.toggle_button.configure(text="开始转写翻译", state="normal")
            self._set_status("已停止", "#64748b")
            self.partial_label.configure(text="已停止。再次点击按钮可重新开始。", fg="#475569")
            if self.closing:
                self.root.destroy()

    def _drain_events(self) -> None:
        try:
            while True:
                kind, data = self.event_q.get_nowait()
                self._handle_event(kind, data)
        except queue.Empty:
            pass
        self.root.after(60, self._drain_events)

    def close(self) -> None:
        if self.closing:
            return
        if self.app:
            self.closing = True
            self.busy = True
            self.toggle_button.configure(state="disabled")
            self._set_status("正在退出", "#b45309")

            def worker() -> None:
                try:
                    if self.app:
                        self.app.stop()
                finally:
                    self.app = None
                    self.post_event("stopped", {})

            threading.Thread(target=worker, daemon=True).start()
            # closing 时继续排空事件，收到 stopped 后再销毁窗口。
            self.root.after(60, self._drain_events)
        else:
            self.root.destroy()

    def run(self) -> None:
        self.root.after(60, self._drain_events)
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="火山双向流式识别 + DeepSeek 翻译 + 百度 TTS")
    parser.add_argument("--cli", action="store_true", help="不打开悬浮面板，使用原命令行模式")
    parser.add_argument("--no-tts", action="store_true", help="不合成语音")
    parser.add_argument("--no-translate", action="store_true", help="只测试识别，不调用 DeepSeek 和 TTS")
    parser.add_argument("--no-partial", action="store_true", help="不显示识别中的暂存文字")
    parser.add_argument(
        "--speaker-mode", choices=("fast", "long"), default="fast",
        help="fast 为原速默认；long 用较长分句尝试改善说话人分离",
    )
    parser.add_argument("--translate-workers", type=int, default=2, help="并行翻译线程数，默认 2")
    parser.add_argument("--tts-workers", type=int, default=2, help="并行合成线程数，默认 2")
    args = parser.parse_args()
    if args.translate_workers < 1 or args.tts_workers < 1:
        parser.error("工作线程数必须大于等于 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""

    load_env_file()  

    need = ["VOLC_APP_ID", "VOLC_ACCESS_KEY"]
    if not args.no_translate:
        need.append("DEEPSEEK_API_KEY")
        if not args.no_tts:
            need += ["BAIDU_TTS_API_KEY", "BAIDU_TTS_SECRET_KEY"]
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        print(f"缺少环境变量: {missing}")
        return 1

    if not args.cli:
        try:
            TranslatorPanel(args).run()
        except Exception as exc:
            print(f"悬浮面板启动失败：{exc}\n可改用命令行模式：python {Path(__file__).name} --cli")
            return 1
        return 0

    app = App(args)
    app.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        app.stop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())