"""听音频文件的核心：取字节 → ffmpeg 压成小 mp3 → 发给能听音频的模型（OpenAI 兼容接口，流式）。

这一层不依赖具体框架，可以单独拿来测。
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import os
import shutil
import socket
import tempfile
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp import ThreadedResolver
from aiohttp.abc import AbstractResolver

AUDIO_EXTS = (
    ".m4a", ".mp3", ".wav", ".amr", ".aac", ".ogg", ".opus", ".flac",
    ".wma", ".silk", ".slk", ".caf", ".aiff", ".aif", ".3gp",
)
# 没装 ffmpeg 时，模型本身就认的格式可以原样发过去
_DIRECT_FORMATS = {".mp3": "mp3", ".wav": "wav", ".aac": "aac", ".amr": "amr", ".3gp": "3gp"}
# 百炼要求 base64 后小于 10MB；原始字节留点余量
_MAX_INLINE_BYTES = 7 * 1024 * 1024
_MAX_REDIRECTS = 5
# 让 ffmpeg 只读本地文件：防止伪装成音频的播放列表让它去拉网址
_FFMPEG_SAFE_INPUT = ("-protocol_whitelist", "file")

DEFAULT_PROMPT = (
    "听这段音频，用简洁中文回答，别啰嗦：\n"
    "1) 这是说话、唱歌还是纯音乐？\n"
    "2) 如果是说话：把内容转写出来（只转写，不评论）。\n"
    "3) 如果是唱歌：先逐句听写你听到的歌词，按听到的原样写，听不清的字用□占位，"
    "不要按记忆里的歌词改写。然后写「歌名判断：」——只在确切知道时写歌名/原唱，不确定就写「不确定」，不要猜。"
    "接着给听感点评：音准、节奏、气息、情感各如何，点评时把被评的那一句引出来。\n"
    "4) 声音本身：性别、大致年龄段、真声/假声、音色特点。每项标注把握程度（确定/比较可能/听不出），听不出就写听不出。"
)


class AudioError(RuntimeError):
    """给用户看的错误（中文、不含密钥）。"""


def is_audio_name(name: str) -> bool:
    return str(name or "").strip().lower().endswith(AUDIO_EXTS)


def _ext(name: str) -> str:
    """只认白名单里的音频扩展名，其他一律当没有（它会被拼进临时文件名）。"""
    name = str(name or "").lower()
    ext = name[name.rfind("."):] if "." in name else ""
    return ext if ext in AUDIO_EXTS else ""


def _is_public_ip(addr: str) -> bool:
    ip = ipaddress.ip_address(str(addr).split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def check_public_url(url: str) -> None:
    """只允许下载公网地址：防止有人伪造文件消息，让 bot 去访问本机、内网或云服务器元数据地址。"""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise AudioError("文件链接不是 http(s) 地址")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise AudioError(f"文件链接的域名解析失败（{parts.hostname}）") from exc
    if not all(_is_public_ip(info[4][0]) for info in infos):
        raise AudioError("文件链接指向本机或内网地址，出于安全不下载")


class PublicOnlyResolver(AbstractResolver):
    """真正连接时用的就是检查过的地址：防止「检查时解析到公网、连接时又解析到内网」的 DNS 重绑定。"""

    def __init__(self) -> None:
        self._inner = ThreadedResolver()

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):  # type: ignore[override]
        infos = await self._inner.resolve(host, port, family)
        if not infos or not all(_is_public_ip(info["host"]) for info in infos):
            raise OSError(f"{host} 解析到了本机或内网地址")
        return infos

    async def close(self) -> None:
        await self._inner.close()


async def fetch_bytes(src: str, max_bytes: int, timeout: float = 120.0, allow_private: bool = False) -> bytes:
    """src 可以是 http(s) 链接、本地路径、file:// 或 base64://。

    http(s) 链接默认只下公网地址，跳转也逐跳检查；allow_private=True 只给本机测试用。
    """
    src = str(src or "").strip()
    if not src:
        raise AudioError("没有拿到音频的下载地址")
    if src.startswith("base64://"):
        return base64.b64decode(src[len("base64://"):])
    if src.startswith("file://"):
        src = src[len("file://"):]
    if src.startswith(("http://", "https://")):
        url = src
        data = None
        connector = None if allow_private else aiohttp.TCPConnector(resolver=PublicOnlyResolver())
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), connector=connector) as session:
                for _ in range(_MAX_REDIRECTS + 1):
                    if not allow_private:
                        await check_public_url(url)  # 先查一遍给出明白的提示；连接时 PublicOnlyResolver 再把关
                    async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=False) as resp:
                        if resp.status in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                            url = urljoin(url, resp.headers["Location"])
                            continue
                        if resp.status != 200:
                            raise AudioError(f"下载音频失败（HTTP {resp.status}，文件链接可能已经过期）")
                        buf = bytearray()
                        async for chunk in resp.content.iter_chunked(64 * 1024):
                            buf.extend(chunk)
                            if len(buf) > max_bytes:
                                break
                        data = bytes(buf)
                        break
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if "本机或内网" in str(exc):
                raise AudioError("文件链接指向本机或内网地址，出于安全不下载") from exc
            raise AudioError(f"下载音频失败（网络错误：{type(exc).__name__}）") from exc
        if data is None:
            raise AudioError("文件链接跳转次数太多")
    elif os.path.isfile(src):
        with open(src, "rb") as f:
            data = f.read(max_bytes + 1)
    else:
        raise AudioError("音频文件在本机找不到（可能已被清理）")
    if len(data) > max_bytes:
        raise AudioError(f"音频文件太大（超过 {max_bytes // 1024 // 1024}MB）")
    return data


def sniff_audio_ext(data: bytes) -> str:
    """按文件头认音频格式（语音条没有文件名）。认不出返回空串。"""
    head = bytes(data[:16])
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if b"#!SILK" in head[:10]:
        return ".silk"
    if head[:5] == b"#!AMR":
        return ".amr"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    return ""


def is_local_audio_file(path: str) -> bool:
    """本机路径是不是一个真的音频文件：扩展名在白名单里、文件头也认得出是音频。
    给「适配器/QQ 端返回的本地路径」把关，免得它返回怪东西时去读本机别的文件。"""
    path = str(path or "")
    if not path or not os.path.isfile(path) or not _ext(path):
        return False
    try:
        with open(path, "rb") as f:
            return bool(sniff_audio_ext(f.read(16)))
    except OSError:
        return False


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def to_small_mp3(raw: bytes, file_name: str, max_seconds: int) -> tuple[bytes, str]:
    """压成 16kHz 单声道 48kbps mp3（听内容足够，10 分钟约 3.6MB）。返回 (字节, 格式)。"""
    ext = _ext(file_name)
    if not ffmpeg_available():
        fmt = _DIRECT_FORMATS.get(ext)
        if fmt and len(raw) <= _MAX_INLINE_BYTES:
            return raw, fmt
        raise AudioError("服务器上没装 ffmpeg，这个格式/大小转不了（装法见说明文档）")

    # m4a 这类容器可能要回头读文件尾，所以落临时文件而不是走管道
    with tempfile.TemporaryDirectory(prefix="listen_audio_") as tmp:
        src = os.path.join(tmp, "in" + (ext or ".bin"))
        dst = os.path.join(tmp, "out.mp3")
        with open(src, "wb") as f:
            f.write(raw)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-y", *_FFMPEG_SAFE_INPUT, "-i", src, "-vn",
            "-t", str(int(max_seconds)), "-ac", "1", "-ar", "16000",
            "-codec:a", "libmp3lame", "-b:a", "48k", dst,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=180)
        except asyncio.TimeoutError:
            proc.kill()
            raise AudioError("ffmpeg 转码超时")
        if proc.returncode != 0 or not os.path.isfile(dst):
            msg = (err or b"").decode(errors="replace").strip().splitlines()
            raise AudioError("ffmpeg 转码失败：" + (msg[-1][:150] if msg else "未知错误"))
        with open(dst, "rb") as f:
            data = f.read()
    if not data:
        raise AudioError("转码后是空的（文件可能不是音频或已损坏）")
    if len(data) > _MAX_INLINE_BYTES:
        raise AudioError("音频太长了，调小「最长听多少秒」再试")
    return data, "mp3"


async def to_wav(raw: bytes, file_name: str, max_seconds: int) -> bytes:
    """转成 16kHz 单声道 wav，给 MaiBot 自带的语音识别模型用（它按 wav 上传）。"""
    if _ext(file_name) == ".wav" and not ffmpeg_available():
        return raw
    if not ffmpeg_available():
        raise AudioError("服务器上没装 ffmpeg，转不了格式（装法见说明文档）；或者在插件配置里填百炼 API Key")
    with tempfile.TemporaryDirectory(prefix="listen_audio_") as tmp:
        src = os.path.join(tmp, "in" + (_ext(file_name) or ".bin"))
        dst = os.path.join(tmp, "out.wav")
        with open(src, "wb") as f:
            f.write(raw)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-y", *_FFMPEG_SAFE_INPUT, "-i", src, "-vn",
            "-t", str(int(max_seconds)), "-ac", "1", "-ar", "16000", "-f", "wav", dst,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=180)
        except asyncio.TimeoutError:
            proc.kill()
            raise AudioError("ffmpeg 转码超时")
        if proc.returncode != 0 or not os.path.isfile(dst):
            msg = (err or b"").decode(errors="replace").strip().splitlines()
            raise AudioError("ffmpeg 转码失败：" + (msg[-1][:150] if msg else "未知错误"))
        with open(dst, "rb") as f:
            return f.read()


async def ask_model(
    audio: bytes,
    fmt: str,
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 180.0,
) -> str:
    """OpenAI 兼容的 chat/completions，音频走 input_audio（base64 data URL），流式收回文字。"""
    if not api_key:
        raise AudioError("插件还没填 API Key（在插件配置里填）")
    b64 = base64.b64encode(audio).decode()
    body = {
        "model": model,
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": f"data:;base64,{b64}", "format": fmt}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/chat/completions"
    parts: list[str] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, json=body, headers=headers) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:300]
                raise AudioError(f"听音频模型报错（HTTP {resp.status}）：{detail}")
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if isinstance(piece, str):
                        parts.append(piece)
                    elif isinstance(piece, list):
                        parts.extend(str(p.get("text", "")) for p in piece if isinstance(p, dict))
    text = "".join(parts).strip()
    if not text:
        raise AudioError("模型没有返回内容")
    return text
