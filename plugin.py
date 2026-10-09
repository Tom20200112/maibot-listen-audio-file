"""听音频文件（MaiBot 1.2.x / 1.3.x 插件）：让麦麦能听聊天里别人发的音频文件（mp3/m4a/wav/amr…）。

QQ 端用 NapCat 或 SnowLuma 都行（MaiBot 官方的 NapCat 适配器、SnowLuma 适配器都支持）。
适配器把「文件」消息转成一行字「[文件] xx.mp3，大小: …，链接: …」，麦麦看得到但听不到。
这个插件：
1. 收到音频文件消息就记下来，并趁下载链接还活着先存一份（QQ 的文件链接大约 13 小时就失效）；
2. 提供 listen_audio_file 工具，麦麦想听时自己调用；
3. /听音频 命令方便手动测试。

听的方式：填了阿里云百炼 API Key 就用 qwen omni 模型完整地听（能听写歌词、点评唱歌）；
没填就退回用你在 MaiBot 里配的语音识别模型（model_config 的 voice 任务，只能转成文字）。
"""

from __future__ import annotations

import html
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import HookMode, ToolParameterInfo, ToolParamType

from .audio_core import (
    DEFAULT_PROMPT,
    AudioError,
    ask_model,
    fetch_bytes,
    is_audio_name,
    to_small_mp3,
    to_wav,
)

logger = logging.getLogger("plugin.listen_audio_file")

# 适配器的格式：[文件] 名字，大小: 123，链接: https://...
FILE_TEXT_RE = re.compile(r"\[文件\] (?P<name>.+?)(?:，大小: (?P<size>\d+))?(?:，链接: (?P<url>https?://\S+))?$")
# QQ 消息编号：NapCat 和 SnowLuma 都是有符号 32 位整数，大约一半是负数
MSG_ID_RE = re.compile(r"-?\d+")
# 适配器接口前缀：官方适配器（NapCat 版、SnowLuma 1.0 起的合并版）都认 adapter.napcat.*；
# SnowLuma 适配器另有同义的 adapter.snowluma.*，前一个调不通时再试它
ADAPTER_API_PREFIXES = ("adapter.napcat.", "adapter.snowluma.")


class PluginSection(PluginConfigBase):
    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本（别改）")


class LLMSection(PluginConfigBase):
    api_key: str = Field(
        default="",
        description="阿里云百炼 API Key（sk- 开头）。填了就用 qwen omni 完整地听；不填就退回用 MaiBot 里配的语音识别模型，只能转成文字",
    )
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="接口地址，一般不用改；百炼控制台给的调用地址不一样就填那个（以 /compatible-mode/v1 结尾）",
    )
    model: str = Field(default="qwen3.5-omni-flash", description="听音频用的模型；想更细可换 qwen3.5-omni-plus（更慢更贵）")
    timeout_seconds: int = Field(default=90, description="等模型回答最多几秒")


class ListenSection(PluginConfigBase):
    max_seconds: int = Field(default=600, description="最长听多少秒，超过的部分不听")
    max_file_mb: int = Field(default=40, description="文件大小上限（MB）")
    cache_on_arrival: bool = Field(default=True, description="收到音频文件就先存一份（QQ 文件链接约 13 小时失效，群文件也常被删）")
    keep_days: int = Field(default=7, description="存的副本留几天")
    allow_private_network: bool = Field(
        default=False,
        description="允许从本机/内网地址下载音频。只在本机测试时打开；平时别开，否则有人伪造文件消息就能让麦麦去访问你的内网",
    )


class ListenAudioConfig(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
    llm: LLMSection = Field(default_factory=LLMSection)
    listen: ListenSection = Field(default_factory=ListenSection)


@dataclass
class AudioRecord:
    msg_id: str
    name: str
    url: str
    sender: str
    ts: float
    stream_id: str = ""
    group_id: str = ""
    file_id: str = ""
    local_path: str = ""


def _message_text(message: dict[str, Any]) -> str:
    """从序列化消息里拼出文字（适配器把文件消息转成了文字段）。"""
    text = str(message.get("processed_plain_text") or "")
    if text:
        return text
    parts = []
    for seg in message.get("raw_message") or []:
        if not isinstance(seg, dict) or seg.get("type") != "text":
            continue
        data = seg.get("data")
        parts.append(data.get("text", "") if isinstance(data, dict) else str(data or ""))
    return "".join(parts)


def _unwrap(result: Any) -> Any:
    """api.call 可能包一层 {"success", "result"}，NapCat 动作可能再包一层 {"status", "data"}。"""
    if isinstance(result, dict) and "result" in result and "success" in result:
        result = result.get("result")
    if isinstance(result, dict) and "data" in result and "status" in result:
        result = result.get("data")
    return result


class ListenAudioFilePlugin(MaiBotPlugin):
    config_model = ListenAudioConfig

    def __init__(self) -> None:
        super().__init__()
        self.recent: dict[str, deque[AudioRecord]] = {}

    async def on_load(self) -> None:
        os.makedirs(self._cache_dir(), exist_ok=True)

    async def on_unload(self) -> None:
        pass

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        del scope, config_data, version

    def _cache_dir(self) -> str:
        try:
            base = str(self.ctx.paths.data_dir)
        except Exception:
            base = os.path.join("data", "plugins", "tom.listen-audio-file")
        return os.path.join(base, "audio_cache")

    @property
    def _max_bytes(self) -> int:
        return int(self.config.listen.max_file_mb) * 1024 * 1024

    # ---------- 1. 收到音频文件就记下、存一份 ----------
    @HookHandler("chat.receive.before_process", name="remember_audio_files", mode=HookMode.OBSERVE)
    async def remember_audio_files(self, message: dict[str, Any] | None = None, **kwargs: Any) -> None:
        del kwargs
        if not isinstance(message, dict):
            return
        m = FILE_TEXT_RE.search(_message_text(message))
        if not m or not is_audio_name(m.group("name")):
            return
        info = message.get("message_info") or {}
        user = info.get("user_info") or {}
        group = info.get("group_info") or {}
        rec = AudioRecord(
            msg_id=str(message.get("message_id") or ""),
            name=m.group("name"),
            url=m.group("url") or "",
            sender=str(user.get("user_cardname") or user.get("user_nickname") or user.get("user_id") or "对方"),
            ts=time.time(),
            stream_id=str(message.get("session_id") or ""),
            group_id=str(group.get("group_id") or ""),
        )
        if not await self._confirm_on_qq(rec):
            return
        self.recent.setdefault(rec.stream_id, deque(maxlen=20)).append(rec)
        logger.info(f"[听音频文件] 记下 {rec.sender} 发的「{rec.name}」 msg_id={rec.msg_id}")
        if self.config.listen.cache_on_arrival:
            await self._save_copy(rec)

    async def _adapter_call(self, name: str, **kwargs: Any) -> Any:
        """调 QQ 适配器的接口（name 不带前缀，如 "message.get_msg"），两种前缀依次试。

        MaiBot 的 api.call 失败时不抛异常，而是返回 {"success": False, "error": ...}，所以要看返回值。
        """
        errors: list[str] = []
        for prefix in ADAPTER_API_PREFIXES:
            try:
                res = await self.ctx.api.call(prefix + name, **kwargs)
            except Exception as exc:
                errors.append(f"{prefix}{name}: {exc}")
                continue
            if isinstance(res, dict) and res.get("success") is False:
                errors.append(f"{prefix}{name}: {res.get('error') or '调用失败'}")
                continue
            return _unwrap(res)
        raise RuntimeError("；".join(errors) or f"调不通适配器接口 {name}")

    async def _check_on_qq(self, msg_id: str) -> tuple[bool | None, str, str]:
        """到 QQ 端查原消息，看它是不是真的文件消息。

        返回 (是不是真文件, file_id, 原消息里的下载链接)。是不是真文件：True 是；False 不是，
        说明有人在聊天里手打了一行「[文件] …，链接: …」；None 查不了（QQ 端只认最近几小时的消息）。
        file_id 用来在链接过期后去群文件重新要地址，所以到达时就查。
        """
        if not MSG_ID_RE.fullmatch(msg_id):
            return None, "", ""
        try:
            detail = await self._adapter_call("message.get_msg", message_id=int(msg_id))
        except Exception as exc:
            logger.info(f"[听音频文件] 到 QQ 端查原消息失败：{exc}")
            return None, "", ""
        segs = detail.get("message") if isinstance(detail, dict) else None
        if isinstance(segs, list):
            for seg in segs:
                if isinstance(seg, dict) and seg.get("type") == "file":
                    data = seg.get("data") or {}
                    return True, str(data.get("file_id") or data.get("id") or ""), str(data.get("url") or "")
            return False, "", ""
        if isinstance(segs, str):  # QQ 端设成字符串消息格式时是 CQ 码：NapCat 写 file_id=，SnowLuma 写 id=
            hit = re.search(r"\[CQ:file,(?:[^\]]*,)?(?:file_id|id)=([^,\]]+)", segs)
            return (True, html.unescape(hit.group(1)), "") if hit else (False, "", "")
        return None, "", ""

    async def _confirm_on_qq(self, rec: AudioRecord) -> bool:
        """确认 rec 是真的文件消息并补上 file_id；是手打的假文件消息就返回 False。"""
        real, rec.file_id, real_url = await self._check_on_qq(rec.msg_id)
        if real is False:
            logger.info(f"[听音频文件] msg_id={rec.msg_id} 里的「[文件] …」是聊天文字，不是真文件，不理它")
            return False
        if real_url.startswith(("http://", "https://")):
            rec.url = real_url  # 以 QQ 端原消息里的链接为准
        return True

    async def _save_copy(self, rec: AudioRecord) -> None:
        try:
            data = await fetch_bytes(rec.url, self._max_bytes, allow_private=self.config.listen.allow_private_network)
            os.makedirs(self._cache_dir(), exist_ok=True)
            safe = re.sub(r"[^\w.\-]+", "_", rec.name)[-80:]
            path = os.path.join(self._cache_dir(), f"{re.sub(r'[^0-9-]', '', rec.msg_id) or 'x'}_{safe}")
            with open(path, "wb") as f:
                f.write(data)
            rec.local_path = path
        except Exception as exc:
            logger.info(f"[听音频文件] 预存「{rec.name}」失败（不影响之后现下）：{exc}")
        self._prune_cache()

    def _prune_cache(self) -> None:
        cutoff = time.time() - int(self.config.listen.keep_days) * 86400
        try:
            names = os.listdir(self._cache_dir())
        except OSError:
            return
        for fn in names:
            path = os.path.join(self._cache_dir(), fn)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                pass

    # ---------- 2. 工具：真正去听 ----------
    @Tool(
        "listen_audio_file",
        description=(
            "听聊天里别人发的音频文件（聊天里显示成「[文件] xxx.mp3，大小: …」这样的一行，m4a/wav/amr/flac 等也行；"
            "文件不会自动听，要听就得调用这个工具）。有人发了音频文件、想让你听听里面是什么、或让你点评唱得怎么样时用。"
            "填那条文件消息的 msg_id；不填就听这个聊天里最近的一个音频文件。"
        ),
        parameters=[
            ToolParameterInfo(
                name="msg_id", param_type=ToolParamType.STRING, required=False,
                description=(
                    "那条「[文件] …」消息的 msg_id，只填那串数字（可能带负号，例如 -1592304852）；"
                    "不填就听这个聊天最近的一个音频文件。想重点听什么写在 focus 里，别写在这里"
                ),
            ),
            ToolParameterInfo(
                name="focus", param_type=ToolParamType.STRING, required=False,
                description="这次重点听什么，例如「唱的哪首歌、音准气息怎么样」「逐句听写歌词」；不填按通用角度听",
            ),
        ],
        visibility="visible",
        timeout_ms=150000,
    )
    async def handle_listen_audio_file(self, msg_id: str = "", focus: str = "", **kwargs: Any) -> dict[str, Any]:
        stream_id = str(kwargs.get("stream_id") or kwargs.get("chat_id") or "")
        msg_id = str(msg_id or "").strip().strip("\"'「」#")
        focus = str(focus or "").strip()
        if msg_id and not MSG_ID_RE.fullmatch(msg_id):
            # 模型偶尔把「想重点听什么」错填进 msg_id（deepseek-v4-flash 实测出现过）：
            # 这不是编号，当成 focus，改听这个聊天最近的音频文件
            logger.info(f"[听音频文件] msg_id 不是数字编号（{msg_id[:40]}），当成想重点听的内容，改听最近一个")
            focus = focus or msg_id
            msg_id = ""
        try:
            rec, text, full = await self._listen(stream_id, msg_id, focus)
        except AudioError as exc:
            return {"name": "listen_audio_file", "content": f"没听成：{exc}。如实告诉对方你这次没听到，别编听感。"}
        except Exception as exc:
            logger.warning(f"[听音频文件] 出错：{type(exc).__name__}: {exc}")
            return {"name": "listen_audio_file", "content": "没听成（插件内部出错）。如实告诉对方你这次没听到，别编听感。"}
        if full:
            head = f"（你刚听了 {rec.sender} 发的音频文件「{rec.name}」）"
        else:
            head = (f"（你刚把 {rec.sender} 发的音频文件「{rec.name}」里的话转成了文字——"
                    "只有文字，听不出唱得怎么样、是男声还是女声）")
        return {
            "name": "listen_audio_file",
            "content": (
                f"{head}\n{text}\n"
                "——以上是你自己听出来的，用你自己的话讲给对方；"
                "标了「不确定/听不出」的地方别说死，歌名原唱没把握就别断言。"
            ),
        }

    # ---------- 3. 手动测试命令 ----------
    @Command("listen_audio_test", description="手动测试：/听音频 [想重点听什么]，听这个聊天最近的一个音频文件",
             pattern=r"^/听音频(?:\s+(?P<focus>.+))?$")
    async def handle_listen_command(self, stream_id: str = "", **kwargs: Any):
        groups = kwargs.get("matched_groups") if isinstance(kwargs.get("matched_groups"), dict) else {}
        focus = str(groups.get("focus") or "").strip()
        try:
            rec, text, full = await self._listen(stream_id, "", focus)
            reply = f"🎧 {rec.name}{'' if full else '（只转了文字）'}\n{text}"
        except AudioError as exc:
            reply = f"没听成：{exc}"
        except Exception as exc:
            logger.warning(f"[听音频文件] 出错：{type(exc).__name__}: {exc}")
            reply = f"没听成（插件内部出错：{type(exc).__name__}）"
        await self.ctx.send.text(reply, stream_id)
        return True, "已回复听音频结果", True

    # ---------- 找文件、取字节、送模型 ----------
    async def _find(self, stream_id: str, msg_id: str) -> AudioRecord | None:
        recs = list(self.recent.get(stream_id, []))
        if not msg_id:
            return recs[-1] if recs else None
        for r in reversed(recs):
            if r.msg_id == msg_id:
                return r
        # 插件装上之前发的、或重启后名单清空了：去库里按 msg_id 找那条消息
        try:
            res = await self.ctx.message.get_by_id(msg_id, stream_id=stream_id)
        except Exception as exc:
            logger.info(f"[听音频文件] 按 msg_id 查消息失败：{exc}")
            return None
        msg = res.get("message") if isinstance(res, dict) and isinstance(res.get("message"), dict) else res
        if not isinstance(msg, dict):
            return None
        m = FILE_TEXT_RE.search(_message_text(msg))
        if not m or not is_audio_name(m.group("name")):
            return None
        info = msg.get("message_info") or {}
        user = info.get("user_info") or {}
        group = info.get("group_info") or {}
        rec = AudioRecord(
            msg_id=msg_id, name=m.group("name"), url=m.group("url") or "",
            sender=str(user.get("user_cardname") or user.get("user_nickname") or "对方"),
            ts=time.time(), stream_id=stream_id, group_id=str(group.get("group_id") or ""),
        )
        if not await self._confirm_on_qq(rec):
            return None
        saved = [fn for fn in self._safe_listdir() if fn.startswith(f"{msg_id}_")]
        if saved:
            rec.local_path = os.path.join(self._cache_dir(), saved[0])
        return rec

    def _safe_listdir(self) -> list[str]:
        try:
            return os.listdir(self._cache_dir())
        except OSError:
            return []

    async def _get_bytes(self, rec: AudioRecord) -> bytes:
        errors = []
        for src in (rec.local_path, rec.url):
            if not src:
                continue
            try:
                return await fetch_bytes(src, self._max_bytes, allow_private=self.config.listen.allow_private_network)
            except AudioError as exc:
                errors.append(str(exc))
        # 链接过期了：群文件还在的话重新要一个下载地址
        if rec.file_id and rec.group_id.isdigit():
            try:
                ret = await self._adapter_call(
                    "file.get_group_file_url",
                    params={"group_id": int(rec.group_id), "file_id": rec.file_id, "busid": 102},
                )
                url = ret.get("url") if isinstance(ret, dict) else None
                if url:
                    return await fetch_bytes(url, self._max_bytes, allow_private=self.config.listen.allow_private_network)
            except Exception as exc:
                errors.append(f"群文件重取失败：{exc}")
        raise AudioError(errors[-1] if errors else "拿不到这个文件（可能已被删除或过期）")

    async def _listen(self, stream_id: str, msg_id: str, focus: str) -> tuple[AudioRecord, str, bool]:
        rec = await self._find(stream_id, msg_id)
        if rec is None:
            raise AudioError("没找到这个音频文件（插件装好之后发的才记得住；或者这条 msg_id 不是音频文件）")
        raw = await self._get_bytes(rec)
        cfg = self.config
        api_key = str(cfg.llm.api_key or "").strip()
        if api_key:
            audio, fmt = await to_small_mp3(raw, rec.name, int(cfg.listen.max_seconds))
            prompt = DEFAULT_PROMPT
            if focus:
                prompt = (
                    f"听这段音频，按下面的要求回答，简洁具体、别说套话：\n{focus}\n"
                    "歌名/原唱这类事实不确定就明说不确定，不要猜；点评要把被评的那句引出来。"
                )
            text = await ask_model(
                audio, fmt, prompt, base_url=str(cfg.llm.base_url), api_key=api_key,
                model=str(cfg.llm.model), timeout=float(cfg.llm.timeout_seconds),
            )
            return rec, text, True
        # 没填 key：用 MaiBot 自己配的语音识别模型（只能转文字）
        wav = await to_wav(raw, rec.name, int(cfg.listen.max_seconds))
        res = await self.ctx.llm.transcribe_audio(wav, task_name="voice")
        text = str((res or {}).get("text") or (res or {}).get("content") or "").strip() if isinstance(res, dict) else str(res or "")
        if not text:
            err = (res or {}).get("error") if isinstance(res, dict) else ""
            raise AudioError(f"语音识别没有返回内容{f'（{err}）' if err else ''}；检查 model_config 的 voice 任务配了没有，或者在插件配置里填百炼 API Key")
        return rec, text, False


def create_plugin() -> ListenAudioFilePlugin:
    return ListenAudioFilePlugin()
