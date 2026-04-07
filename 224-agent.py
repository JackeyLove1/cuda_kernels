"""
OpenAI SDK agent loop with tool use: 多轮对话直到模型不再请求工具调用。
依赖: pip install "openai>=1.40.0" "python-dotenv>=1.0.0"

环境变量（可在仓库根目录或本脚本同目录的 .env 中设置；已存在的环境变量优先，不被 .env 覆盖）：
- OPENAI_API_KEY（必填）
- OPENAI_BASE_URL（可选，兼容代理 / Azure 等）
- OPENAI_AGENT_MODEL 或 OPENAI_MODEL（可选，默认 gpt-4o-mini）
- AGENT_TOOL_TRACE=1 / true / yes：在 stderr 输出工具调用的 trace_id / span_id 日志（logging INFO）
- AGENT_FS_ROOT：read_file / write_file 仅允许该目录及其子路径；未设置时默认为当前工作目录

自定义工具：使用 @tool 装饰函数，从函数名、docstring、类型注解生成 schema。
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ForwardRef, Union, get_args, get_origin, get_type_hints

from dotenv import load_dotenv
from openai import OpenAI


def load_env_from_dotenv() -> None:
    """从 .env 加载变量（不覆盖已在进程环境中设置的键）。"""
    script_dir = Path(__file__).resolve().parent
    for path in (script_dir / ".env", Path.cwd() / ".env"):
        if path.is_file():
            load_dotenv(path, override=False)


def default_chat_model() -> str:
    """模型名：OPENAI_AGENT_MODEL，否则 OPENAI_MODEL，否则内置默认。"""
    return (
        os.environ.get("OPENAI_AGENT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o-mini"
    )


_SYSTEM_PROMPT = (
    "你是一个可以使用工具完成任务的助手。"
    "需要计算或格式化文本时优先调用工具，再基于工具结果用自然语言回答用户。"
    "读写本地文件时使用 read_file / write_file / patch_file，路径须位于环境变量 AGENT_FS_ROOT "
    "所允许的目录内（未设置时以运行时的当前工作目录为根）。patch_file 用精确子串替换；"
    "默认要求 old_string 在文件中唯一出现一次，避免误改。"
)

_logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _ensure_tool_trace_logging() -> None:
    """若 AGENT_TOOL_TRACE 开启且 logger 无 handler，则向 stderr 输出 INFO 日志。"""
    if not _env_flag("AGENT_TOOL_TRACE"):
        return
    if _logger.handlers:
        _logger.setLevel(logging.INFO)
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
    _logger.addHandler(h)
    _logger.setLevel(logging.INFO)


def _chat_until_assistant_reply(
    client: OpenAI,
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_turns: int = 16,
) -> str:
    """
    在已有 messages 末尾已追加本轮 user 的前提下，请求模型并执行 tool 循环，
    直到 assistant 返回无 tool_calls；就地修改 messages。
    """
    _ensure_tool_trace_logging()
    turn_trace_id = str(uuid.uuid4())
    tools = openai_tool_schemas()
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message
        messages.append(_assistant_message_to_dict(msg))

        if not getattr(msg, "tool_calls", None) or len(msg.tool_calls) == 0:
            return (msg.content or "").strip() or "(模型未返回文本)"

        for tc in msg.tool_calls:
            fn = tc.function
            span_id = str(uuid.uuid4())
            out = run_tool(
                fn.name,
                fn.arguments or "{}",
                trace_id=turn_trace_id,
                tool_call_id=tc.id,
                span_id=span_id,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": out,
                }
            )

    return f"已达到最大轮次限制 ({max_turns})，请增大 max_turns 或简化任务。"


# ---------------------------------------------------------------------------
# 装饰器注册与 schema 生成
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, Any]  # OpenAI function.parameters (JSON Schema object)


_TOOL_SPECS: list[ToolSpec] = []


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    将函数注册为 agent 工具。
    - name: 覆盖工具名；默认使用函数名 __name__。
    - description: 覆盖工具说明；默认使用 docstring 首段（遇空行或 Args/Parameters 节结束）。
    - 参数类型来自类型注解；参数说明从 docstring 的 Google Args / Sphinx :param 解析。
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        raw_doc = inspect.getdoc(fn) or ""
        tool_name = (name or fn.__name__).strip()
        tool_desc = (
            description if description is not None else _description_from_docstring(raw_doc)
        )
        param_docs = _parse_param_descriptions(raw_doc)
        params_schema = _build_parameters_schema(fn, param_docs)
        spec = ToolSpec(
            name=tool_name,
            description=tool_desc,
            fn=fn,
            parameters=params_schema,
        )
        _TOOL_SPECS.append(spec)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper.__tool_spec__ = spec  # 便于调试或 introspection
        return wrapper

    return decorator


def _description_from_docstring(doc: str) -> str:
    if not doc.strip():
        return ""
    lines: list[str] = []
    for line in doc.splitlines():
        s = line.strip()
        if not s:
            if lines:
                break
            continue
        low = s.lower()
        if low.startswith(
            ("args:", "arguments:", "parameters:", "returns:", "return:", "yields:", "raises:")
        ):
            break
        if s.startswith(":param "):
            break
        lines.append(s)
    return "\n".join(lines).strip()


def _parse_param_descriptions(doc: str) -> dict[str, str]:
    """Google Args 块与 Sphinx :param 行。"""
    out: dict[str, str] = {}
    if not doc:
        return out

    # Sphinx: :param name: text
    for m in re.finditer(r"^:param\s+(\w+)\s*:\s*(.+)$", doc, re.MULTILINE):
        out[m.group(1)] = m.group(2).strip()

    in_args = False
    args_indent: int | None = None
    for line in doc.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if not in_args:
            if low in ("args:", "arguments:", "parameters:"):
                in_args = True
                args_indent = None
            continue

        if not stripped:
            continue
        # 新的小节标题（简单启发式）
        if (
            stripped.endswith(":")
            and low
            in (
                "returns:",
                "return:",
                "yields:",
                "raises:",
                "examples:",
                "note:",
                "notes:",
            )
        ):
            break

        indent = len(line) - len(line.lstrip(" "))
        if args_indent is None and stripped:
            args_indent = indent

        if args_indent is not None and indent < args_indent and stripped and ":" in stripped:
            # 可能进入其它顶级节
            if not re.match(r"^\w+\s*:", stripped):
                break

        m2 = re.match(r"^(\w+)\s*:\s*(.*)$", stripped)
        if m2:
            pname, pdesc = m2.group(1), m2.group(2).strip()
            if pname not in out and pname.lower() not in ("args", "arguments", "parameters"):
                out[pname] = pdesc

    return out


def _is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        return any(a is type(None) for a in get_args(annotation))
    return False


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _json_type_for(annotation: Any) -> dict[str, Any]:
    """单个参数 -> JSON Schema 片段（含 type / items 等）。"""
    if annotation is inspect.Parameter.empty:
        return {}

    ann = _strip_optional(annotation)
    origin = get_origin(ann)

    if ann is str:
        return {"type": "string"}
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if ann is bool:
        return {"type": "boolean"}

    if origin is list:
        args = get_args(ann)
        item_ann = args[0] if args else Any
        inner = _json_type_for(item_ann)
        return {"type": "array", "items": inner if inner else {"type": "string"}}

    if origin is dict:
        return {"type": "object", "additionalProperties": True}

    if ann is Any or ann is Ellipsis:
        return {"type": "string"}

    # 其余（含 ForwardRef、自定义类等）：宽松为 string（避免生成非法 schema）
    if isinstance(ann, (str, ForwardRef)):
        return {"type": "string"}
    return {"type": "string"}


def _build_parameters_schema(
    fn: Callable[..., Any],
    param_docs: dict[str, str],
) -> dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ann = hints.get(pname, param.annotation)
        schema = _json_type_for(ann)
        if pname in param_docs:
            schema = {**schema, "description": param_docs[pname]}
        properties[pname] = schema

        if param.default is inspect.Parameter.empty and not _is_optional(ann):
            required.append(pname)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def openai_tool_schemas() -> list[dict[str, Any]]:
    """当前已注册的 @tool，转为 Chat Completions 的 tools 列表。"""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in _TOOL_SPECS
    ]


def _run_registered_tool(spec: ToolSpec, args: dict[str, Any]) -> Any:
    return spec.fn(**args)


def run_tool(
    name: str,
    arguments_json: str,
    *,
    trace_id: str | None = None,
    tool_call_id: str | None = None,
    span_id: str | None = None,
) -> str:
    def _trace_log(msg: str, *args: object) -> None:
        if trace_id and _env_flag("AGENT_TOOL_TRACE"):
            _logger.info(msg, *args)

    spec_by_name = {s.name: s for s in _TOOL_SPECS}
    _trace_log(
        "tool_call_start trace_id=%s span_id=%s openai_tool_call_id=%s tool=%s",
        trace_id,
        span_id or "-",
        tool_call_id or "-",
        name,
    )
    if name not in spec_by_name:
        err = json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
        _trace_log(
            "tool_call_end trace_id=%s span_id=%s tool=%s ok=false",
            trace_id,
            span_id or "-",
            name,
        )
        return err
    spec = spec_by_name[name]
    try:
        raw = json.loads(arguments_json) if arguments_json else {}
        result = _run_registered_tool(spec, raw)
        if isinstance(result, (dict, list)):
            out = json.dumps(result, ensure_ascii=False)
        else:
            out = str(result)
        _trace_log(
            "tool_call_end trace_id=%s span_id=%s tool=%s ok=true",
            trace_id,
            span_id or "-",
            name,
        )
        return out
    except TypeError as e:
        _trace_log(
            "tool_call_end trace_id=%s span_id=%s tool=%s ok=false err=TypeError",
            trace_id,
            span_id or "-",
            name,
        )
        return json.dumps({"error": f"bad arguments: {e}"}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        _trace_log(
            "tool_call_end trace_id=%s span_id=%s tool=%s ok=false err=%s",
            trace_id,
            span_id or "-",
            name,
            type(e).__name__,
        )
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 示例工具（装饰器定义；启动 agent 前必须已 import / 执行注册）
# ---------------------------------------------------------------------------


@tool()
def add(a: float, b: float) -> float:
    """计算两个数字之和。

    Args:
        a: 第一个数
        b: 第二个数
    """
    return float(a) + float(b)


@tool(name="echo_upper", description="将输入字符串转为大写并返回（演示覆盖 name/description）。")
def echo(text: str) -> str:
    """
    此函数体的首段 docstring 被 description= 覆盖时不会暴露给模型。
    """
    return text.upper()


def _agent_fs_root() -> Path:
    raw = (os.environ.get("AGENT_FS_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_path_under_fs_root(path_str: str) -> Path:
    """将路径解析为绝对路径，且必须位于 AGENT_FS_ROOT（或默认 cwd）之下。"""
    root = _agent_fs_root()
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        full = (root / candidate).resolve()
    else:
        full = candidate.resolve()
    root_r = root.resolve()
    try:
        full.relative_to(root_r)
    except ValueError as e:
        raise PermissionError(
            f"path must be under AGENT_FS_ROOT ({root_r}): {path_str!r}"
        ) from e
    return full


@tool()
def read_file(path: str, max_bytes: int = 512_000) -> str:
    """读取本地文本文件（UTF-8，无法解码的字节以替换字符显示）。

    Args:
        path: 相对 AGENT_FS_ROOT 的路径，或位于该根下的绝对路径
        max_bytes: 最多读取字节数，超出部分截断并在文末注明
    """
    p = _resolve_path_under_fs_root(path)
    if not p.is_file():
        raise FileNotFoundError(f"not a file or does not exist: {p}")
    total = p.stat().st_size
    chunk = min(total, max_bytes)
    with p.open("rb") as f:
        data = f.read(chunk)
    truncated = total > max_bytes
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n\n[truncated: first {max_bytes} of {total} bytes]"
    return text


@tool()
def write_file(path: str, content: str, append: bool = False) -> dict[str, Any]:
    """写入文本文件（UTF-8）；父目录不存在则创建。

    Args:
        path: 相对 AGENT_FS_ROOT 的路径，或位于该根下的绝对路径
        content: 要写入的完整文本
        append: 为 true 时在文件末尾追加，否则覆盖
    """
    p = _resolve_path_under_fs_root(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8", newline="") as f:
        n = f.write(content)
    return {"ok": True, "path": str(p), "chars_written": n, "append": append}


@tool()
def patch_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> dict[str, Any]:
    """对文本文件做一次或多次精确子串替换（UTF-8 整文件读写，区分大小写）。

    replace_all 为 false 时，old_string 必须在文件中恰好出现 1 次，否则拒绝写入以防误改。
    replace_all 为 true 时替换所有出现次数（至少 1 次）。
    old_string 不能为空。

    Args:
        path: 相对 AGENT_FS_ROOT 的路径，或位于该根下的绝对路径
        old_string: 要被替换的原文（与文件内容逐字符一致）
        new_string: 替换后的文本
        replace_all: 为 true 时替换每一处匹配；为 false 时要求唯一匹配
    """
    if not old_string:
        raise ValueError("old_string must be non-empty")
    p = _resolve_path_under_fs_root(path)
    if not p.is_file():
        raise FileNotFoundError(f"not a file or does not exist: {p}")
    text = p.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        raise ValueError("old_string not found in file")
    if not replace_all and count > 1:
        raise ValueError(
            f"old_string matches {count} times; use replace_all=true or narrow old_string "
            "so the match is unique"
        )
    if replace_all:
        new_text = text.replace(old_string, new_string)
        replacements = count
    else:
        new_text = text.replace(old_string, new_string, 1)
        replacements = 1
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    return {
        "ok": True,
        "path": str(p),
        "replacements": replacements,
        "replace_all": replace_all,
    }


def _assistant_message_to_dict(msg: Any) -> dict[str, Any]:
    """将 SDK 的 assistant message 转为 API 可再次发送的 dict。"""
    out: dict[str, Any] = {
        "role": "assistant",
        "content": msg.content or "",
    }
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def agent_loop(
    client: OpenAI,
    *,
    user_text: str,
    model: str | None = None,
    max_turns: int = 16,
) -> str:
    """
    Agent loop: 请求模型 -> 若有 tool_calls 则执行并追加 tool 消息 -> 再请求，直到无 tool_calls。
    返回最后一轮 assistant 的文本内容（可能为空则拼接说明）。
    """
    model = model or default_chat_model()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return _chat_until_assistant_reply(client, messages, model=model, max_turns=max_turns)


def main() -> None:
    load_env_from_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "请设置 OPENAI_API_KEY（环境变量或与本脚本同目录 / 当前工作目录下的 .env）",
            file=sys.stderr,
        )
        sys.exit(1)

    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = default_chat_model()

    messages: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    pending_cli = " ".join(sys.argv[1:]).strip()

    print("多轮对话。输入 /exit 退出；Ctrl+C 中断退出。", flush=True)

    while True:
        if pending_cli:
            user_line = pending_cli
            pending_cli = ""
        else:
            try:
                user_line = input("You> ").strip()
            except KeyboardInterrupt:
                print("\n已退出。", flush=True)
                break

        if user_line == "/exit":
            print("已退出。", flush=True)
            break
        if not user_line:
            continue

        messages.append({"role": "user", "content": user_line})
        reply = _chat_until_assistant_reply(client, messages, model=model, max_turns=16)
        print(reply, flush=True)


if __name__ == "__main__":
    main()
