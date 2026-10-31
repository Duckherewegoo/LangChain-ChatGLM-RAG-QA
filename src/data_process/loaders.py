"""多格式文档加载器。

安全说明：scan / load 不限制 root 范围，/v1/ingest 的 body.path 由客户端控制，
暴露到公网前必须加白名单校验（或在部署层限制入口）。
"""

from __future__ import annotations

import json
import os
import typing

from base.exceptions import DocumentLoadError

_MAX_SCAN_DEPTH = 12


class DocumentLoader:
    """支持多种格式的文档加载器。"""

    def __init__(self, allowed_extensions: typing.Iterable[str] | None = None) -> None:
        self._extensions = [
            ext.lower().lstrip(".")
            for ext in (allowed_extensions or [])
        ]

    @property
    def allowed_extensions(self) -> list[str]:
        return list(self._extensions)

    def scan(
        self,
        root: str | os.PathLike[str],
        pattern: str = "**/*",
    ) -> list[str]:
        """在 root 下发现匹配扩展名的文件。

        深度限制通过清空 dirs 实现真正阻断，而非 continue 跳过。
        pattern 为预留参数，当前按扩展名过滤，不使用 pattern。
        """
        base = os.path.abspath(root)
        if not os.path.exists(base):
            raise DocumentLoadError(f"输入路径不存在: {base}")
        if os.path.isfile(base):
            return [base] if self._matches(base) else []

        results: list[str] = []
        for directory, dirs, files in os.walk(base, followlinks=False):
            depth = directory[len(base):].count(os.sep)
            if depth >= _MAX_SCAN_DEPTH:
                dirs.clear()
                continue
            for filename in files:
                full = os.path.join(directory, filename)
                if self._matches(full):
                    results.append(full)
        results.sort()
        return results

    def load(self, path: str | os.PathLike[str]) -> tuple[str, dict[str, typing.Any]]:
        """加载文件文本与基础元数据。"""
        absolute = os.path.abspath(path)
        if not os.path.isfile(absolute):
            raise DocumentLoadError(f"文件不存在: {absolute}")

        extension = os.path.splitext(absolute)[1].lower().lstrip(".")
        if self._extensions and extension not in self._extensions:
            raise DocumentLoadError(
                f"不支持的文件扩展名: .{extension}", details={"path": absolute}
            )

        loader = _LOADERS.get(extension, _load_text)
        text, extra = loader(absolute)
        if not text.strip():
            raise DocumentLoadError("文档规范化后为空。", details={"path": absolute})

        # 核心字段由系统生成，业务元数据不得覆盖
        metadata = {"bytes_size": os.path.getsize(absolute), "extension": extension}
        for key, value in extra.items():
            metadata.setdefault(key, value)
        return text, metadata

    def _matches(self, path: str) -> bool:
        if not self._extensions:
            return True
        return os.path.splitext(path)[1].lower().lstrip(".") in self._extensions


def _load_text(path: str) -> tuple[str, dict[str, typing.Any]]:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            with open(path, encoding=encoding) as stream:
                return stream.read(), {}
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise DocumentLoadError(
                f"文件读取失败: {exc}", details={"path": path}
            ) from exc
    raise DocumentLoadError("无法用 utf-8/gb18030/gbk 解码文件。", details={"path": path})


def _load_json(path: str) -> tuple[str, dict[str, typing.Any]]:
    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise DocumentLoadError(
            f"JSON 解析失败: {exc}", details={"path": path}
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"文件读取失败: {exc}", details={"path": path}
        ) from exc
    if isinstance(data, dict):
        text = data.get("content") or data.get("text")
        extra = {k: v for k, v in data.items() if k not in {"content", "text"}}
        if isinstance(text, str) and text.strip():
            return text, extra
        return json.dumps(data, ensure_ascii=False, indent=2), {}
    if isinstance(data, list):
        return "\n".join(
            json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
            for item in data
        ), {}
    return json.dumps(data, ensure_ascii=False), {}


def _load_jsonl(path: str) -> tuple[str, dict[str, typing.Any]]:
    """逐行解析 JSONL，每行必须是一个独立 JSON 值。"""
    items: list[str] = []
    try:
        with open(path, encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DocumentLoadError(
                        f"JSONL 第 {lineno} 行解析失败: {exc}",
                        details={"path": path, "line": lineno},
                    ) from exc
                items.append(
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else str(value)
                )
    except OSError as exc:
        raise DocumentLoadError(
            f"文件读取失败: {exc}", details={"path": path}
        ) from exc
    if not items:
        raise DocumentLoadError("JSONL 文件为空。", details={"path": path})
    return "\n".join(items), {"lines": len(items)}


def _load_markdown(path: str) -> tuple[str, dict[str, typing.Any]]:
    text, _ = _load_text(path)
    return text, {"format": "markdown"}


def _load_pdf(path: str) -> tuple[str, dict[str, typing.Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError(
            "PDF 解析需要 pypdf。请安装或转换为文本后重试。", details={"path": path}
        ) from exc

    reader = PdfReader(path)
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages), {"pages": len(pages)}


def _load_docx(path: str) -> tuple[str, dict[str, typing.Any]]:
    try:
        from docx import Document
    except ImportError as exc:
        raise DocumentLoadError(
            "DOCX 解析需要 python-docx。请安装或转换为文本后重试。", details={"path": path}
        ) from exc

    document = Document(path)
    paragraphs = [p.text for p in document.paragraphs]
    return "\n\n".join(paragraphs), {"paragraphs": len(paragraphs)}


_LOADERS: dict[str, typing.Callable[[str], tuple[str, dict[str, typing.Any]]]] = {
    "txt": _load_text,
    "text": _load_text,
    "md": _load_markdown,
    "markdown": _load_markdown,
    "json": _load_json,
    "jsonl": _load_jsonl,
    "pdf": _load_pdf,
    "docx": _load_docx,
}


__all__ = ["DocumentLoader"]
