"""
本地数据预处理脚本：
1. 递归读取指定目录下的所有 txt 文件（默认针对 pirate-1/chunk_*.txt）。
2. 对文本进行规范化处理（统一编码、全角转半角、空白整理、去除重复空行等）。
3. 计算规范化文本的 SHA-256 哈希，实现快速重复检测。
4. 输出规范化后的文本文件，并将去重后的代表文本写入单独目录。
5. 记录元数据（原始路径、哈希、字符数等）到 JSON Lines 文件，便于后续处理。

该脚本只依赖标准库，可作为进一步大模型处理前的本地预处理流程。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent.parent / "pirate-1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_local"


@dataclass
class ChunkRecord:
    """记录单个 chunk 文件的关键信息。"""

    original_path: str
    normalized_rel_path: str
    sha256: str
    original_char_len: int
    normalized_char_len: int
    normalized_line_len: int
    is_representative: bool
    duplicate_count: int


def find_txt_files(root: Path) -> List[Path]:
    """递归查找 root 目录下所有 txt 文件。"""
    return sorted([path for path in root.rglob("*.txt") if path.is_file()])


def normalize_text(text: str) -> str:
    """
    规范化文本：
    - 去除 BOM；
    - 统一换行符；
    - 使用 NFKC 进行字符标准化（全角转半角等）；
    - 去除行首尾空白、压缩重复空格；
    - 移除连续空行，仅保留单空行。
    """
    # 去除 UTF-8 BOM
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 全角到半角等标准化
    text = unicodedata.normalize("NFKC", text)

    normalized_lines: List[str] = []
    blank_pending = False

    for raw_line in text.split("\n"):
        # 去除首尾空白，并压缩中间的多余空格
        stripped = raw_line.strip()
        stripped = re.sub(r"[ \t]+", " ", stripped)

        if not stripped:
            # 控制空行只保留一个
            if not blank_pending and normalized_lines:
                normalized_lines.append("")
                blank_pending = True
            continue

        normalized_lines.append(stripped)
        blank_pending = False

    # 去除首尾的空行
    while normalized_lines and not normalized_lines[0]:
        normalized_lines.pop(0)
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()

    return "\n".join(normalized_lines)


def compute_sha256(text: str) -> str:
    """计算文本的 SHA-256 哈希值。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text(path: Path, text: str) -> None:
    """写入文本到指定路径，确保父目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def generate_records(
    root: Path, output_dir: Path, unique_dir: Path
) -> Tuple[List[ChunkRecord], Dict[str, List[Path]]]:
    """
    读取 root 下的所有 txt 文件，生成 ChunkRecord 列表，并写出规范化文本。

    返回：
        - records: 规范化后的元数据记录列表
        - hash_to_paths: 哈希值到原始路径列表的映射（用于分析重复）
    """
    txt_files = find_txt_files(root)
    if not txt_files:
        raise FileNotFoundError(f"未在目录 {root} 下找到任何 txt 文件")

    normalized_root = output_dir / "normalized"
    metadata: List[ChunkRecord] = []
    hash_to_paths: Dict[str, List[Path]] = {}

    for path in txt_files:
        original_text = path.read_text(encoding="utf-8", errors="ignore")
        normalized_text = normalize_text(original_text)
        sha256 = compute_sha256(normalized_text)

        relative_path = path.relative_to(root)
        normalized_rel_path = relative_path.as_posix()
        normalized_path = normalized_root / relative_path

        write_text(normalized_path, normalized_text)

        hash_to_paths.setdefault(sha256, []).append(path)

        record = ChunkRecord(
            original_path=str(path),
            normalized_rel_path=normalized_rel_path,
            sha256=sha256,
            original_char_len=len(original_text),
            normalized_char_len=len(normalized_text),
            normalized_line_len=normalized_text.count("\n") + (1 if normalized_text else 0),
            is_representative=False,  # 暂时占位，稍后更新
            duplicate_count=0,  # 暂时占位，稍后更新
        )
        metadata.append(record)

    # 更新代表性标志和重复计数
    sha_to_first: Dict[str, Path] = {}
    for sha256, paths in hash_to_paths.items():
        sha_to_first[sha256] = paths[0]
        for path in paths:
            for record in metadata:
                if record.original_path == str(path):
                    record.is_representative = path == paths[0]
                    record.duplicate_count = len(paths)
                    break

    # 写出去重后的文本
    for idx, (sha256, paths) in enumerate(hash_to_paths.items(), start=1):
        representative_path = paths[0]
        normalized_rel = Path(representative_path).relative_to(root)
        normalized_path = normalized_root / normalized_rel
        unique_path = unique_dir / f"chunk_{idx:05d}.txt"
        text = normalized_path.read_text(encoding="utf-8")
        write_text(unique_path, text)

    return metadata, hash_to_paths


def write_metadata(records: Iterable[ChunkRecord], output_path: Path) -> None:
    """将元数据写入 JSON Lines 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="chunk txt 本地规范化与去重脚本")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="原始 chunk 文件所在目录，默认使用 pirate-1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="规范化输出根目录，默认 dataProcess/outputs_local",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "metadata.jsonl",
        help="元数据输出路径，默认在输出目录下的 metadata.jsonl",
    )

    args = parser.parse_args()

    root = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    unique_dir = output_dir / "unique"

    print(f"[INFO] 输入目录: {root}")
    print(f"[INFO] 输出目录: {output_dir}")

    metadata, hash_to_paths = generate_records(root, output_dir, unique_dir)
    write_metadata(metadata, args.metadata_file.resolve())

    total_files = len(metadata)
    unique_files = len(hash_to_paths)
    duplicate_files = total_files - unique_files

    print(f"[INFO] 完成处理，共 {total_files} 个文件，唯一内容 {unique_files} 个。")
    print(f"[INFO] 检测到重复文件 {duplicate_files} 个。")
    print(f"[INFO] 元数据写入: {args.metadata_file.resolve()}")
    print(f"[INFO] 规范化文本目录: {output_dir / 'normalized'}")
    print(f"[INFO] 去重文本目录: {unique_dir}")


if __name__ == "__main__":
    main()


