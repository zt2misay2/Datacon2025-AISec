"""
模型处理脚本：
1. 读取 dataProcess 阶段输出的去重 chunk（默认 dataProcess/outputs_local/unique）。
2. 调用外部大模型生成嵌入，依据余弦相似度进行近似重复聚类。
3. 对每个簇按长度优先合并，并在必要时按语义拆分，限制单段长度。
4. 输出簇分段文件、聚类元数据以及不超过 100 条的 outputs 列表。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Optional

from difflib import SequenceMatcher
import numpy as np

# 确保可以导入项目根目录下的 chat.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat import get_response_embed_blocking, get_response_embed_concurrent

DEFAULT_INPUT_DIR = Path("dataProcess") / "outputs_local" / "unique"
DEFAULT_OUTPUT_DIR = Path("modelProcess") / "outputs_model"


def longest_common_prefix_length(a: str, b: str) -> int:
    """计算两个字符串的公共前缀长度。"""
    max_len = min(len(a), len(b))
    idx = 0
    while idx < max_len and a[idx] == b[idx]:
        idx += 1
    return idx


def trim_redundant_prefix(reference_texts: Sequence[str], text: str, min_overlap: int = 80) -> str:
    """
    去除与参考文本共享的大段前缀，以减少重复。
    仅当公共前缀长度超过 min_overlap 时才截断。
    """
    for ref in reference_texts:
        prefix_len = longest_common_prefix_length(ref, text)
        if prefix_len >= min_overlap:
            return text[prefix_len:].lstrip()
    return text


@dataclass
class Chunk:
    """基础 chunk 信息。"""

    chunk_id: str
    path: Path
    text: str


@dataclass
class ClusterResult:
    """聚类后的结果。"""

    cluster_id: int
    member_ids: List[str] = field(default_factory=list)
    representative_id: Optional[str] = None
    segments: List[str] = field(default_factory=list)


def load_chunks(input_dir: Path) -> List[Chunk]:
    """加载目录下的所有 txt 文件，返回 Chunk 列表。"""
    files = sorted([p for p in input_dir.glob("*.txt") if p.is_file()])
    chunks: List[Chunk] = []
    for idx, path in enumerate(files):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        chunk_id = f"chunk_{idx:05d}"
        chunks.append(Chunk(chunk_id=chunk_id, path=path, text=text))
    if not chunks:
        raise FileNotFoundError(f"在目录 {input_dir} 下未找到有效的 txt chunk")
    return chunks


def compute_embeddings(
    chunks: Sequence[Chunk],
    concurrency: int = 50,
) -> np.ndarray:
    """调用外部模型生成嵌入，返回 ndarray 形式的向量。"""
    texts = [chunk.text for chunk in chunks]

    def embed_func(query: str) -> List[float]:
        return get_response_embed_blocking(query)

    embeddings = get_response_embed_concurrent(texts, embed_func, concurrency=concurrency)
    if embeddings.size == 0:
        raise RuntimeError("嵌入生成失败，返回空数组")
    return embeddings.astype(np.float32)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """对嵌入向量进行 L2 归一化，便于计算余弦相似度。"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def cluster_by_similarity(
    embeddings: np.ndarray,
    chunk_ids: Sequence[str],
    threshold: float = 0.92,
) -> List[List[int]]:
    """
    基于余弦相似度的简单增量聚类。
    - 使用聚类质心（向量和归一化）进行匹配；
    - 若与任何已有聚类的相似度超过阈值，则归入该聚类；
    - 否则创建新的聚类。
    返回索引列表的列表，每个子列表表示聚类成员在原数组中的索引。
    """
    normalized = normalize_embeddings(embeddings)
    clusters: List[Dict[str, any]] = []

    for idx, vector in enumerate(normalized):
        assigned = False
        for cluster in clusters:
            centroid = cluster["centroid"]
            similarity = float(np.dot(vector, centroid))
            if similarity >= threshold:
                cluster["indices"].append(idx)
                cluster["vector_sum"] += vector
                norm = np.linalg.norm(cluster["vector_sum"])
                if norm > 0:
                    cluster["centroid"] = cluster["vector_sum"] / norm
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "indices": [idx],
                    "vector_sum": vector.copy(),
                    "centroid": vector.copy(),
                }
            )

    return [cluster["indices"] for cluster in clusters]


def split_text_by_length(text: str, max_length: int, min_length: int = 1800) -> List[str]:
    """
    将超长文本按语义分段，尝试截断在句子或段落边界。
    - 优先按双换行分段；
    - 若段落仍过长，按句子切分；
    - 继续过长则按 max_length 强制切分。
    返回分段列表。
    """
    if len(text) <= max_length:
        return [text.strip()]

    segments: List[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) <= max_length:
            buffer = candidate
            continue

        if buffer:
            segments.extend(split_text_by_length(buffer.strip(), max_length, min_length))
            buffer = ""

        if len(para) <= max_length:
            buffer = para
        else:
            sentences = [s.strip() for s in re.split(r"(?<=[。？！.!?])\s*", para) if s.strip()]
            temp = ""
            for sentence in sentences:
                candidate_sentence = f"{temp} {sentence}".strip() if temp else sentence
                if len(candidate_sentence) <= max_length:
                    temp = candidate_sentence
                    continue
                if temp:
                    segments.extend(split_text_by_length(temp.strip(), max_length, min_length))
                temp = sentence
            if temp:
                if len(temp) <= max_length:
                    buffer = temp
                else:
                    for i in range(0, len(temp), max_length):
                        segments.append(temp[i : i + max_length].strip())
                    buffer = ""

    if buffer:
        segments.extend(split_text_by_length(buffer.strip(), max_length, min_length))

    return [seg for seg in segments if seg]


def merge_cluster_texts(
    texts: Sequence[str],
    similarity_threshold: float = 0.88,
    max_merged_length: int = 3000,
    split_max_length: int = 2400,
    split_min_length: int = 1600,
) -> List[str]:
    """
    依据长度优先保留 chunk 的完整性，并对结果做长度控制：
    - 先按字符数从大到小排序，选取最长 chunk 作为主体。
    - 仅当新增 chunk 与主体差异明显且合并后长度未超限时才追加。
    - 合并输出会进一步按语义拆分，保证单段长度上限。
    """
    if not texts:
        return []
    ordered = sorted([t for t in texts if t], key=len, reverse=True)
    primary = ordered[0]

    kept_texts = [primary]
    combined = primary
    merged_outputs: List[str] = []

    for text in ordered[1:]:
        if any(text == existing or text in existing or existing in text for existing in kept_texts):
            continue
        similarity_scores = [
            SequenceMatcher(None, text, existing).ratio() for existing in kept_texts
        ]
        if similarity_scores and max(similarity_scores) >= similarity_threshold:
            continue

        if len(combined) + len(text) + 2 > max_merged_length:
            merged_outputs.append(combined.strip())
            references = list(reversed(merged_outputs[-2:])) if len(merged_outputs) >= 2 else merged_outputs[-1:]
            references.extend(kept_texts)
            trimmed_text = trim_redundant_prefix(references, text)
            if not trimmed_text:
                kept_texts = []
                combined = ""
                continue

            combined = trimmed_text
            kept_texts = [combined]
            continue

        combined = f"{combined}\n\n{text}"
        kept_texts.append(text)

    if combined:
        merged_outputs.append(combined.strip())

    final_outputs: List[str] = []
    for item in merged_outputs:
        final_outputs.extend(split_text_by_length(item, max_length=split_max_length, min_length=split_min_length))

    return final_outputs


def filter_similar_outputs(texts: Sequence[str], similarity_threshold: float = 0.97) -> List[str]:
    """
    对最终输出进行余弦相似度去重。
    输入假定已按长度降序排序。
    """
    kept_texts: List[str] = []
    kept_embeddings: Optional[np.ndarray] = None

    for text in texts:
        if not text:
            continue
        if not kept_texts:
            kept_texts.append(text)
            kept_embeddings = normalize_embeddings(
                np.array([get_response_embed_blocking(text)], dtype=np.float32)
            )
            continue

        current_embedding = normalize_embeddings(
            np.array([get_response_embed_blocking(text)], dtype=np.float32)
        )

        assert kept_embeddings is not None
        similarities = np.dot(current_embedding, kept_embeddings.T).flatten()
        if float(np.max(similarities)) >= similarity_threshold:
            continue

        kept_texts.append(text)
        kept_embeddings = np.vstack([kept_embeddings, current_embedding])

    return kept_texts


def save_jsonl(records: Iterable[Dict], path: Path) -> None:
    """将字典列表写入 JSON Lines 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def save_texts(texts: Dict[str, str], output_dir: Path) -> None:
    """将文本写入指定目录，每条记录一个文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in texts.items():
        (output_dir / f"{key}.txt").write_text(value, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="模型嵌入与聚类处理脚本")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="去重后 chunk 所在目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="模型处理输出目录",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.92,
        help="余弦相似度聚类阈值，默认 0.92",
    )
    parser.add_argument(
        "--cluster-append-sim-threshold",
        type=float,
        default=0.88,
        help="簇内附加文本时的相似度上限，超过则不追加",
    )
    parser.add_argument(
        "--cluster-max-merged-length",
        type=int,
        default=3000,
        help="簇内连续拼接文本的最大长度（超过则开启新段）",
    )
    parser.add_argument(
        "--segment-max-length",
        type=int,
        default=2400,
        help="语义分段后的最大字符数（建议不超过 3000）",
    )
    parser.add_argument(
        "--segment-min-length",
        type=int,
        default=1600,
        help="语义分段时期望的最小字符数（用于保持段落完整性）",
    )
    parser.add_argument(
        "--max-segments-per-cluster",
        type=int,
        default=2,
        help="每个簇输出的最大段落数量，增大可覆盖更多细节",
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=100,
        help="输出片段最大数量（比赛限制为 100）",
    )
    parser.add_argument(
        "--output-similarity-threshold",
        type=float,
        default=0.95,
        help="最终 outputs 去重的余弦相似度阈值",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="嵌入请求的并发度",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    merged_dir = output_dir / "merged_chunks"
    clusters_path = output_dir / "clusters.jsonl"
    outputs_path = output_dir / "outputs.json"

    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 输出目录: {output_dir}")

    chunks = load_chunks(input_dir)
    print(f"[INFO] 载入 {len(chunks)} 个 chunk")

    embeddings = compute_embeddings(chunks, concurrency=args.concurrency)
    print("[INFO] 嵌入生成完成")

    cluster_indices = cluster_by_similarity(embeddings, [c.chunk_id for c in chunks], threshold=args.similarity_threshold)
    print(f"[INFO] 聚类完成，得到 {len(cluster_indices)} 个簇")

    cluster_results: List[ClusterResult] = []
    merged_texts: Dict[str, str] = {}

    for cluster_id, indices in enumerate(cluster_indices):
        member_ids = [chunks[i].chunk_id for i in indices]
        member_texts = [chunks[i].text for i in indices]
        merged_segments = merge_cluster_texts(
            member_texts,
            similarity_threshold=args.cluster_append_sim_threshold,
            max_merged_length=args.cluster_max_merged_length,
            split_max_length=args.segment_max_length,
            split_min_length=args.segment_min_length,
        )
        if not merged_segments:
            continue
        representative_id = member_ids[0] if member_ids else None

        cluster_results.append(
            ClusterResult(
                cluster_id=cluster_id,
                member_ids=member_ids,
                representative_id=representative_id,
                segments=merged_segments,
            )
        )

        for seg_idx, segment in enumerate(merged_segments):
            file_key = f"cluster_{cluster_id:04d}_seg_{seg_idx:02d}"
            merged_texts[file_key] = segment

    save_jsonl((asdict(result) for result in cluster_results), clusters_path)
    save_texts(merged_texts, merged_dir)
    print(f"[INFO] 聚类与合并结果写入 {clusters_path} 与 {merged_dir}")

    # 生成输出列表（每个条目保持 chunk 完整性）
    aggregated_entries: List[str] = []
    sorted_clusters = sorted(
        cluster_results,
        key=lambda r: len(r.segments[0]) if r.segments else 0,
        reverse=True,
    )
    for result in sorted_clusters:
        limited_segments = result.segments[: args.max_segments_per_cluster]
        aggregated_entries.extend(limited_segments)
    aggregated_entries = [entry.strip() for entry in aggregated_entries if entry.strip()]
    aggregated_entries = filter_similar_outputs(
        aggregated_entries,
        similarity_threshold=args.output_similarity_threshold,
    )
    if len(aggregated_entries) > args.max_output:
        aggregated_entries = aggregated_entries[: args.max_output]

    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    with outputs_path.open("w", encoding="utf-8") as f:
        json.dump({"outputs": aggregated_entries}, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 生成 outputs.json，共包含 {len(aggregated_entries)} 个片段")
    print(f"[INFO] 完成。")


if __name__ == "__main__":
    main()


