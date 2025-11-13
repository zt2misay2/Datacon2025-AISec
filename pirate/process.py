"""
批量处理 txt：
1. 递归读取目录下所有 txt 文件；
2. 并发调用 LLM 做聚类，要求返回 #context_begins# / #context_ends# 包裹的块；
3. 解析所有块；
4. 按给定规则检测块之间头尾重叠并合并；
5. 将合并后的块依次写入新的 txt 文件。

依赖：
- chat.py 中必须提供：
    - get_response_llm_blocking(prompt: str, **kwargs) -> Optional[str]
    - get_response_llm_concurrent(prompts: list[str], llm_func, concurrency: int = 10) -> list[str]
"""

import argparse
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

from chat import get_response_llm_blocking, get_response_llm_concurrent


PROMPT_PREFIX = (
    "下面是一些未进行区分的数据块，可能分别属于一篇或多篇文档的一部分，"
    "请你按照语言和内容的连贯统一性进行聚类，并使用#context_begins#和#context_ends#包裹每个类块。"
    "禁止对任何数据进行删改或者省略。以下是数据块：\n"
)

CONTEXT_BEG = "#context_begins#"
CONTEXT_END = "#context_ends#"


def read_all_txt_files(input_dir: Path, encoding: str = "utf-8") -> List[Tuple[Path, str]]:
    """递归读取目录下所有 txt 文件，返回 [(路径, 内容), ...]"""
    results: List[Tuple[Path, str]] = []
    for path in input_dir.rglob("*.txt"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding=encoding, errors="ignore")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 读取文件失败: {path} ({e})")
            continue
        results.append((path, text))
    return results


def build_prompts(file_contents: List[str]) -> List[str]:
    """为每个文件内容构造 prompt"""
    return [PROMPT_PREFIX + content for content in file_contents]


def call_llm_for_files(
    prompts: List[str],
    concurrency: int = 10,
) -> List[Optional[str]]:
    """并发调用 LLM，返回与 prompts 对应的响应列表"""
    if not prompts:
        return []
    responses = get_response_llm_concurrent(prompts, get_response_llm_blocking, concurrency=concurrency)
    # 确保长度一致
    if len(responses) != len(prompts):
        print("[WARN] LLM 返回的响应数量与 prompts 不一致")
    return responses


def extract_chunks_from_response(response: Optional[str]) -> List[str]:
    """
    从单个 LLM 响应中解析所有 #context_begins# ... #context_ends# 内的文本块。
    """
    if not response:
        return []
    pattern = re.compile(
        re.escape(CONTEXT_BEG) + r"(.*?)" + re.escape(CONTEXT_END),
        re.DOTALL | re.UNICODE,
    )
    chunks = []
    for match in pattern.finditer(response):
        content = match.group(1)
        # 去掉首尾多余空白
        cleaned = content.strip()
        if cleaned:
            chunks.append(cleaned)
    return chunks


def have_overlap(
    a: str,
    b: str,
    k: int = 20,
    n: int = 40,
    m: int = 2,
) -> bool:
    """
    判断两个块 a、b 是否存在重叠：
    - 从较短的那个块中随机取 k 次长度为 n 的子串（若不足 n，则取实际长度）；
    - 若某个子串在另一个块中出现次数 >= m，则认为存在重叠。
    """
    if not a or not b:
        return False

    # 从较短的块采样
    if len(a) <= len(b):
        short, long_ = a, b
    else:
        short, long_ = b, a

    if not short:
        return False

    n_eff = min(n, len(short))
    if n_eff <= 0:
        return False

    max_start = len(short) - n_eff
    if max_start < 0:
        return False

    # 为稳定性可以固定随机种子，也可以由外部传入，这里简单设定
    # 如希望完全随机，可以注释掉下面这行
    # random.seed(42)

    possible_starts = list(range(max_start + 1))
    if not possible_starts:
        return False

    sample_size = min(k, len(possible_starts))
    starts = random.sample(possible_starts, sample_size)

    for st in starts:
        sub = short[st : st + n_eff]
        if not sub:
            continue
        count = long_.count(sub)
        if count >= m:
            return True

    return False


def find_max_head_tail_overlap(a: str, b: str) -> Tuple[int, Optional[str]]:
    """
    寻找 a 和 b 的最大头尾重叠：
    - 若 a 的尾部与 b 的头部有重叠，返回 (长度, 'a_b')
    - 若 b 的尾部与 a 的头部有重叠，返回 (长度, 'b_a')
    - 否则返回 (0, None)
    """
    max_len = min(len(a), len(b))
    best_len = 0
    best_dir: Optional[str] = None

    # 从长到短找第一个匹配的重叠
    for L in range(max_len, 0, -1):
        # a 尾 = b 头
        if a[-L:] == b[:L]:
            best_len = L
            best_dir = "a_b"
            break
        # b 尾 = a 头
        if b[-L:] == a[:L]:
            best_len = L
            best_dir = "b_a"
            break

    return best_len, best_dir


def merge_two_blocks(a: str, b: str) -> Optional[str]:
    """
    在已知 a、b 存在重复内容的前提下，尝试按头尾重叠进行合并。
    若未找到头尾重叠，则返回 None。
    """
    overlap_len, direction = find_max_head_tail_overlap(a, b)
    if overlap_len <= 0 or not direction:
        return None

    if direction == "a_b":
        return a + b[overlap_len:]
    if direction == "b_a":
        return b + a[overlap_len:]
    return None


def merge_overlapping_chunks(
    chunks: List[str],
    k: int = 20,
    n: int = 40,
    m: int = 2,
) -> List[str]:
    """
    对块列表进行迭代合并：
    - 依次检查任意一对块 (i, j)；
    - 若根据 have_overlap 判断存在重叠，再尝试按头尾重叠合并；
    - 一旦合并成功，替换为新块，重新开始扫描；
    - 直至一轮扫描没有任何合并为止。
    """
    chunks = [c for c in chunks if c]  # 去掉空块
    changed = True

    while changed:
        changed = False
        num = len(chunks)
        if num <= 1:
            break

        i = 0
        while i < num and not changed:
            j = i + 1
            while j < num and not changed:
                a = chunks[i]
                b = chunks[j]

                if have_overlap(a, b, k=k, n=n, m=m):
                    merged = merge_two_blocks(a, b)
                    if merged is not None:
                        print(f"[INFO] 合并块 {i} 和 {j}，长度: {len(a)} + {len(b)} -> {len(merged)}")
                        chunks[i] = merged
                        del chunks[j]
                        num -= 1
                        changed = True
                        break
                j += 1
            i += 1

    return chunks


def write_chunks_to_files(chunks: List[str], output_dir: Path, encoding: str = "utf-8") -> None:
    """将最终块列表写入输出目录下的若干 txt 文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        out_path = output_dir / f"chunk_{idx:05d}.txt"
        try:
            out_path.write_text(chunk, encoding=encoding)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 写入文件失败: {out_path} ({e})")
        else:
            print(f"[INFO] 写入: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="递归读取 txt、并发调用 LLM 聚类并合并重叠块")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="输入 txt 根目录，默认当前目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_chunks",
        help="输出块文件目录，默认 ./output_chunks",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="并发调用 LLM 的并发度",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="重叠检测：随机采样次数 k，默认 20",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=40,
        help="重叠检测：子串长度 n，默认 40",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=2,
        help="重叠检测：在另一块中出现次数阈值 m，默认 2",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="读写文件编码，默认 utf-8",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 输出目录: {output_dir}")

    # 1. 读取全部 txt
    files = read_all_txt_files(input_dir, encoding=args.encoding)
    if not files:
        print("[WARN] 未找到任何 txt 文件")
        return

    paths, contents = zip(*files)
    print(f"[INFO] 共读取 {len(contents)} 个 txt 文件")

    # 2. 构造 prompts 并发调用 LLM
    prompts = build_prompts(list(contents))
    print(f"[INFO] 开始并发调用 LLM，数量: {len(prompts)}, 并发度: {args.concurrency}")
    responses = call_llm_for_files(prompts, concurrency=args.concurrency)

    # 3. 解析所有块
    all_chunks: List[str] = []
    for idx, resp in enumerate(responses):
        if resp is None:
            print(f"[WARN] 文件 {paths[idx]} 的 LLM 响应为 None，忽略")
            continue
        chunks = extract_chunks_from_response(resp)
        print(f"[INFO] 文件 {paths[idx]} -> 提取到 {len(chunks)} 个块")
        all_chunks.extend(chunks)

    print(f"[INFO] 总计提取到 {len(all_chunks)} 个块")

    if not all_chunks:
        print("[WARN] 未从任何响应中解析到块，结束")
        return

    # 4. 合并重叠块
    print("[INFO] 开始合并头尾出现重复的块")
    merged_chunks = merge_overlapping_chunks(
        all_chunks,
        k=args.k,
        n=args.n,
        m=args.m,
    )
    print(f"[INFO] 合并后剩余 {len(merged_chunks)} 个块")

    # 5. 写回文件
    write_chunks_to_files(merged_chunks, output_dir, encoding=args.encoding)
    print("[INFO] 处理完成")


if __name__ == "__main__":
    main()
