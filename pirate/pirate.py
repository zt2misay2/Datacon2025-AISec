import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from chat import (
    get_response_embed_concurrent,
    get_response_embed_blocking,
    get_response_RAG_blocking,
    get_response_llm_blocking,
    get_response_llm_concurrent,
)
import re
from collections import defaultdict


class PirateAttack:
    def __init__(
        self,
        beta: int = 1,
        alpha1: float = 0.95,  # 块相似度阈值
        alpha2: float = 0.8,   # 锚点相似度阈值
        alpha3: float = 3,     # 重复块对锚点的惩罚系数
        epsilon_query: float = 0.1,  # 锚点发散噪声水平
        n_anchors: int = 4,    # 每次采样的锚点数
        max_chunks_per_query: int = 5,
        merge_anchor_extraction_requests: int = 2,
        fixed_anchors: List[str] = None,
        keyword: str = None,
    ):

        print("[+]Initializing Pirate Attack...")

        self.fixed_anchors = fixed_anchors if fixed_anchors else []
        self.beta = beta
        self.keyword = keyword
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.epsilon_query = epsilon_query
        self.n_anchors = n_anchors
        self.max_chunks_per_query = max_chunks_per_query
        self.merge_anchor_extraction_requests = merge_anchor_extraction_requests

        # 初始化数据结构
        self.anchors: List[str] = [keyword]  # 锚点列表
        self.relevances: List[float] = []  # 相关性分数
        self.stolen_knowledge: List[str] = []  # 已窃取的知识块

        # 嵌入缓存
        self.chunk_embed_cache: Dict[str, List[float]] = {}   # chunk文本 -> 嵌入向量
        self.anchor_embed_cache: Dict[str, List[float]] = {}  # anchor文本 -> 嵌入向量

        # 注入命令池
        self.injection_commands_path = "./prompt.suffix"

        # 初始化第一个锚点
        self._initialize_first_anchor()

    def _initialize_first_anchor(self):
        """初始化第一个锚点"""
        print("[+]Initializing the first anchor...")

        initial_anchor = self.fixed_anchors[0] if self.fixed_anchors else self.keyword
        self.anchors = [initial_anchor]
        self.relevances = [self.beta]

        # 获取初始锚点的嵌入并缓存
        self._get_anchor_embeddings_batch([initial_anchor])

    def _get_chunk_embedding(self, chunk: str) -> List[float]:
        """获取chunk的嵌入（使用缓存）"""
        if chunk not in self.chunk_embed_cache:
            print("[-] Report a cache miss for chunk embedding.")
            self.chunk_embed_cache[chunk] = get_response_embed_blocking(chunk)
        return self.chunk_embed_cache[chunk]

    def _get_anchor_embedding(self, anchor: str) -> List[float]:
        """获取anchor的嵌入（使用缓存）"""
        if anchor not in self.anchor_embed_cache:
            print("[-] Report a cache miss for anchor embedding.")
            self.anchor_embed_cache[anchor] = get_response_embed_blocking(anchor)
        return self.anchor_embed_cache[anchor]

    def _get_chunk_embeddings_batch(self, chunks: List[str]) -> List[List[float]]:
        """批量获取chunk嵌入"""
        chunks_to_query = [chunk for chunk in chunks if chunk not in self.chunk_embed_cache]

        if chunks_to_query:
            embeddings_array = get_response_embed_concurrent(
                chunks_to_query, get_response_embed_blocking, concurrency=100
            )

            for chunk, embedding in zip(chunks_to_query, embeddings_array):
                self.chunk_embed_cache[chunk] = (
                    embedding.tolist() if hasattr(embedding, "tolist") else embedding
                )

        return [self.chunk_embed_cache[chunk] for chunk in chunks]

    def _get_anchor_embeddings_batch(self, anchors: List[str]) -> List[List[float]]:
        """批量获取anchor嵌入"""
        anchors_to_query = [anchor for anchor in anchors if anchor not in self.anchor_embed_cache]

        if anchors_to_query:
            embeddings_array = get_response_embed_concurrent(
                anchors_to_query, get_response_embed_blocking, concurrency=100
            )

            for anchor, embedding in zip(anchors_to_query, embeddings_array):
                self.anchor_embed_cache[anchor] = (
                    embedding.tolist() if hasattr(embedding, "tolist") else embedding
                )

        return [self.anchor_embed_cache[anchor] for anchor in anchors]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return sim

    def sample_anchors(self) -> List[str]:
        """根据相关性分数采样锚点"""
        if not self.anchors:
            return []

        print(f"[+]Sampling {self.n_anchors} anchors based on relevances...")

        relevances = np.array(self.relevances)
        exp_relevances = np.exp(relevances - np.max(relevances))
        probabilities = exp_relevances / np.sum(exp_relevances)

        sampled_indices = np.random.choice(
            len(self.anchors),
            size=self.n_anchors,
            p=probabilities,
            replace=True,
        )

        samples = [self.anchors[i] for i in sampled_indices]

        # 从fixed_anchors中随机选择一个锚点进行补充
        if self.fixed_anchors:
            fixed_anchor = random.choice(self.fixed_anchors)
            if fixed_anchor not in [self.anchors[i] for i in sampled_indices]:
                samples.append(fixed_anchor)

        print(f"[+]Sampled anchors: {samples}")
        return samples

    def generate_base_query(self, anchors: List[str]) -> str:
        """使用攻击侧LLM基于锚点生成基础查询"""
        print("[+]Generating base query from sampled anchors...")

        anchor_text = ", ".join(anchors)

        prompt = f"""
        Generate a natural question or query that incorporates the following topics: {anchor_text}.
        The query should be coherent and seem like a legitimate user question.
        Return only the query text without any additional explanation.
        """

        response = get_response_llm_blocking(prompt)
        return response.strip()

    def extract_anchors_from_text(self, text: str) -> List[str]:
        """从文本中提取锚点（统一逻辑，包含一定发散）"""
        print("[+]Extracting anchors from stolen chunk...")

        prompt = f"""
You are generating exploratory search anchors.

1. 从下面文本中提取 3-5 个**核心主题**（必须在文本中出现或非常明显）。
2. 再额外给出 2-3 个**联想/相邻/更泛化的主题**，可以不直接出现在文本中，
   但要和文本内容有明显语义关联，用来拓展范围。

要求:
- 每个锚点 1-3 个词。
- 使用与原文本相同的语言。
- 用英文逗号 ',' 分隔。
- 输出形式必须是一个 Python 列表，例如：
  ["网络安全", "入侵检测", "威胁情报"]
- 不要任何额外解释。

文本:
{text}

锚点列表:
"""

        # 用 epsilon_query 控制联想有多大胆
        if np.random.uniform(0, 1) < self.epsilon_query:
            prompt += """
在联想主题时，可以适当跨一级抽象（例如从“SQL 注入”联想到“Web 安全”、“数据库安全”）。
"""

        response = get_response_llm_blocking(prompt).strip()

        import ast
        anchors: List[str] = []
        try:
            anchors_list = ast.literal_eval(response)
            if isinstance(anchors_list, list):
                anchors = [
                    a.strip()
                    for a in anchors_list
                    if isinstance(a, str) and a.strip()
                ]
        except Exception:
            anchors = [anchor.strip() for anchor in response.split(",") if anchor.strip()]

        return anchors[:8]

    def generate_exploration_anchors(self, anchors: List[str]) -> List[Tuple[str, List[float]]]:
        """在没有新 chunk 时，基于现有锚点生成一批发散锚点"""
        print("[+]Generating exploration anchors from duplicate chunks...")

        if not anchors:
            anchors = self.anchors[: min(5, len(self.anchors))]

        topic_text = ", ".join(anchors)

        prompt = f"""
You are expanding search topics.

Given these topics: {topic_text}

Generate 6-10 NEW topics that:
- are related but NOT synonyms or trivial variations of the given ones,
- tend to broaden or shift the focus (e.g. from "SQL 注入" to "Web 安全", "数据库审计"),
- could lead a retrieval system to different documents.

Requirements:
- Each topic is 1-3 words.
- Use the same language as the input topics.
- Return a Python list of strings, for example:
  ["网络攻防", "安全合规", "日志审计"]
- No extra explanation.
"""

        response = get_response_llm_blocking(prompt).strip()

        import ast
        try:
            topic_list = ast.literal_eval(response)
            if not isinstance(topic_list, list):
                raise ValueError
            candidates = [
                t.strip()
                for t in topic_list
                if isinstance(t, str) and t.strip()
            ]
        except Exception:
            candidates = [t.strip() for t in response.split(",") if t.strip()]

        if not candidates:
            return []

        unique_new_anchors = list(set(candidates))
        new_anchor_embeddings = self._get_anchor_embeddings_batch(unique_new_anchors)

        if self.anchors:
            existing_anchor_embeddings = self._get_anchor_embeddings_batch(self.anchors)
            existing_matrix = np.array(existing_anchor_embeddings)
            new_matrix = np.array(new_anchor_embeddings)

            existing_norm = existing_matrix / np.linalg.norm(
                existing_matrix, axis=1, keepdims=True
            )
            new_norm = new_matrix / np.linalg.norm(new_matrix, axis=1, keepdims=True)

            sim_matrix = np.dot(new_norm, existing_norm.T)
            is_anchor_duplicate = np.max(sim_matrix, axis=1) >= self.alpha2

            results: List[Tuple[str, List[float]]] = []
            for anchor, embedding, duplicate in zip(
                unique_new_anchors, new_anchor_embeddings, is_anchor_duplicate
            ):
                if not duplicate:
                    results.append((anchor, embedding))
        else:
            results = list(zip(unique_new_anchors, new_anchor_embeddings))

        print(f"[+]Exploration anchors generated: {len(results)}")
        return results

    def is_duplicate_chunk(self, chunk: str, threshold: float = None) -> bool:
        """检查块是否为重复内容"""
        if threshold is None:
            threshold = self.alpha1

        if not self.stolen_knowledge:
            return False

        existing_embeddings = self._get_chunk_embeddings_batch(self.stolen_knowledge)
        chunk_embedding = self._get_chunk_embedding(chunk)

        for existing_embedding in existing_embeddings:
            similarity = self.cosine_similarity(chunk_embedding, existing_embedding)
            if similarity >= threshold:
                return True

        return False

    def is_duplicate_anchor(self, anchor: str, threshold: float = None) -> bool:
        """检查锚点是否为重复内容"""
        if threshold is None:
            threshold = self.alpha2

        if not self.anchors:
            return False

        existing_embeddings = self._get_anchor_embeddings_batch(self.anchors)
        anchor_embedding = self._get_anchor_embedding(anchor)

        for existing_embedding in existing_embeddings:
            similarity = self.cosine_similarity(anchor_embedding, existing_embedding)
            if similarity >= threshold:
                return True

        return False

    def parse_chunks_from_response(self, response: str) -> List[str]:
        """从RAG系统响应中解析出知识块"""
        chunks: List[str] = []

        pattern = r"#reference_imformation_begin#(.*?)#reference_imformation_end#"
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            chunk = match.strip()
            if chunk:
                chunks.append(chunk)

        return chunks[:self.max_chunks_per_query]

    def compute_penalties(self, duplicate_chunks: List[str]) -> List[float]:
        """计算惩罚项"""
        if not duplicate_chunks or not self.anchors:
            return [0.0] * len(self.anchors)

        penalties = [0.0] * len(self.anchors)

        anchor_embeddings = self._get_anchor_embeddings_batch(self.anchors)

        for chunk in duplicate_chunks:
            chunk_embedding = self._get_chunk_embedding(chunk)

            similarities = []
            for anchor_embedding in anchor_embeddings:
                similarity = self.cosine_similarity(chunk_embedding, anchor_embedding)
                similarities.append(similarity)

            exp_similarities = np.exp(similarities - np.max(similarities))
            softmax_probs = exp_similarities / np.sum(exp_similarities)

            for i in range(len(penalties)):
                penalties[i] += self.alpha3 * softmax_probs[i]

        penalties = [p / len(duplicate_chunks) for p in penalties]
        return penalties

    def update_relevances(
        self,
        new_anchors: List[Tuple[str, List[float]]],
        duplicate_chunks: List[str],
    ):
        """更新相关性分数"""
        penalties = self.compute_penalties(duplicate_chunks)

        for i in range(len(self.anchors)):
            if self.anchors[i] in [anchor for anchor, _ in new_anchors]:
                continue
            elif any(self.anchors[i] in chunk for chunk in duplicate_chunks):
                self.relevances[i] = max(0, self.relevances[i] - penalties[i])

        for anchor, embedding in new_anchors:
            if not self.is_duplicate_anchor(anchor, self.alpha2):
                self.anchors.append(anchor)
                self.anchor_embed_cache[anchor] = embedding
                max_relevance = max(self.relevances) if self.relevances else self.beta
                self.relevances.append(max_relevance)

    def execute_attack(self, max_iterations: int = 1000):
        """执行攻击算法"""
        iteration = 0
        chunk_cnt = 0
        while iteration < max_iterations and max(self.relevances) > 0:
            st = time.time()
            print(
                f"[+]Iteration {iteration + 1}, Active anchors: {len([r for r in self.relevances if r > 0])}"
            )
            print(f"Anchors: {self.anchors}")

            # 1. 采样锚点
            print("[+]Sampling anchors...")
            sampled_anchors = self.sample_anchors()
            if not sampled_anchors:
                break

            # 2. 生成基础查询
            print("[+]Generating base query...")
            base_query = self.generate_base_query(sampled_anchors)

            # 3. 尝试不同的注入命令
            stolen_chunks: List[str] = []
            with open(self.injection_commands_path, "r", encoding="utf-8") as f:
                injection_commands = f.read()
            print(f"find injection command {injection_commands}\n")
            for i, command in enumerate([injection_commands]):
                attack_query = f"{base_query} {command}"

                try:
                    print(f"[+]Sending attack query with injection command {i + 1}...")
                    print(f"Attack Query: {attack_query}")

                    response = get_response_RAG_blocking(attack_query)
                    while response is None:
                        print("Error happened, retrying...")
                        response = get_response_RAG_blocking(attack_query)
                    print(f"response: {response}")

                    print("[+]Parsing response for stolen chunks...")
                    chunks = self.parse_chunks_from_response(response)

                    if chunks:
                        stolen_chunks.extend(chunks)
                        print(f"[+]Retrieved {len(chunks)} chunks from response.")
                        print(f"Sample chunk: {chunks}")
                        break

                except Exception as e:
                    print(f"Error in attack query: {e}")
                    continue

            if not stolen_chunks:
                iteration += 1
                continue

            # 6. 批量去重处理
            print("[+]Batch processing stolen chunks for duplicates...")

            stolen_embeddings = self._get_chunk_embeddings_batch(stolen_chunks)

            if self.stolen_knowledge:
                knowledge_embeddings = self._get_chunk_embeddings_batch(self.stolen_knowledge)

                stolen_matrix = np.array(stolen_embeddings)
                knowledge_matrix = np.array(knowledge_embeddings)

                stolen_norm = stolen_matrix / np.linalg.norm(stolen_matrix, axis=1, keepdims=True)
                knowledge_norm = knowledge_matrix / np.linalg.norm(
                    knowledge_matrix, axis=1, keepdims=True
                )

                similarity_matrix = np.dot(stolen_norm, knowledge_norm.T)

                is_duplicate = np.max(similarity_matrix, axis=1) >= self.alpha1

                non_duplicate_chunks = [
                    chunk for chunk, duplicate in zip(stolen_chunks, is_duplicate) if not duplicate
                ]
                duplicate_chunks = [
                    chunk for chunk, duplicate in zip(stolen_chunks, is_duplicate) if duplicate
                ]
            else:
                non_duplicate_chunks = stolen_chunks
                duplicate_chunks = []

            # 6.1 如果本轮只有重复 chunk，则强制发散生成 exploration anchors
            if not non_duplicate_chunks:
                print("[+]Only duplicate chunks this round; generating exploration anchors...")

                exploration_anchors = self.generate_exploration_anchors(sampled_anchors)
                print(f"[+]Generated {len(exploration_anchors)} exploration anchors from duplicates.")

                self.update_relevances(exploration_anchors, duplicate_chunks)

                ed = time.time()
                print(
                    f"\n\n[+]Iteration {iteration + 1} (duplicate-only) completed in {ed - st:.2f} seconds.\n"
                )

                iteration += 1
                continue

            # 7. 正常情况：有新 chunk
            self.stolen_knowledge.extend(non_duplicate_chunks)
            print(f"[+]Added {len(non_duplicate_chunks)} new chunks, {len(duplicate_chunks)} duplicates")

            for c in non_duplicate_chunks:
                chunk_cnt += 1
                print(f"writing chunk {chunk_cnt} into file: {c}")
                with open(f"chunk_{chunk_cnt}.txt", "w", encoding="utf-8") as fd:
                    fd.write(c)

            # 7.1 从新块中批量提取锚点（统一走 extract_anchors_from_text）
            print("[+]Batch extracting and adding new anchors...")

            new_anchors_with_embeddings: List[Tuple[str, List[float]]] = []
            if non_duplicate_chunks:
                all_extracted_anchors: List[str] = []

                for chunk in non_duplicate_chunks:
                    anchors = self.extract_anchors_from_text(chunk)
                    all_extracted_anchors.extend(anchors)

                unique_new_anchors = list(set(all_extracted_anchors))
                if unique_new_anchors:
                    new_anchor_embeddings = self._get_anchor_embeddings_batch(unique_new_anchors)

                    if self.anchors:
                        existing_anchor_embeddings = self._get_anchor_embeddings_batch(self.anchors)
                        existing_matrix = np.array(existing_anchor_embeddings)
                        new_matrix = np.array(new_anchor_embeddings)

                        existing_norm = existing_matrix / np.linalg.norm(
                            existing_matrix, axis=1, keepdims=True
                        )
                        new_norm = new_matrix / np.linalg.norm(new_matrix, axis=1, keepdims=True)

                        anchor_similarity_matrix = np.dot(new_norm, existing_norm.T)
                        is_anchor_duplicate = (
                            np.max(anchor_similarity_matrix, axis=1) >= self.alpha2
                        )

                        for anchor, embedding, duplicate in zip(
                            unique_new_anchors, new_anchor_embeddings, is_anchor_duplicate
                        ):
                            if not duplicate:
                                new_anchors_with_embeddings.append((anchor, embedding))
                    else:
                        for anchor, embedding in zip(unique_new_anchors, new_anchor_embeddings):
                            new_anchors_with_embeddings.append((anchor, embedding))

            # 8. 更新相关性分数
            print("[+]Updating relevances...")
            self.update_relevances(new_anchors_with_embeddings, duplicate_chunks)

            print(
                f"Stolen {len(non_duplicate_chunks)} new chunks, Total: {len(self.stolen_knowledge)}"
            )
            print(f"Total anchors: {len(self.anchors)}")

            ed = time.time()
            print(f"\n\n[+]Iteration {iteration + 1} completed in {ed - st:.2f} seconds.\n")

            iteration += 1

        print(f"Attack completed after {iteration} iterations")
        print(f"Total stolen knowledge chunks: {len(self.stolen_knowledge)}")

    def get_results(self) -> Dict:
        """获取攻击结果"""
        return {
            "stolen_knowledge": self.stolen_knowledge,
            "total_chunks": len(self.stolen_knowledge),
            "final_anchors": self.anchors,
            "final_relevances": self.relevances,
        }


if __name__ == "__main__":
    attacker = PirateAttack(
        beta=1,
        alpha1=0.95,
        alpha2=0.85,
        alpha3=4,
        n_anchors=3,
        max_chunks_per_query=5,   # 非结构化数据
        epsilon_query=0.6,        # 60% 概率更激进发散
        keyword="网络安全",
    )

    attacker.execute_attack(max_iterations=170)

    results = attacker.get_results()
    print(f"Attack completed. Stolen {results['total_chunks']} chunks.")

    print("Final anchors and relevances:")
    for anchor, relevance in zip(results["final_anchors"], results["final_relevances"]):
        print(f"Anchor: {anchor}, Relevance: {relevance}")

    print("Saving stolen knowledge to 'stolen_knowledge.txt'...")
    with open("stolen_knowledge.txt", "w", encoding="utf-8") as f:
        for chunk in results["stolen_knowledge"]:
            f.write(chunk + "\n\n")

    print("Stolen knowledge saved.")
