
import requests
import json
import toml as tomllib  # 解析toml文件（Python 3.11+内置，低版本需安装toml库：pip install toml）
import os
from typing import Dict, Optional, List, Any, Tuple
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed # 导入并发库
from tqdm import tqdm # 导入进度条库
import numpy as np

#!/usr/bin/env python3
"""
RAG 防御系统客户端示例
演示如何与本地 RAG 服务进行交互

使用方法:
    python3 demo.py
"""

import requests
import json
import time
from typing import Dict, List, Optional


class RAGClient:
    """RAG 服务客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端

        参数:
            base_url: RAG API 服务地址
        """
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> bool:
        """
        检查服务健康状态

        返回:
            True 表示服务正常，False 表示服务异常
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False

    def query(
        self,
        question: str,
        application_id: str,  # 注意：请填入对应题号的应用 ID，否则会返回错误信息
        timeout: int = 180
    ) -> Dict:
        """
        发送查询请求

        参数:
            question: 要询问的问题
            application_id: 应用 ID（默认为 "default"）
            timeout: 请求超时时间（秒）

        返回:
            包含响应数据的字典
        """
        endpoint = f"{self.base_url}/query"
        payload = {
            "application_id": application_id,
            "query": question
        }

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # 处理 HTTP 错误
            try:
                error_detail = response.json()
                return {
                    "error": f"HTTP {response.status_code}",
                    "detail": error_detail,
                    "query": question
                }
            except:
                return {
                    "error": str(e),
                    "query": question
                }
        except Exception as e:
            return {
                "error": str(e),
                "query": question
            }

    def batch_query(
        self,
        questions: List[str],
        application_id: str ,   # 注意：请填入对应题号的应用 ID，否则会返回错误信息
        show_progress: bool = True
    ) -> List[Dict]:
        """
        批量发送查询请求

        参数:
            questions: 问题列表
            show_progress: 是否显示进度

        返回:
            响应列表
        """
        results = []
        total = len(questions)

        for idx, question in enumerate(questions, 1):
            if show_progress:
                print(f"处理中 {idx}/{total}: {question[:50]}...")

            result = self.query(question, application_id)
            results.append(result)

            if show_progress and "error" not in result:
                output_preview = result.get('output', '')[:80]
                print(f"  ✓ 回答: {output_preview}...")

            # 避免请求过快
            time.sleep(0.5)

        return results


def print_separator(char="=", length=70):
    """打印分隔线"""
    print(char * length)


def chat_once(question: str):
    """客户端使用示例"""
    client = RAGClient(base_url="http://localhost:9000")

    # ========================================================================
    # 示例 1: 单次查询
    # ========================================================================
    print("\n示例 1: 单次查询")
    print("-" * 70)
    print(f"问题: {question}")

    result = client.query(  question=question,
                            application_id="64e2355f-aab3-4f84-9f58-1680882666a7" )

    if "error" in result:
        print(f"错误: {result['error']}\n Retrying...\n")
        import time
        time.sleep(5)
        return chat_once(question)
        if "detail" in result:
            print(f"详情: {json.dumps(result['detail'], indent=2, ensure_ascii=False)}")
    else:
        print(f"\n回答: {result.get('output', '无回答')}")
        print(f"推理时间: {result.get('elapsed_time', 0):.2f} 秒")
        print(f"检测到的意图: {result.get('intent_detected', 'UNKNOWN')}")
        return result.get('output', '无回答'), None

    # ========================================================================
    # 示例 2: 批量
    # ========================================================================
    '''
    print("\n示例 2: 批量查询")
    print("-" * 70)

    questions = [
        "深度学习的主要应用领域有哪些？",
        "什么是神经网络？",
        "解释一下卷积神经网络的工作原理"
    ]

    results = client.batch_query(   questions,
                                    application_id="15ad28a3-7739-4c2d-9776-0f6e10cef094",     # 填入对应题号的 Application ID
                                    show_progress=True)

    print(f"\n✓ 完成 {len(results)} 个查询")
    print(" --- 结果预览 --- ")
    print(results)
    '''

# 辅助函数：加载配置文件
def _load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件（内部辅助函数）"""
    try:
        with open(config_path, 'r') as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise Exception(f"配置文件 {config_path} 未找到")
    except Exception as e:
        raise Exception(f"加载配置文件失败: {str(e)}")


# 辅助函数：从配置中提取硅基API必要信息
def _extract_silicon_info(config: Dict[str, Any]) -> Dict[str, str]:
    """从配置中提取API密钥、URL和模型信息（内部辅助函数）"""
    silicon_config = config.get('silicon', {})
    return {
        'api_key': silicon_config.get('silicon_api_key', ''),
        'llm_url': silicon_config.get('llm_url', ''),
        'embed_url': silicon_config.get('embed_url', ''),
        'model': silicon_config.get('model', ''),
        'embed_model': silicon_config.get('embed_model', '')
    }


# 辅助函数：构建请求头
def _build_headers(api_key: str) -> Dict[str, str]:
    """构建API请求头（内部辅助函数）"""
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }


# 辅助函数：处理LLM请求逻辑
def _handle_llm_request(prompt: str, silicon_info: Dict[str, str], headers: Dict[str, str], **kwargs) -> Optional[str]:
    """处理LLM API请求及响应解析（内部辅助函数）"""
    # 构建请求数据
    data = {
        "model": kwargs.get('model', silicon_info['model']),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    # 添加可选参数
    optional_params = ['temperature', 'max_tokens', 'top_p']
    for param in optional_params:
        if param in kwargs:
            data[param] = kwargs[param]
    
    try:
        response = requests.post(
            silicon_info['llm_url'],
            headers=headers,
            json=data,
            timeout=kwargs.get('timeout', 60)
        )
        response.raise_for_status()
        result = response.json()
        
        # 提取回复内容
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            print(f"API响应格式异常: {result}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {str(e)}")
        return None
    except Exception as e:
        print(f"处理响应时发生错误: {str(e)}")
        return None


# 辅助函数：处理嵌入请求逻辑
def _handle_embed_request(query: str, silicon_info: Dict[str, str], headers: Dict[str, str],** kwargs) -> Optional[List[float]]:
    """处理嵌入API请求及响应解析（内部辅助函数）"""
    # 构建请求数据
    data = {
        "model": kwargs.get('model', silicon_info['embed_model']),
        "input": query
    }
    
    try:
        response = requests.post(
            silicon_info['embed_url'],
            headers=headers,
            json=data,
            timeout=kwargs.get('timeout', 30)
        )
        response.raise_for_status()
        result = response.json()
        
        # 提取嵌入向量
        if 'data' in result and len(result['data']) > 0:
            return result['data'][0]['embedding']
        else:
            print(f"API响应格式异常: {result}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {str(e)}")
        return None
    except Exception as e:
        print(f"处理响应时发生错误: {str(e)}")
        return None

def get_response_llm_blocking(prompt: str, **kwargs) -> Optional[str]:
    """
    使用硅基流动API获取对话响应的函数
    注意如果请求时出现错误，该函数会自动重试，直到成功为止。
    
    Args:
        prompt: 用户输入的提示词
        **kwargs: 其他可选参数（支持config_path指定配置文件路径）
        
    Returns:
        API返回的文本响应
    """
    config_path = kwargs.pop('config_path', "config.toml")  # 支持通过kwargs指定配置路径
    try:
        config = _load_config(config_path)
        silicon_info = _extract_silicon_info(config)
        headers = _build_headers(silicon_info['api_key'])
        result = _handle_llm_request(prompt, silicon_info, headers,** kwargs)
        while result is None:
            print("请求失败，正在重试...")
            result = _handle_llm_request(prompt, silicon_info, headers,** kwargs)
        return result
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return None

def get_response_embed_blocking(query: str, **kwargs) -> Optional[List[float]]:
    """
    使用硅基流动API获取嵌入向量的函数。
    注意如果请求时出现错误，该函数会自动重试，直到成功为止。
    
    Args:
        query: 需要嵌入的文本
        **kwargs: 其他可选参数（支持config_path指定配置文件路径）
        
    Returns:
        嵌入向量，以列表形式返回
    """
    config_path = kwargs.pop('config_path', "config.toml")  # 支持通过kwargs指定配置路径
    try:
        config = _load_config(config_path)
        silicon_info = _extract_silicon_info(config)
        headers = _build_headers(silicon_info['api_key'])
        result = _handle_embed_request(query, silicon_info, headers,** kwargs)
        while result is None:
            print("请求失败，正在重试...")
            result = _handle_embed_request(query, silicon_info, headers,** kwargs)
        return result
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return None

def get_response_embed_concurrent(chunks: list[str], embed_func, concurrency: int = 50) -> np.ndarray:
    """
    为每个文本块并发生成嵌入向量，并显示进度条。
    """
    if not chunks:
        return np.array([])
        
    print(f"  正在为 {len(chunks)} 个文本块生成嵌入向量 (并发数: {concurrency})...")
    
    # 创建一个与 chunks 等长的列表，用于按原始顺序存放结果
    embeddings = [None] * len(chunks)
    
    # 使用 ThreadPoolExecutor 进行并发处理
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 提交所有任务，并创建一个字典将 future 对象映射到其原始索引
        future_to_index = {executor.submit(embed_func, chunk): i for i, chunk in enumerate(chunks)}
        
        # 使用 tqdm 创建一个进度条，as_completed 会在任务完成时立即返回 future
        for future in tqdm(as_completed(future_to_index), total=len(chunks), desc="  生成嵌入"):
            index = future_to_index[future]
            try:
                # 获取任务结果
                vec_list = future.result()
                # 将结果存放到正确的位置
                embeddings[index] = vec_list
            except Exception as e:
                # 如果某个请求失败，打印错误并用零向量填充，避免整个过程崩溃
                print(f"\n错误: 处理块索引 {index} (内容: '{chunks[index][:30]}...') 时发生异常: {e}")
                embeddings[index] = [0.0] * 4096 # 维度是4096

    # 将 list of lists 转换为一个 2D numpy 数组
    return np.array(embeddings, dtype=np.float32)

def get_response_llm_concurrent(prompts: list[str], llm_func, concurrency: int = 10) -> list[str]:
    """
    为每个提示词并发调用LLM函数，并显示进度条。
    
    Args:
        prompts: 提示词列表
        llm_func: LLM调用函数，接受字符串参数并返回字符串
        concurrency: 并发数，根据API限制调整
        
    Returns:
        与输入顺序对应的响应列表
    """
    if not prompts:
        return []
        
    print(f"  正在为 {len(prompts)} 个提示词调用LLM (并发数: {concurrency})...")
    
    # 创建一个与 prompts 等长的列表，用于按原始顺序存放结果
    responses = [None] * len(prompts)
    
    # 使用 ThreadPoolExecutor 进行并发处理
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 提交所有任务，并创建一个字典将 future 对象映射到其原始索引
        future_to_index = {executor.submit(llm_func, prompt): i for i, prompt in enumerate(prompts)}
        
        # 使用 tqdm 创建一个进度条
        for future in tqdm(as_completed(future_to_index), total=len(prompts), desc="  LLM调用"):
            index = future_to_index[future]
            try:
                # 获取任务结果
                response = future.result()
                # 将结果存放到正确的位置
                responses[index] = response
            except Exception as e:
                # 如果某个请求失败，打印错误并用默认值填充
                print(f"\n错误: 处理提示词索引 {index} (内容: '{prompts[index][:30]}...') 时发生异常: {e}")
                responses[index] = f"错误: {str(e)}"  # 用错误信息作为响应

    return responses

def get_response_RAG_blocking(prompt: str) -> Optional[str]:
    """使用 Dify 的 RAG 框架进行阻塞式对话的函数。也就是 dify_stream_turn 的封装版。
    
    和其他函数一致，如果出现错误会自动重试，直到成功为止。

    Args:
        prompt (str): 用户输入的提示词

    Returns:
        answer(str): 返回的结果
    """
    answer, error = chat_once(prompt)
    return answer

def get_response_RAG_concurrent(prompts: list[str], concurrency: int = 10) -> list[str]:
    """
    为每个提示词并发调用 Dify RAG 框架，并显示进度条。
    
    Args:
        prompts: 提示词列表
        concurrency: 并发数，根据API限制调整
        
    Returns:
        与输入顺序对应的响应列表
    """
    return get_response_llm_concurrent(prompts, get_response_RAG_blocking, concurrency=concurrency)
