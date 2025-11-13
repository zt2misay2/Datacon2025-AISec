import requests
import json
from typing import Optional

import requests
import json
import sys
import io
from typing import Optional

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def siliconflow_stream_chat(
    prompt: str,
    api_key: str,
    model: str = "deepseek-ai/DeepSeek-V3.2-Exp",
    base_url: str = "https://api.siliconflow.cn/v1/chat/completions",
    print_stream: bool = True,
    timeout: int = 600,
) -> str:
    """
    使用硅基流动 chat.completions 的流式接口
    """
    
    session = requests.Session()
    session.trust_env = False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    resp = session.post(
        base_url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()
    resp.encoding = 'utf-8'  # 显式设置编码

    full_text = ""
    
    # 使用更健壮的方式读取流数据
    for line in resp.iter_lines(decode_unicode=False):  # 先不解码
        if not line:
            continue
            
        # 手动解码为UTF-8
        try:
            line_decoded = line.decode('utf-8').strip()
        except UnicodeDecodeError:
            continue
            
        if line_decoded.startswith("data:"):
            data_str = line_decoded[len("data:"):].strip()
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = data["choices"][0]["delta"].get("content")
            if delta:
                full_text += delta
                if print_stream:
                    print(delta, end="", flush=True)

    if print_stream:
        print()
    return full_text



if __name__ == "__main__":
    API_KEY = "sk-ayzuywteajduprnlfdhsmjmealxtdnnvbddyetnkrvomskcj"
    prompt = "请用中文解释什么是大语言模型，并举一个应用场景。"

    result = siliconflow_stream_chat(prompt, api_key=API_KEY)
    print("\n===== 完整结果 =====")
    print(result)