import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI


SYSTEM_PROMPT = """
你是一名专业的国宴大厨陈师傅，任务是根据用户输入的食材，推荐合适的菜谱。
这是一个菜谱推荐系统，请给出详细的菜谱步骤，包括所需的调料、烹饪时间和火候。
""".strip()

OUTPUT_REQUIREMENTS = """
给出的菜谱步骤要清晰易懂，调料用量要适中，烹饪时间和火候要合理。
语气要友好、专业，让用户能轻松上手操作。
""".strip()

INPUT_TEXT = """
{
  "ingredients": ["鸡肉", "土豆", "青椒"],
  "dietary_preference": "无辣不欢"
}
""".strip()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


load_dotenv(dotenv_path=ENV_FILE, override=False)


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 OPENAI_API_KEY。请在项目根目录的 .env 文件中配置，或先在终端中设置环境变量。"
        )

    base_url = os.getenv("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def build_user_prompt() -> str:
    return f"""
请根据以下用户输入推荐一道合适的菜谱。

输出要求：
{OUTPUT_REQUIREMENTS}

用户输入：
{INPUT_TEXT}
""".strip()


def openai_chat() -> None:
    client = build_client()
    model = os.getenv("OPENAI_MODEL")
    if not model or model.strip() == "":
        raise RuntimeError("缺少 OPENAI_MODEL。请在项目根目录的 .env 文件中配置。")

    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt()},
        ],
        "temperature": 0.5,  # 控制输出的随机性，0 表示完全随机，0.0 表示完全确定
        "max_tokens": 1024,  # 最大输出 token 数量
        "top_p": 0.95,  # 控制输出的多样性，0.0 到 1.0 之间
        "frequency_penalty": 0.0,  # 控制重复输出的惩罚，0.0 到 1.0 之间
        "presence_penalty": 0.0,  # 控制新输出的惩罚，0.0 到 1.0 之间
    }

    completion = client.chat.completions.create(**request_kwargs)

    print(completion.choices[0].message.content or "模型未返回文本内容。")


def main() -> None:
    try:
        print(f"开始调用模型: {os.getenv('OPENAI_MODEL', '<未配置>')}")
        openai_chat()
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
    except APIConnectionError:
        print("网络连接失败，请检查 OPENAI_BASE_URL、网络代理或服务端地址是否可用。")
    except APIStatusError as exc:
        print(f"API 调用失败: status_code={exc.status_code}")
        print(f"响应头: {dict(exc.response.headers)}")
        print(f"响应正文: {exc.response.text}")


if __name__ == "__main__":
    main()