import json
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

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

# 加载环境变量
load_dotenv(dotenv_path=ENV_FILE, override=False)

# 支持的 API 模式
SUPPORTED_API_MODES = {"chat.completions", "responses"}
MODEL_TEMPERATURE_CONSTRAINTS = {
    "kimi-k2.5": 1.0,
}


def build_client() -> OpenAI:
    """
    构建一个 OpenAI 客户端实例。

    :return: OpenAI 客户端实例
    """
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
    """
    构建用户提示。

    :return: 用户提示字符串
    """
    return f"""
请根据以下用户输入推荐一道合适的菜谱。

输出要求：
{OUTPUT_REQUIREMENTS}

用户输入：
{INPUT_TEXT}
""".strip()


def get_env_str(name: str, default: str) -> str:
    """
    获取环境变量值，返回字符串类型。

    :param name: 环境变量名
    :param default: 默认值
    :return: 环境变量值或默认值
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def get_env_float(name: str, default: float) -> float:
    """
    获取环境变量值，返回浮点数类型。

    :param name: 环境变量名
    :param default: 默认值
    :return: 环境变量值或默认值
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是合法数字，当前值为: {value}") from exc


def get_env_int(name: str, default: int) -> int:
    """
    获取环境变量值，返回整数类型。

    :param name: 环境变量名
    :param default: 默认值
    :return: 环境变量值或默认值
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是合法整数，当前值为: {value}") from exc


def get_api_mode() -> str:
    """
    获取 API 模式。

    :return: API 模式字符串
    """
    raw_mode = get_env_str("OPENAI_API_MODE", "chat.completions").lower()
    aliases = {
        "chat": "chat.completions",
        "chat.completions": "chat.completions",
        "responses": "responses",
        "response": "responses",
    }
    api_mode = aliases.get(raw_mode)
    if api_mode is None or api_mode not in SUPPORTED_API_MODES:
        supported = ", ".join(sorted(SUPPORTED_API_MODES))
        raise RuntimeError(
            f"OPENAI_API_MODE 不支持当前值: {raw_mode}。可选值: {supported}"
        )
    return api_mode


def get_model_constraints(model: str) -> dict:
    """
    获取模型约束。

    :param model: 模型名称
    :return: 模型约束字典
    """
    normalized_model = model.strip().lower()
    return {
        "fixed_temperature": MODEL_TEMPERATURE_CONSTRAINTS.get(normalized_model),
    }


def build_sampling_options(model: str) -> dict:
    """
    构建采样选项。

    :param model: 模型名称
    :return: 采样选项字典
    """
    constraints = get_model_constraints(model)
    fixed_temperature = constraints["fixed_temperature"]
    default_temperature = fixed_temperature if fixed_temperature is not None else 0.5

    options = {
        "temperature": get_env_float("OPENAI_TEMPERATURE", default_temperature),
        "top_p": get_env_float("OPENAI_TOP_P", 0.95),
    }

    if fixed_temperature is not None:
        options["temperature"] = fixed_temperature

    return options


def build_chat_request_kwargs(model: str) -> dict:
    """
    构建聊天请求参数。

    :param model: 模型名称
    :return: 聊天请求参数字典
    """
    sampling_options = build_sampling_options(model)

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt()},
        ],
        **sampling_options,
        "max_tokens": get_env_int("OPENAI_MAX_TOKENS", 1024),
        "frequency_penalty": get_env_float("OPENAI_FREQUENCY_PENALTY", 0.0),
        "presence_penalty": get_env_float("OPENAI_PRESENCE_PENALTY", 0.0),
    }


def build_responses_request_kwargs(model: str) -> dict:
    """
    构建响应请求参数。

    :param model: 模型名称
    :return: 响应请求参数字典
    """
    sampling_options = build_sampling_options(model)

    return {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": build_user_prompt(),
        **sampling_options,
        "max_output_tokens": get_env_int("OPENAI_MAX_TOKENS", 1024),
    }


def extract_responses_text(response) -> str:
    """
    从响应中提取文本文本。

    :param response: 响应对象
    :return: 文本内容
    """
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)

    return "\n".join(parts).strip()


def parse_error_payload(response_text: str) -> tuple[str, object]:
    """
    解析错误响应文本。

    :param response_text: 错误响应文本
    :return: 错误消息和错误负载元组
    """
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text, None

    if isinstance(payload, dict):
        error_value = payload.get("error")
        if isinstance(error_value, dict):
            return error_value.get("message", response_text), payload
        if isinstance(error_value, str):
            return error_value, payload
        if isinstance(payload.get("message"), str):
            return payload["message"], payload

    return response_text, payload


def should_fallback_to_chat(exc: APIStatusError, api_mode: str) -> bool:
    """
    判断是否需要回退到聊天模式。

    :param exc: API 状态错误异常
    :param api_mode: API 模式
    :return: 是否需要回退到聊天模式
    """
    if api_mode != "responses":
        return False

    message, payload = parse_error_payload(exc.response.text)
    normalized_message = message.lower()
    if exc.status_code == 404 and (
        "/v1/responses" in exc.response.text
        or "url.not_found" in normalized_message
        or "not found" in normalized_message
    ):
        return True

    if isinstance(payload, dict) and payload.get("url") == "/v1/responses":
        return True

    return False


def explain_api_status_error(exc: APIStatusError, model: str, api_mode: str) -> list[str]:
    """
    解释 API 状态错误。

    :param exc: API 状态错误异常
    :param model: 模型名称
    :param api_mode: API 模式
    :return: 建议列表
    """
    message, payload = parse_error_payload(exc.response.text)
    normalized_message = message.lower()
    suggestions = [
        f"当前接口模式: {api_mode}",
        f"当前模型: {model}",
    ]

    if "temperature" in normalized_message and "only 1 is allowed" in normalized_message:
        suggestions.append(
            "该模型限制 temperature=1，可在 .env 中设置 OPENAI_TEMPERATURE=1，或直接使用脚本内置默认值。"
        )

    if "model" in normalized_message and (
        "not found" in normalized_message
        or "does not exist" in normalized_message
        or "invalid model" in normalized_message
    ):
        suggestions.append(
            "请检查 OPENAI_MODEL 是否与当前 OPENAI_BASE_URL 对应服务商支持的模型名称一致。"
        )

    if "unsupported" in normalized_message and "parameter" in normalized_message:
        suggestions.append(
            "当前模型或接口模式可能不支持某些采样参数，可尝试降低自定义参数数量，或切换 OPENAI_API_MODE。"
        )

    if "messages" in normalized_message and "input" in normalized_message:
        suggestions.append(
            "这通常是请求体格式与接口模式不匹配导致的，请检查 OPENAI_API_MODE 是否设置正确。"
        )

    if should_fallback_to_chat(exc, api_mode):
        suggestions.append(
            "当前服务商未实现 /v1/responses 端点，建议使用 OPENAI_API_MODE=chat.completions。"
        )

    if isinstance(payload, dict) and payload.get("code") == 5:
        suggestions.append("服务端返回 url.not_found，通常表示当前 base_url 不支持该接口路径。")

    if exc.status_code == 401:
        suggestions.append("请检查 OPENAI_API_KEY 是否有效，或是否缺少对应模型的访问权限。")
    elif exc.status_code == 429:
        suggestions.append("请求可能触发了限流或配额限制，请稍后重试并检查服务商配额。")

    return suggestions


def call_model() -> None:
    """
    调用模型。

    :return: None
    """
    client = build_client()
    model = os.getenv("OPENAI_MODEL")
    if not model or model.strip() == "":
        raise RuntimeError("缺少 OPENAI_MODEL。请在项目根目录的 .env 文件中配置。")
    model = model.strip()
    api_mode = get_api_mode()

    print(f"调用模式: {api_mode}")

    if api_mode == "responses":
        try:
            response = client.responses.create(**build_responses_request_kwargs(model))
            print(extract_responses_text(response) or "模型未返回文本内容。")
            return
        except APIStatusError as exc:
            if should_fallback_to_chat(exc, api_mode):
                print("提示: 当前服务端不支持 responses 端点，自动回退到 chat.completions。")
            else:
                raise

    completion = client.chat.completions.create(**build_chat_request_kwargs(model))
    print(completion.choices[0].message.content or "模型未返回文本内容。")


def main() -> None:
    try:
        print(f"开始调用模型: {os.getenv('OPENAI_MODEL', '<未配置>')}")
        call_model()
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
    except APIConnectionError:
        print("网络连接失败，请检查 OPENAI_BASE_URL、网络代理或服务端地址是否可用。")
    except APIStatusError as exc:
        print(f"API 调用失败: status_code={exc.status_code}")
        print(f"响应头: {dict(exc.response.headers)}")
        print(f"响应正文: {exc.response.text}")
        model = get_env_str("OPENAI_MODEL", "<未配置>")
        api_mode = get_env_str("OPENAI_API_MODE", "chat.completions")
        for tip in explain_api_status_error(exc, model, api_mode):
            print(f"诊断建议: {tip}")


if __name__ == "__main__":
    main()