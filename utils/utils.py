# utils/utils.py
import tiktoken

def count_tokens(text: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def truncate_text(text: str, max_tokens: int, model_name: str) -> str:
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
