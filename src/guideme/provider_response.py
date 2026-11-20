"""Validate and safely record OpenAI-compatible provider responses."""

from copy import deepcopy


def filter_provider_response(raw):
    """Keep known output fields and omitted field types, never opaque metadata values."""
    result = {key: deepcopy(raw[key]) for key in (
        "id", "model", "object", "created", "system_fingerprint") if key in raw}
    if "choices" in raw:
        choices = raw["choices"]
        result["choices"] = [] if isinstance(choices, list) else None
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    result["choices"].append(None)
                    continue
                item = {key: deepcopy(choice[key]) for key in (
                    "index", "finish_reason") if key in choice}
                message = choice.get("message")
                item["message"] = ({key: deepcopy(message[key]) for key in (
                    "role", "content", "refusal", "tool_calls", "function_call") if key in message}
                    if isinstance(message, dict) else None)
                if isinstance(message, dict):
                    omitted = {key: type(value).__name__ for key, value in message.items()
                               if key not in item["message"]}
                    if omitted:
                        item["omitted_message_field_types"] = omitted
                result["choices"].append(item)
    if isinstance(raw.get("usage"), dict):
        result["usage"] = {key: deepcopy(raw["usage"][key]) for key in (
            "prompt_tokens", "completion_tokens", "total_tokens") if key in raw["usage"]}
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            value = raw["usage"].get(key)
            if type(value) is int and value >= 0:
                result["usage"][key] = value
        for key, fields in (
                ("prompt_tokens_details", ("cached_tokens", "text_tokens", "video_tokens")),
                ("completion_tokens_details", ("reasoning_tokens", "text_tokens"))):
            details = raw["usage"].get(key)
            if isinstance(details, dict):
                counts = {field: details[field] for field in fields
                          if type(details.get(field)) is int and details[field] >= 0}
                if counts:
                    result["usage"][key] = counts
    return result


def extract_response_text(raw):
    if not isinstance(raw, dict):
        raise ValueError("Provider response must be a JSON object")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Exactly one provider choice is required")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ValueError("Provider completion did not finish normally")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Provider must return ordinary assistant text without mixed output")
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("Provider reasoning metadata must be text or null")
    blocks = message.get("thinking_blocks")
    if blocks is not None:
        if not isinstance(blocks, list):
            raise ValueError("Provider thinking metadata must be a list or null")
        for block in blocks:
            if (not isinstance(block, dict) or set(block) - {"type", "thinking", "signature"}
                    or block.get("type") != "thinking" or not isinstance(block.get("thinking"), str)
                    or block.get("signature") is not None and not isinstance(block["signature"], str)):
                raise ValueError("Provider thinking blocks must contain only typed thinking metadata")
    metadata = message.get("provider_specific_fields")
    if metadata is not None and (
            not isinstance(metadata, dict) or set(metadata) - {"thought_signatures"}
            or not isinstance(metadata.get("thought_signatures", []), list)
            or any(not isinstance(value, str) for value in metadata.get("thought_signatures", []))):
        raise ValueError("Provider metadata must contain only a list of thought signatures")
    if (message.get("role") != "assistant"
            or any(message.get(key) not in (None, "", []) for key in (
                "refusal", "tool_calls", "function_call", "reasoning"))
            or any(value not in (None, "", []) for key, value in message.items()
                   if key not in {"role", "content", "refusal", "tool_calls", "function_call",
                                  "reasoning", "reasoning_content", "thinking_blocks", "provider_specific_fields"})
            or not isinstance(message.get("content"), str) or not message["content"].strip()):
        raise ValueError("Provider must return ordinary assistant text without mixed output")
    return message["content"]
