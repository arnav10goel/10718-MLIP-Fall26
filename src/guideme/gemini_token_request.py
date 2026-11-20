"""Exact pure mapping of this runtime's Gemini request to countTokens kwargs."""

import base64
import json
import math


def build_count_tokens_request(generation_request):
    """Return detached count_tokens kwargs for the supported generation shape.

    Map every supported semantic setting to the REST generateContentRequest in
    http_options.extra_body, including system text, response schema and optional
    global media resolution. Preserve the original model argument, adding its
    models/ prefix only once inside the REST body. Only bare aliases or one
    models/alias resource are supported; other resource paths are rejected.
    Preserve timeout and one SDK
    attempt. Disabled automatic function calling is validated but omitted from
    that body because it is an SDK-only behavior, not a generation wire field.

    Unsupported fields or invalid values raise ValueError instead of silently
    counting a different request. Image/JPEG/text/call budgets belong upstream;
    this mapper introduces no cap, truncation, resampling or schema inference.
    Base64 expands payload size. No SDK, credential, environment, file or network
    access occurs, and returned containers never alias the input containers.
    """
    def keys(record, required, optional=()):
        if (not isinstance(record, dict) or not set(required) <= record.keys()
                or not record.keys() <= set(required) | set(optional)):
            raise ValueError("Request has missing or unsupported fields")

    def text(value, nonblank=False):
        if not isinstance(value, str) or (nonblank and not value.strip()):
            raise ValueError("Request text must be UTF-8 strings")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Request text must be UTF-8 strings") from exc
        return value

    def positive_integer(value):
        if type(value) is not int or value <= 0:
            raise ValueError("Request limits must be positive ordinary integers")
        return value

    def copy_json(value, ancestors):
        kind = type(value)
        if value is None or kind in (bool, int):
            return value
        if kind is str:
            return text(value)
        if kind is float and math.isfinite(value):
            return value
        if kind not in (dict, list) or id(value) in ancestors:
            raise ValueError("Response schema must contain finite acyclic JSON values")
        ancestors.add(id(value))
        try:
            if kind is dict:
                if any(type(key) is not str for key in value):
                    raise ValueError("Response schema object keys must be strings")
                return {text(key): copy_json(item, ancestors) for key, item in value.items()}
            return [copy_json(item, ancestors) for item in value]
        finally:
            ancestors.remove(id(value))

    keys(generation_request, ("model", "contents", "config"))
    model = text(generation_request["model"], nonblank=True)
    alias = model.removeprefix("models/")
    if not alias.strip() or "/" in alias:
        raise ValueError("Model must be a bare alias or one models/alias resource")
    contents = generation_request["contents"]
    if not isinstance(contents, list) or len(contents) != 1:
        raise ValueError("Expected exactly one user content record")
    keys(contents[0], ("role", "parts"))
    if contents[0]["role"] != "user":
        raise ValueError("Only user content is supported")
    parts = contents[0]["parts"]
    if not isinstance(parts, list) or not parts:
        raise ValueError("User content must contain parts")
    wire_parts = []
    for part in parts:
        if isinstance(part, dict) and set(part) == {"text"}:
            wire_parts.append({"text": text(part["text"])})
        else:
            keys(part, ("inline_data",))
            blob = part["inline_data"]
            keys(blob, ("mime_type", "data"))
            if blob["mime_type"] != "image/jpeg" or type(blob["data"]) is not bytes:
                raise ValueError("Inline parts require image/jpeg and immutable bytes")
            wire_parts.append({"inlineData": {"mimeType": "image/jpeg",
                "data": base64.b64encode(blob["data"]).decode("ascii")}})

    config = generation_request["config"]
    keys(config, ("system_instruction", "response_mime_type", "response_json_schema",
                  "candidate_count", "max_output_tokens", "automatic_function_calling",
                  "thinking_config", "http_options"), ("media_resolution",))
    system = text(config["system_instruction"], nonblank=True)
    if config["response_mime_type"] != "application/json":
        raise ValueError("Only application/json response MIME type is supported")
    if type(config["candidate_count"]) is not int or config["candidate_count"] != 1:
        raise ValueError("Exactly one generated candidate is supported")
    output_limit = positive_integer(config["max_output_tokens"])
    keys(config["automatic_function_calling"], ("disable",))
    if config["automatic_function_calling"]["disable"] is not True:
        raise ValueError("Automatic function calling must be disabled")
    thinking = config["thinking_config"]
    keys(thinking, ("thinking_level", "include_thoughts"))
    if thinking["thinking_level"] != "MINIMAL" or thinking["include_thoughts"] is not False:
        raise ValueError("Expected MINIMAL thinking without thought output")
    http = config["http_options"]
    keys(http, ("timeout", "retry_options"))
    timeout = positive_integer(http["timeout"])
    keys(http["retry_options"], ("attempts",))
    if type(http["retry_options"]["attempts"]) is not int or http["retry_options"]["attempts"] != 1:
        raise ValueError("Exactly one SDK attempt is supported")
    if type(config["response_json_schema"]) is not dict:
        raise ValueError("Response schema must be a JSON object")
    try:
        schema = copy_json(config["response_json_schema"], set())
        json.dumps(schema, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Response schema must contain finite UTF-8 JSON values") from exc
    generation = {"candidateCount": 1, "maxOutputTokens": output_limit,
                  "responseMimeType": "application/json", "responseJsonSchema": schema,
                  "thinkingConfig": {"thinkingLevel": "MINIMAL", "includeThoughts": False}}
    if "media_resolution" in config:
        resolution = config["media_resolution"]
        if resolution not in ("MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH"):
            raise ValueError("Unsupported global media resolution")
        generation["mediaResolution"] = resolution
    wire = {"model": "models/" + alias,
            "contents": [{"role": "user", "parts": wire_parts}],
            "systemInstruction": {"role": "user", "parts": [{"text": system}]},
            "generationConfig": generation}
    return {"model": model, "contents": None, "config": {"http_options": {
        "timeout": timeout, "retry_options": {"attempts": 1},
        "extra_body": {"generateContentRequest": wire}}}}
