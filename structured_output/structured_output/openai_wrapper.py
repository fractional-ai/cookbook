from typing import TypeVar

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from structured_output.pydantic_helpers import (
    make_openai_compatible,
    patch_openai_value,
)

T = TypeVar("T", bound=BaseModel)


def structured_ask(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
    response_model: type[T],
    **kwargs,
) -> T | None:
    patched_model = make_openai_compatible(response_model)

    response = client.beta.chat.completions.parse(
        messages=messages,
        response_format=patched_model,
        **kwargs,  # type: ignore
    )

    if not response.choices:
        raise ValueError("LLM response did not contain parsed output")

    parsed_response = response.choices[0].message.parsed
    if not parsed_response:
        return None

    # patch in default values
    return patch_openai_value(parsed_response, response_model)
