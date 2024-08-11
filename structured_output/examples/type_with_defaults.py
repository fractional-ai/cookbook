from openai import OpenAI
from pydantic import BaseModel
from structured_output.openai_wrapper import structured_ask

from structured_output.pydantic_helpers import (
    UNKNOWN_PLACEHOLDER,
    make_openai_compatible,
    patch_openai_value,
)


class Article(BaseModel):
    title: str
    author: str = "DEFAULT AUTHOR"
    text: str


ARTICLE_TEXT = """
Hello world!
By:

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus a malesuada ex. Praesent efficitur, justo at suscipit efficitur, ex tortor blandit diam, consectetur malesuada lectus risus sed nisl.
"""

client = OpenAI()
print(
    structured_ask(
        client,
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Extract the provided article.",
            },
            {"role": "user", "content": ARTICLE_TEXT},
        ],
        response_model=Article,
    )
)
# title='Hello world!' author='DEFAULT AUTHOR' text='...'
