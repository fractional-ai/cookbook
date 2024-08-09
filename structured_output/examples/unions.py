from typing import Literal
from openai import OpenAI
from pydantic import BaseModel, Field
from structured_output.openai_wrapper import structured_ask

from structured_output.type_helpers import (
    UNKNOWN_PLACEHOLDER,
    make_openai_compatible,
    patch_openai_value,
)


class Article(BaseModel):
    type: Literal["article"] = "article"
    title: str
    author: str = "DEFAULT AUTHOR"
    text: str

class Tweet(BaseModel):
    type: Literal["tweet"] = "tweet"
    content: str
    author_name: str 
    author_handle: str

class Content(BaseModel):
    reasoning: str
    content: Article | Tweet 

ARTICLE_TEXT = """
Title: Hello world!
By:

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus a malesuada ex. Praesent efficitur, justo at suscipit efficitur, ex tortor blandit diam, consectetur malesuada lectus risus sed nisl.
"""

TWEET_TEXT = """
Sam Altman
@sama

Aug 7
i love summer in the garden
"""

client = OpenAI()
print(
    structured_ask(
        client,
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Extract the provided content.",
            },
            {"role": "user", "content": ARTICLE_TEXT},
        ],
        response_model=Content,
    )
)
# reasoning='...' content=Article(type='article', title='Hello world!', author='DEFAULT AUTHOR', text='...')

print(
    structured_ask(
        client,
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Extract the provided content.",
            },
            {"role": "user", "content": TWEET_TEXT},
        ],
        response_model=Content,
    )
)
# reasoning='...' content=Tweet(type='tweet', content='i love summer in the garden', author_name='Sam Altman', author_handle='@sama')