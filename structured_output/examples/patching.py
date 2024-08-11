from pydantic import BaseModel

from structured_output.pydantic_helpers import (
    UNKNOWN_PLACEHOLDER,
    make_openai_compatible,
    patch_openai_value,
)


class Article(BaseModel):
    title: str
    author: str = "DEFAULT AUTHOR"
    text: str


# Creates a patched model with default values removed
patched_model = make_openai_compatible(Article)
article = patched_model(title="Title", author=UNKNOWN_PLACEHOLDER, text="Text")

print(article)
# title='Title' author='__UNKNOWN' text='Text'

print(patch_openai_value(article, Article))
# title='Title' author='DEFAULT AUTHOR' text='Text'
