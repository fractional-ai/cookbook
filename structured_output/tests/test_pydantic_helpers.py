from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from structured_output.pydantic_helpers import (
    UNKNOWN_PLACEHOLDER,
    make_openai_compatible,
    patch_openai_value,
)


class SampleModel(BaseModel):
    field1: int
    field2: str = "default"


class NestedModel(BaseModel):
    nested_field: SampleModel


class Model1(BaseModel):
    type: Literal["model1"]


class Model2(BaseModel):
    type: Literal["model2"]


class Model3(BaseModel):
    type: Literal["model3"]


class UnionModel(BaseModel):
    field: Model1 | Model2 | Model3


def test_removes_defaults():
    model = make_openai_compatible(SampleModel)
    assert model.model_fields["field2"].default is PydanticUndefined
    assert model.model_fields["field2"].default_factory is None


def test_patch_openai_value():
    orig_instance = SampleModel(field1=1)
    model = make_openai_compatible(SampleModel)
    instance = model(**orig_instance.model_dump())
    patched_instance = patch_openai_value(instance, SampleModel)
    assert patched_instance == orig_instance


def test_nested_fields():
    orig_instance = NestedModel(nested_field=SampleModel(field1=1))
    model = make_openai_compatible(NestedModel)
    instance = model(nested_field=dict(field1=1, field2=UNKNOWN_PLACEHOLDER))
    patched_instance = patch_openai_value(instance, NestedModel)
    assert patched_instance == orig_instance


def test_expanded_union():
    orig_instance = UnionModel(field=Model1(type="model1"))
    model = make_openai_compatible(UnionModel)
    print(model.model_fields["field"].annotation.model_fields)
    instance = model(
        field=(
            dict(type="Model1", Model1=dict(type="model1"), Model2=None, Model3=None)
        )
    )
    patched_instance = patch_openai_value(instance, UnionModel)
    assert patched_instance == orig_instance


def test_union_with_default():
    class UnionModelWithDefault(BaseModel):
        field: Model1 | Model2 | Model3 = Model1(type="model1")

    orig_instance = UnionModelWithDefault()
    model = make_openai_compatible(UnionModelWithDefault)
    instance = model(field=UNKNOWN_PLACEHOLDER)
    patched_instance = patch_openai_value(instance, UnionModelWithDefault)
    assert patched_instance == orig_instance


def test_union_with_default_none():
    class UnionModelWithDefaultNone(BaseModel):
        field: Model1 | Model2 | Model3 | None = None

    orig_instance = UnionModelWithDefaultNone()
    model = make_openai_compatible(UnionModelWithDefaultNone)
    instance = model(field=UNKNOWN_PLACEHOLDER)
    patched_instance = patch_openai_value(instance, UnionModelWithDefaultNone)
    assert patched_instance == orig_instance


def test_nested_union_with_default():
    class NestedUnionModelWithDefault(BaseModel):
        field: UnionModel | None = UnionModel(field=Model1(type="model1"))

    orig_instance = NestedUnionModelWithDefault()
    model = make_openai_compatible(NestedUnionModelWithDefault)
    instance = model(field=UNKNOWN_PLACEHOLDER)
    patched_instance = patch_openai_value(instance, NestedUnionModelWithDefault)
    assert patched_instance == orig_instance


def test_union_with_non_pydantic_type():
    class UnionModelWithNonPydanticType(BaseModel):
        field: Model1 | int = 1

    orig_instance = UnionModelWithNonPydanticType()
    model = make_openai_compatible(UnionModelWithNonPydanticType)
    instance = model(field=UNKNOWN_PLACEHOLDER)
    patched_instance = patch_openai_value(instance, UnionModelWithNonPydanticType)
    assert patched_instance == orig_instance


def test_union_with_non_pydantic_type_none():
    class UnionModelWithNonPydanticTypeNone(BaseModel):
        field: Model1 | int | None = None

    orig_instance = UnionModelWithNonPydanticTypeNone(field=1)
    model = make_openai_compatible(UnionModelWithNonPydanticTypeNone)
    instance = model(field=dict(type="int", int=1, Model1=None, NoneType=None))
    patched_instance = patch_openai_value(instance, UnionModelWithNonPydanticTypeNone)
    assert patched_instance == orig_instance


def test_union_with_all_primitive_types():
    class UnionModelWithAllPrimitiveTypes(BaseModel):
        field: int | float | str | bool = 1

    orig_instance = UnionModelWithAllPrimitiveTypes()
    model = make_openai_compatible(UnionModelWithAllPrimitiveTypes)
    instance = model(field=UNKNOWN_PLACEHOLDER)
    patched_instance = patch_openai_value(instance, UnionModelWithAllPrimitiveTypes)
    assert patched_instance == orig_instance


def test_complex_unions():
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

    orig_instance = Content(
        reasoning="reasoning", content=Article(title="title", text="text")
    )
    model = make_openai_compatible(Content)
    instance1 = model(
        reasoning="reasoning",
        content=dict(
            type="Article",
            Article=dict(
                title="title", text="text", type="article", author=UNKNOWN_PLACEHOLDER
            ),
            Tweet=None,
        ),
    )
    instance2 = model(
        reasoning="reasoning",
        content=dict(
            type="Tweet",
            Tweet=dict(
                content="content",
                author_name="author_name",
                author_handle="author_handle",
                type="tweet",
            ),
            Article=None,
        ),
    )

    patched_instance1 = patch_openai_value(instance1, Content)
    assert patched_instance1 == orig_instance

    patched_instance2 = patch_openai_value(instance2, Content)
    assert patched_instance2 == Content(
        reasoning="reasoning",
        content=Tweet(
            content="content", author_name="author_name", author_handle="author_handle"
        ),
    )


def test_idempotent():
    fields = UnionModel.model_fields

    model = make_openai_compatible(UnionModel)
    model2 = make_openai_compatible(UnionModel)

    assert fields == UnionModel.model_fields

    model_data = dict(
        type="Model1", Model1=dict(type="model1"), Model2=None, Model3=None
    )

    instance1 = model(field=model_data)
    instance2 = model2(field=model_data)

    assert instance1.model_dump() == instance2.model_dump()


def test_invalid_model():
    with pytest.raises(ValidationError):
        SampleModel(field1="not an int")
