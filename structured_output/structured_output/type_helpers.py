from dataclasses import dataclass
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

T = TypeVar("T", bound=BaseModel)
V = TypeVar("V", bound=BaseModel)

UNKNOWN_PLACEHOLDER = "__UNKNOWN"


@dataclass
class DefaultContainer:
    value: Any | None
    factory: Callable | None

    def get(self) -> Any:
        if self.value is not None:
            return self.value
        elif self.factory is not None:
            return self.factory()
        return None

    def __hash__(self) -> int:
        return hash((self.value, self.factory))


@dataclass
class ExpandedUnion:
    members: tuple[Type[Any]]

    def extract_value(self, value: Any) -> tuple[Type[Any], Any]:
        if not hasattr(value, "type"):
            return type(None), None

        selection = getattr(value, "type")
        if not hasattr(value, selection):
            return type(None), None

        value_type = next(
            (member for member in self.members if member.__name__ == selection),
            type(None),
        )

        return value_type, getattr(value, selection)

    def __hash__(self) -> int:
        return hash(self.members)


def _is_pydantic(model: Any) -> bool:
    return (
        model
        and not get_origin(model)
        and isinstance(model, type)
        and issubclass(model, BaseModel)
    )


def _expand_union(members: tuple[Type[Any]]) -> Type[BaseModel]:
    fields = {}

    fields["type"] = (
        Literal[tuple(member.__name__ for member in members)],  # type: ignore
        Field(...),
    )

    for member in members:
        fields[member.__name__] = (
            Union[tuple([_unwrap_pydantic_type(member), None])],  # type: ignore
            Field(...),
        )

    model = create_model(  # type: ignore
        "UnionModel",
        **fields,
        __doc__="Choose one of the options. Fill out 'type' with your choice and the corresponding value.",
    )

    return Annotated[model, ExpandedUnion(members)]  # type: ignore


def _unwrap_pydantic_type(model: Type[Any] | None) -> Type[Any]:
    if model is None:
        return type(None)

    origin = get_origin(model)
    args = get_args(model)

    if origin is list:
        return List[_unwrap_pydantic_type(args[0])]  # type: ignore
    if origin is dict:
        return Dict[_unwrap_pydantic_type(args[0]), _unwrap_pydantic_type(args[1])]  # type: ignore
    if origin is UnionType or origin is Union:
        # If there's more than one pydantic field, expand the union into a model to ensure
        # compatibility
        if sum(_is_pydantic(arg) for arg in args) > 1:
            return _expand_union(args)
        # # Otherwise, just return the union
        else:
            return Union[tuple(_unwrap_pydantic_type(arg) for arg in args)]  # type: ignore
    if _is_pydantic(model):
        return make_openai_compatible(model)

    return model


def make_openai_compatible(model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Removes default values from a Pydantic model.

    Replaces fields that have a default e.g.,
        my_field: int = Field(default=0)
    with
        my_field: int | Literal[UNKNOWN_PLACEHOLDER]

    Args:
        model: The Pydantic model to remove default values from.
    Returns:
        A new Pydantic model with default values removed.
    """
    updated_fields: dict[str, tuple[Type, FieldInfo]] = {}

    for name, field in model.model_fields.items():
        updated_type = _unwrap_pydantic_type(field.annotation)

        # Remove the default value from the field (if there is one)
        if not field.is_required():
            # the "..." replaces default. cannot use default=None because that sets the default to None
            # rather than un-setting it.
            updated_field = FieldInfo.merge_field_infos(
                field,
                Field(
                    ...,
                    default_factory=None,
                ),
            )

            updated_type = Annotated[
                # Allow "unknown" response
                Union[updated_type, Literal[UNKNOWN_PLACEHOLDER]],  # type: ignore
                # keep track of what the default is so we can fill it in later
                DefaultContainer(value=field.default, factory=field.default_factory),
            ]  # type: ignore
        else:
            updated_field = field

        updated_fields[name] = (updated_type, updated_field)

    model = create_model(model.__name__, **updated_fields)  # type: ignore

    return model


def patch_openai_value(value: T, target_model: type[V]) -> V:
    """
    Maps an instance of a type created using `remove_default_values` to a target model.

    Args:
        value: The instance of the model to patch.
        target_model: The target model to patch the instance to.
    Returns:
        The patched instance of the model.
    """
    union_values: dict[str, Any] = {}

    for name, field in value.model_fields.items():
        if getattr(value, name) == UNKNOWN_PLACEHOLDER:
            for metadata in field.metadata:
                if isinstance(metadata, DefaultContainer):
                    setattr(value, name, metadata.get())
                    break

        # recurse if the field is a pydantic model
        field_value = getattr(value, name)
        field_model: Type[Any] | None

        if union_annotation := next(
            (
                metadata
                for metadata in field.metadata
                if isinstance(metadata, ExpandedUnion)
            ),
            None,
        ):
            field_model, field_value = union_annotation.extract_value(field_value)
            union_values[name] = field_value
        else:
            field_model = target_model.model_fields[name].annotation
        model_args = get_args(field_model)

        if field_model:
            if isinstance(field_value, BaseModel):
                setattr(value, name, patch_openai_value(field_value, field_model))
            elif isinstance(field_value, list) and _is_pydantic(model_args[0]):
                setattr(
                    value,
                    name,
                    [patch_openai_value(item, model_args[0]) for item in field_value],
                )
            elif isinstance(field_value, dict) and _is_pydantic(model_args[1]):
                setattr(
                    value,
                    name,
                    {
                        key: patch_openai_value(item, model_args[1])
                        for key, item in field_value.items()
                    },
                )

    dumped = value.model_dump()

    # Target model fields which were expanded unions won't dump the correct value. We want
    # the field to have the value of the union member, not the expanded union model.
    for name, value in union_values.items():
        dumped[name] = value.model_dump()

    return target_model.model_validate(dumped)
