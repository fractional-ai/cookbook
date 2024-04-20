from enum import Enum
from typing import Any, Callable, Generic, Literal, Sequence, Type, TypeVar, cast
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    StrOutputParser,
    SystemMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import BaseModel


InputType = TypeVar("InputType")
T = TypeVar("T")
PyT = TypeVar("PyT", bound=BaseModel)

UndefinedType = Literal["__undefined__"]
UNDEFINED: UndefinedType = "__undefined__"

OUTPUT_VAR_KEY = "last_output"
OUTPUTS_KEY = "outputs"
HISTORY_KEY = "history"
INPUTS_KEY = "inputs"
TMP_OUTPUT_KEY = "__tmp_output"


class HistoryUpdateMode(Enum):
    APPEND = "append"
    REPLACE = "replace"
    SKIP = "skip"


class StatefulChainBuilder(Generic[InputType]):
    """
    A class for building a chain of stateful operations
    """

    chain: Runnable[Any, dict[str, Any]]
    index: int
    llm: BaseChatModel
    history: ChatPromptTemplate

    def __init__(
        self,
        llm: BaseChatModel,
        prefix: str = "",
        existing_history: ChatPromptTemplate | None = None,
        initial_state: dict[str, Any] | None = None,
    ):
        self.history = ChatPromptTemplate.from_messages(
            existing_history.messages if existing_history else []
        )
        self.llm = llm
        self.index = 0
        self.chain = StatefulChainBuilder._inject_state(initial_state)
        self.prefix = prefix

    def system(self, message: str) -> "StatefulChainBuilder[InputType]":
        """
        Add a system message to the chat history
        """
        return self._append_prompt(
            messages=ChatPromptTemplate.from_messages([SystemMessage(content=message)]),
            append_to_history=True,
            include_history=True,
            update_output=False,
        )
        # self.history = ChatPromptTemplate.from_messages(
        #     [
        #         *self.history.messages,
        #         SystemMessage(content=message),
        #     ]
        # )
        # return self

    def prompt(
        self,
        prompt: str | None = None,
        messages: Sequence[MessageLikeRepresentation] = [],
        include_history: bool = True,
        llm: BaseChatModel | None = None,
    ) -> "StatefulChainBuilder[str]":
        """
        Prompt the LLM with a message, adding response to the chat history

        Args:
            prompt_message: The message to prompt the LLM with
            other_messages: Additional messages to add to the chat history
            include_history: Whether to include the chat history in the prompt
        Returns:
            The current builder
        """
        llm = llm or self.llm
        _messages: list[MessageLikeRepresentation] = []

        if prompt:
            _messages.append(HumanMessage(content=prompt))
        if messages:
            _messages.extend(messages)

        return self._append_with_prompt(
            messages=self._build_messages(_messages),
            runnable=llm | StrOutputParser(),
            output_type=str,
            include_history=include_history,
        )

    def structured_prompt(
        self,
        output_schema: Type[PyT],
        prompt: str | None = None,
        messages: Sequence[MessageLikeRepresentation] = [],
        include_history: bool = True,
        append_to_history: bool = True,
        llm: BaseChatModel | None = None,
    ) -> "StatefulChainBuilder[PyT]":
        """
        Prompt the LLM, asking it to respond with the provided `output_schema`.
        The JSON-ified response will be added to the chat history.

        Args:
            other_messages: Additional messages to add to the chat history
            include_history: Whether to include the chat history in the prompt
            prompt: The message to prompt the LLM with
        Returns:
            The current builder
        """
        llm = llm or self.llm
        _messages: list[MessageLikeRepresentation] = []

        if prompt:
            _messages.append(HumanMessage(content=prompt))
        if messages:
            _messages.extend(messages)

        return self._append_with_prompt(
            messages=self._build_messages(_messages),
            output_type=output_schema,
            runnable=llm.with_structured_output(output_schema),
            include_history=include_history,
            append_to_history=append_to_history,
        )

    def branch(
        self,
        condition: (
            Callable[[InputType], bool] | Callable[[InputType, dict[str, Any]], bool]
        ),
        if_true: (
            Callable[["StatefulChainBuilder[InputType]"], "StatefulChainBuilder[T]"]
            | None
        ) = None,
        if_false: (
            Callable[["StatefulChainBuilder[InputType]"], "StatefulChainBuilder[T]"]
            | None
        ) = None,
        value_if_false: T | UndefinedType | None = UNDEFINED,
        value_if_true: T | UndefinedType | None = UNDEFINED,
        output_field: str | None = None,
        output_type: Type[T] | None = None,
    ) -> "StatefulChainBuilder[T]":
        """
        Branch the chain based on a condition.

        Args:
            condition: The condition to branch on. Input will be the output of the previous step in the builder
            then: A function that takes a new builder and returns the chain to run if the condition is true
            otherwise: A function that takes a new builder and returns the chain to run if the condition is false
        """

        def _route(state: dict[str, Any]) -> Runnable[Any, dict[str, Any]]:
            _sub_builder = StatefulChainBuilder[InputType](
                self.llm, "branch_" + self.prefix, initial_state=state
            )
            sub_builder = _sub_builder.with_history(state.get(HISTORY_KEY, []))

            if value_if_true != UNDEFINED:
                _if_true = lambda b: b.run_lambda(
                    lambda _: value_if_true,
                    name=f"{self.prefix}if_true",
                )
            elif if_true:
                _if_true = if_true
            else:
                raise ValueError("Must provide either if_true or value_if_true")

            if value_if_false != UNDEFINED:
                _if_false = lambda b: b.run_lambda(
                    lambda _: value_if_false,
                    name=f"{self.prefix}if_false",
                )
            elif if_false is not None:
                _if_false = if_false
            else:
                raise ValueError("Must provide either if_false or value_if_false")

            if StatefulChainBuilder._call_with_output(condition, state):
                branch = _if_true(sub_builder)
            else:
                branch = _if_false(sub_builder)

            return RunnableLambda(lambda _: state[INPUTS_KEY]) | branch.build()

        return self._append(
            runnable=RunnableLambda(_route),
            output_field=output_field,
        )

    def run_lambda(
        self,
        _lambda: Callable[[InputType], T] | Callable[[InputType, dict[str, Any]], T],
        output_field: str | None = None,
        name: str | None = None,
    ) -> "StatefulChainBuilder[T]":
        """
        Add a step which calls a lambda function on the output of the previous step.

        Args:
            _lambda: The lambda function to call
        Returns:
            The current builder
        """
        output_field = output_field or self._get_output_var_name()

        def _lambda_with_output(state: dict[str, Any]) -> T:
            return StatefulChainBuilder._call_with_output(_lambda, state)

        _runnable: Runnable = RunnableLambda(_lambda_with_output, name=name)

        return self._append(
            runnable=_runnable,
            output_field=output_field,
        )

    def run(self, inputs: dict[str, Any] = {}) -> InputType:
        output: InputType = self.build().invoke(inputs)
        return output

    def build(
        self, output_parser: Callable[[dict[str, Any]], T] | None = None
    ) -> Runnable[Any, T]:
        """
        Build the chain into a Runnable object

        Args:
            output_variable: The variable to return from the chain (default is the output of the last step)
        Returns:
            The Runnable object
        """

        def _get_output(state: dict[str, Any]) -> T:
            value = state[OUTPUTS_KEY][state[OUTPUT_VAR_KEY]]

            if output_parser:
                value = output_parser(value)

            return cast(T, value)

        chain = self.build_raw() | _get_output
        return chain

    def build_passthrough(self, *keys: str) -> Runnable[Any, dict[str, Any]]:
        """
        Build the chain, appending all outputs to inputs

        Args:
            keys: The keys to include in the output. If none are provided, all keys will be included
        Returns:
            The Runnable object
        """

        def _get_output(state: dict[str, Any]) -> dict[str, Any]:
            if keys:
                return {
                    **state[INPUTS_KEY],
                    **{
                        key: state[OUTPUTS_KEY][key]
                        for key in keys
                        if key in state[OUTPUTS_KEY]
                    },
                }
            else:
                return {**state[INPUTS_KEY], **state[OUTPUTS_KEY]}

        return self.build_raw() | RunnableLambda(_get_output)

    def build_raw(self) -> Runnable[Any, dict[str, Any]]:
        """
        Build the chain with no output parser -- will return the full state dictionary.

        Returns:
            The Runnable object
        """
        return self.chain

    def with_history(
        self, history: Sequence[MessageLikeRepresentation]
    ) -> "StatefulChainBuilder[InputType]":
        """
        Sets the chat history to the provided messages.

        Args:
            history: The chat history
        Returns:
            The current builder
        """
        self.history = ChatPromptTemplate.from_messages(history)
        return self

    def clone_state(self, state: dict[str, Any]):
        self.chain = RunnableLambda(lambda _: state)
        return self

    def _get_output_var_name(self):
        var = f"_output_{self.index}"
        self.index += 1
        return var

    def _build_messages(
        self, messages: Sequence[MessageLikeRepresentation] | str
    ) -> ChatPromptTemplate:
        _messages: Sequence[MessageLikeRepresentation]

        if isinstance(messages, str):
            _messages = [HumanMessage(content=messages)]
        else:
            _messages = messages

        return ChatPromptTemplate.from_messages(_messages)

    @staticmethod
    def _call_with_output(
        _lambda: Callable[[InputType], T] | Callable[[InputType, dict[str, Any]], T],
        state: dict[str, Any],
    ) -> T:
        # if there is no last output, default to the input of the chain
        chain_inputs = state[INPUTS_KEY]
        last_output = (
            state[OUTPUTS_KEY][state[OUTPUT_VAR_KEY]]
            if state[OUTPUT_VAR_KEY]
            else chain_inputs
        )
        num_args = _lambda.__code__.co_argcount

        if num_args == 1:
            result = cast(Callable[[InputType], T], _lambda)(last_output)
        elif num_args == 2:
            result = cast(Callable[[InputType, dict[str, Any]], T], _lambda)(
                last_output, chain_inputs
            )

        return result

    @staticmethod
    def _append_value(
        input: dict[str, Any], value: Any, fallback_key: str
    ) -> dict[str, Any]:
        if isinstance(value, dict):
            return {**input, **value}
        else:
            return {**input, fallback_key: value}

    def _build_part_input(self) -> Runnable[dict[str, Any], Any]:
        def _select_output_field(state: dict[str, Any]) -> Any:
            current_output_key = state[OUTPUT_VAR_KEY]

            if not current_output_key:
                return state[INPUTS_KEY]
            else:
                return state[OUTPUTS_KEY][current_output_key]

        return RunnableLambda(_select_output_field)

    def _build_prompt_input(
        self, prompt: ChatPromptTemplate, include_history: bool
    ) -> Runnable[dict[str, Any], Any]:
        def _build_prompt(state: dict[str, Any]) -> list[BaseMessage]:
            current_output_key = state[OUTPUT_VAR_KEY]
            prompt_params = {}

            if include_history:
                prompt_params[HISTORY_KEY] = state[HISTORY_KEY]

            if not current_output_key:
                prompt_params = StatefulChainBuilder._append_value(
                    prompt_params, state[INPUTS_KEY], TMP_OUTPUT_KEY
                )
            else:
                prompt_params = StatefulChainBuilder._append_value(
                    prompt_params,
                    state[OUTPUTS_KEY][current_output_key],
                    TMP_OUTPUT_KEY,
                )

            return prompt.invoke(prompt_params).to_messages()

        return RunnableLambda(_build_prompt)

    @staticmethod
    def _inject_state(
        initial_state: dict[str, Any] | None
    ) -> Runnable[dict[str, Any], dict[str, Any]]:
        def _inject(inputs: Any) -> dict[str, Any]:
            return {
                INPUTS_KEY: inputs,
                HISTORY_KEY: [],
                OUTPUTS_KEY: {},
                OUTPUT_VAR_KEY: None,
                **(initial_state or {}),
            }

        return RunnableLambda(_inject)

    def _append_with_prompt(
        self,
        messages: ChatPromptTemplate,
        runnable: Runnable[Any, Any],
        include_history: bool = True,
        append_to_history: bool = True,
        output_field: str | None = None,
        output_type: Type[T] | None = None,
    ) -> "StatefulChainBuilder[T]":
        self._append_prompt(
            messages,
            append_to_history=append_to_history,
            include_history=include_history,
        )

        return self._append(
            runnable=self._build_part_input() | runnable,
            output_field=output_field,
            output_type=output_type,
            history_update_mode=(
                HistoryUpdateMode.APPEND
                if append_to_history
                else HistoryUpdateMode.SKIP
            ),
        )

    def _append_prompt(
        self,
        messages: ChatPromptTemplate,
        append_to_history: bool = True,
        include_history: bool = True,
        update_output: bool = True,
    ) -> "StatefulChainBuilder[T]":
        prompt = ChatPromptTemplate.from_messages(
            [
                *(
                    [MessagesPlaceholder(HISTORY_KEY, optional=True)]
                    if include_history
                    else []
                ),
                *(messages.messages if messages else []),
            ]
        )

        return self._append(
            runnable=self._build_prompt_input(prompt, include_history=include_history),
            output_field=TMP_OUTPUT_KEY,
            history_update_mode=(
                HistoryUpdateMode.REPLACE
                if append_to_history
                else HistoryUpdateMode.SKIP
            ),
            update_output=update_output,
        )

    def _append(
        self,
        runnable: Runnable[Any, Any],
        history_update_mode: HistoryUpdateMode = HistoryUpdateMode.SKIP,
        output_field: str | None = None,
        output_type: Type[T] | None = None,
        update_output: bool = True,
    ) -> "StatefulChainBuilder[T]":
        output_field = output_field or self._get_output_var_name()

        result = self.chain | (RunnablePassthrough.assign(**{TMP_OUTPUT_KEY: runnable}))

        def _update_history(state: dict[str, Any]) -> dict[str, Any]:
            if history_update_mode == HistoryUpdateMode.APPEND:
                return {
                    **state,
                    HISTORY_KEY: [
                        *state.get(HISTORY_KEY, []),
                        AIMessage(content=str(state[TMP_OUTPUT_KEY])),
                    ],
                }
            elif history_update_mode == HistoryUpdateMode.REPLACE:
                return {
                    **state,
                    HISTORY_KEY: state[TMP_OUTPUT_KEY],
                }
            else:
                return state

        result = result | _update_history

        def _update_output(state: dict[str, Any]) -> dict[str, Any]:
            return {
                **state,
                OUTPUT_VAR_KEY: output_field,
                OUTPUTS_KEY: {
                    **state[OUTPUTS_KEY],
                    output_field: state[TMP_OUTPUT_KEY],
                },
            }

        if update_output:
            result = result | _update_output

        self.chain = result

        return cast("StatefulChainBuilder[T]", self)
