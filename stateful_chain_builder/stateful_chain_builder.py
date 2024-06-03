import asyncio
import copy
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    cast,
)
from uuid import uuid4
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
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, ValidationError


ChainInputType = TypeVar("ChainInputType")
LastOutputType = TypeVar("LastOutputType")
T = TypeVar("T")
PyT = TypeVar("PyT", bound=BaseModel)

UndefinedType = Literal["__undefined__"]
UNDEFINED: UndefinedType = "__undefined__"

SentinelType = Literal["__sentinel__"]
SENTINEL: SentinelType = "__sentinel__"

OUTPUT_VAR_KEY = "last_output_key"
OUTPUTS_KEY = "outputs"
HISTORY_KEY = "history"
INPUTS_KEY = "inputs"
TMP_OUTPUT_KEY = "tmp_output"

RunState = dict[str, Any]


class HistoryUpdateMode(Enum):
    APPEND = "append"
    REPLACE = "replace"
    SKIP = "skip"
    CLEAR = "clear"


class OutputUpdateMode(Enum):
    LAST_OUTPUT = "last_output"
    FULL_STATE = "full_state"
    SKIP = "skip"


class StepRunnable(Runnable[RunState, RunState]):
    """Class that manages a single step in a stateful chain."""

    def __init__(
        self,
        step: Runnable[RunState, T],
        output_field: str | None,
        output_update_mode: OutputUpdateMode,
        history_update_mode: HistoryUpdateMode,
    ):
        """Run a step in a stateful chain."""

        self.step = step
        self.output_field = output_field
        self.output_update_mode = output_update_mode
        self.history_update_mode = history_update_mode

    def invoke(self, state: RunState, config: RunnableConfig | None = None) -> RunState:
        result = self.step.invoke(state, config)
        return self._update_state(state, result)

    def _update_state(self, state: RunState, result: T) -> RunState:
        state[TMP_OUTPUT_KEY] = result

        # Skip updating history if replacing output
        if self.output_update_mode != OutputUpdateMode.FULL_STATE:
            if self.history_update_mode == HistoryUpdateMode.APPEND:
                state[HISTORY_KEY].append(AIMessage(content=str(result)))
            elif self.history_update_mode == HistoryUpdateMode.REPLACE:
                state[HISTORY_KEY] = result
            elif self.history_update_mode == HistoryUpdateMode.CLEAR:
                state[HISTORY_KEY] = []

        if (
            self.output_update_mode != OutputUpdateMode.SKIP
            and self.output_field is None
        ):
            raise ValueError(
                "Must provide output_field if output_update_mode is not SKIP"
            )

        if self.output_update_mode == OutputUpdateMode.LAST_OUTPUT:
            state[OUTPUT_VAR_KEY] = self.output_field
            state[OUTPUTS_KEY][self.output_field] = result
        elif self.output_update_mode == OutputUpdateMode.FULL_STATE:
            new_state = cast(RunState, result)
            new_state[OUTPUTS_KEY][self.output_field] = new_state[OUTPUTS_KEY][
                new_state[OUTPUT_VAR_KEY]
            ]
            new_state[OUTPUT_VAR_KEY] = self.output_field
            new_state[OUTPUTS_KEY].update(state[OUTPUTS_KEY])
            state = new_state

        return state


class _StatefulChainBuilder(Generic[ChainInputType, LastOutputType]):
    """
    A class for building a chain of stateful operations
    """

    chain: Runnable[Any, dict[str, Any]]
    index: int
    llm: BaseChatModel
    history: ChatPromptTemplate
    name: str

    def __init__(
        self,
        llm: BaseChatModel,
        prefix: str = "",
        existing_history: ChatPromptTemplate | None = None,
        initial_state: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        self.history = ChatPromptTemplate.from_messages(
            existing_history.messages if existing_history else []
        )
        self.llm = llm
        self.index = 0
        self.chain = _StatefulChainBuilder._inject_state(initial_state)
        self.prefix = prefix
        self.name = name or self.__class__.__name__

    def system(
        self, message: str
    ) -> "_StatefulChainBuilder[ChainInputType, LastOutputType]":
        """
        Add a system message to the chat history
        """
        return self._update_history(
            messages=ChatPromptTemplate.from_messages([SystemMessage(content=message)])
        )

    def clear_history(
        self,
    ) -> "_StatefulChainBuilder[ChainInputType, LastOutputType]":
        """
        Clear the chat history
        """
        return self._append(
            # cast is safe because we are not actually changing the output type
            runnable=cast(Runnable[RunState, LastOutputType], RunnablePassthrough()),
            output_field=None,
            history_update_mode=HistoryUpdateMode.CLEAR,
            output_update_mode=OutputUpdateMode.SKIP,
        )

    def prompt(
        self,
        prompt: str | None = None,
        messages: Sequence[MessageLikeRepresentation] = [],
        include_history: bool = True,
        llm: BaseChatModel | None = None,
        output_field: str | None = None,
    ) -> "_StatefulChainBuilder[ChainInputType, str]":
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
            include_history=include_history,
            output_field=output_field or self._get_output_var_name("prompt"),
        )

    def add_messages(
        self,
        *messages: MessageLikeRepresentation,
    ) -> "_StatefulChainBuilder[ChainInputType, LastOutputType]":
        """
        Add messages to the chat history
        """
        return self._update_history(
            messages=self._build_messages(messages),
        )

    def structured_prompt(
        self,
        output_schema: Type[PyT],
        prompt: str | None = None,
        messages: Sequence[MessageLikeRepresentation] = [],
        include_history: bool = True,
        append_to_history: bool = True,
        llm: BaseChatModel | None = None,
        output_field: str | None = None,
        fallback_value: Optional[PyT] | UndefinedType = UNDEFINED,
    ) -> "_StatefulChainBuilder[ChainInputType, PyT]":
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

        runnable: Runnable[list[BaseMessage], dict | BaseModel] = (
            llm.with_structured_output(output_schema)
        )

        if fallback_value != UNDEFINED:
            runnable = runnable.with_fallbacks(
                fallbacks=[RunnableLambda(lambda _: fallback_value)],  # type: ignore
                exceptions_to_handle=(ValidationError,),
            )

        return self._append_with_prompt(
            messages=self._build_messages(_messages),
            # assume that model will output our type
            runnable=cast(Runnable[list[BaseMessage], PyT], runnable),
            include_history=include_history,
            append_to_history=append_to_history,
            output_field=output_field or self._get_output_var_name("structured_prompt"),
        )

    def branch(
        self,
        condition: (
            Callable[[LastOutputType], bool]
            | Callable[[LastOutputType, RunState], bool]
        ),
        if_true: (
            Callable[
                ["_StatefulChainBuilder[ChainInputType, LastOutputType]"],
                "_StatefulChainBuilder[ChainInputType, T]",
            ]
            | None
        ) = None,
        if_false: (
            Callable[
                ["_StatefulChainBuilder[ChainInputType, LastOutputType]"],
                "_StatefulChainBuilder[ChainInputType, T]",
            ]
            | None
        ) = None,
        value_if_false: T | UndefinedType | None = UNDEFINED,
        value_if_true: T | UndefinedType | None = UNDEFINED,
        output_field: str | None = None,
        output_type: Type[T] | None = None,
    ) -> "_StatefulChainBuilder[ChainInputType, T]":
        """
        Branch the chain based on a condition.

        Args:
            condition: The condition to branch on. Input will be the output of the previous step in the builder
            then: A function that takes a new builder and returns the chain to run if the condition is true
            otherwise: A function that takes a new builder and returns the chain to run if the condition is false
        """

        def _merge_outputs(parent_state: RunState) -> Callable[[RunState], RunState]:
            def _merge(state: RunState) -> RunState:
                return {
                    **parent_state,
                    **state,
                    OUTPUTS_KEY: {
                        **parent_state[OUTPUTS_KEY],
                        **state[OUTPUTS_KEY],
                    },
                }

            return _merge

        def _route(state: RunState) -> Runnable[RunState, RunState]:
            sub_builder = _StatefulChainBuilder[ChainInputType, T](
                self.llm, "branch_" + self.prefix, initial_state=copy.deepcopy(state)
            )

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

            if _StatefulChainBuilder._call_with_output(condition, state):
                branch = _if_true(sub_builder)
            else:
                branch = _if_false(sub_builder)

            return (
                RunnableLambda(lambda _: state[INPUTS_KEY])
                | branch.build_raw()
                | RunnableLambda(_merge_outputs(state))
            )

        return self._append(
            runnable=RunnableLambda(_route),
            output_field=output_field or self._get_output_var_name("branch"),
            output_update_mode=OutputUpdateMode.FULL_STATE,
        )

    def run_lambda(
        self,
        _lambda: (
            Callable[[LastOutputType], T] | Callable[[LastOutputType, RunState], T]
        ),
        output_field: str | None = None,
        name: str | None = None,
    ) -> "_StatefulChainBuilder[ChainInputType, T]":
        """
        Add a step which calls a lambda function on the output of the previous step.

        Args:
            _lambda: The lambda function to call
        Returns:
            The current builder
        """
        output_field = output_field or self._get_output_var_name("lambda")

        def _lambda_with_output(state: RunState) -> T:
            return _StatefulChainBuilder._call_with_output(_lambda, state)

        _runnable: Runnable = RunnableLambda(_lambda_with_output, name=name)

        return self._append(
            runnable=_runnable,
            output_field=output_field,
            history_update_mode=HistoryUpdateMode.SKIP,
        )

    def run_passthrough_lambda(
        self,
        _lambda: (
            Callable[[LastOutputType], T] | Callable[[LastOutputType, RunState], T]
        ),
        name: str | None = None,
    ) -> "_StatefulChainBuilder[ChainInputType, LastOutputType]":
        def _wrap_with_state(state: RunState):
            return _StatefulChainBuilder._call_with_output(_lambda, state)

        _runnable: Runnable = RunnableLambda(_wrap_with_state, name=name)

        return self._append(
            runnable=_runnable,
            output_update_mode=OutputUpdateMode.SKIP,
            history_update_mode=HistoryUpdateMode.SKIP,
            output_field=None,
        )

    def run(self, inputs: ChainInputType | None = None) -> LastOutputType:
        output: LastOutputType = self.build().invoke(inputs)
        return output

    async def arun(self, inputs: ChainInputType | None = None) -> LastOutputType:
        output: LastOutputType = await self.build().ainvoke(inputs)
        return output

    async def abatch(
        self,
        inputs: AsyncIterator[ChainInputType],
        concurrency: int = 10,
        filter: Callable[[ChainInputType], Any] | None = None,
        pass_through_filtered: bool = True,
    ) -> AsyncIterator[LastOutputType]:
        queue = asyncio.Queue[ChainInputType | SentinelType](maxsize=concurrency)
        output_queue = asyncio.Queue[LastOutputType | SentinelType]()
        done_event = asyncio.Event()
        active_workers = [True] * concurrency
        chain = self.build()

        async def _worker(i: int):
            try:
                while True:
                    active_workers[i] = False
                    item = await queue.get()
                    if item == SENTINEL:
                        break
                    active_workers[i] = True
                    if not filter or filter(item):
                        output = await chain.ainvoke(item)
                        await output_queue.put(output)
                    elif pass_through_filtered:
                        await output_queue.put(cast(LastOutputType, item))
            except:
                active_workers[i] = False
                await output_queue.put(SENTINEL)
                raise

        async def _producer():
            try:
                async for item in inputs:
                    await queue.put(item)
            finally:
                for _ in range(concurrency):
                    await queue.put(SENTINEL)

                await output_queue.put(SENTINEL)
                done_event.set()

        workers = [asyncio.create_task(_worker(i)) for i in range(concurrency)]
        producer = asyncio.create_task(_producer())

        while (
            any(active_workers) or not output_queue.empty() or not done_event.is_set()
        ):
            if (result := await output_queue.get()) != SENTINEL:
                yield result

        await asyncio.gather(*workers, producer)

    def build(
        self, output_parser: Callable[[RunState], LastOutputType] | None = None
    ) -> Runnable[Any, LastOutputType]:
        """
        Build the chain into a Runnable object

        Args:
            output_variable: The variable to return from the chain (default is the output of the last step)
        Returns:
            The Runnable object
        """

        def _get_output(state: RunState) -> LastOutputType:
            value = state[OUTPUTS_KEY][state[OUTPUT_VAR_KEY]]

            if output_parser:
                value = output_parser(value)

            return cast(LastOutputType, value)

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

        def _get_output(state: RunState) -> dict[str, Any]:
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

    def build_raw(self) -> Runnable[ChainInputType, RunState]:
        """
        Build the chain with no output parser -- will return the full state dictionary.

        Returns:
            The Runnable object
        """
        return self.chain.with_config(run_name=self.name)

    def with_history(
        self, history: Sequence[MessageLikeRepresentation]
    ) -> "_StatefulChainBuilder[ChainInputType, LastOutputType]":
        """
        Sets the chat history to the provided messages.

        Args:
            history: The chat history
        Returns:
            The current builder
        """
        return cast(
            _StatefulChainBuilder[ChainInputType, LastOutputType],
            self.clear_history()._append_prompt(
                messages=ChatPromptTemplate.from_messages(history),
                append_to_history=True,
                include_history=False,
                update_output=False,
            ),
        )

    def _get_output_var_name(self, type_prefix: str = "output"):
        return f"_{self.prefix}_{type_prefix}_{str(uuid4())}"

    def _build_messages(
        self, messages: Sequence[MessageLikeRepresentation] | str
    ) -> ChatPromptTemplate:
        _messages: Sequence[MessageLikeRepresentation]

        if isinstance(messages, str):
            _messages = [HumanMessage(content=messages)]
        else:
            _messages = messages

        return ChatPromptTemplate.from_messages(_messages)

    @classmethod
    def _call_with_output(
        cls: Type["_StatefulChainBuilder"],
        _lambda: (
            Callable[[ChainInputType], T] | Callable[[ChainInputType, RunState], T]
        ),
        state: RunState,
    ) -> T:
        # if there is no last output, default to the input of the chain
        last_output = state[OUTPUTS_KEY].get(state[OUTPUT_VAR_KEY], state[INPUTS_KEY])
        args = [last_output, state]
        result = _lambda(*args[: _lambda.__code__.co_argcount])

        return result

    @staticmethod
    def _append_value(
        input: dict[str, Any], value: Any, fallback_key: str
    ) -> dict[str, Any]:
        if isinstance(value, dict):
            input.update(value)
        else:
            input[fallback_key] = value
        return input

    def _build_part_input(self) -> Runnable[RunState, LastOutputType]:
        def _select_output_field(state: RunState) -> LastOutputType:
            current_output_key = state[OUTPUT_VAR_KEY]

            if not current_output_key:
                return state[INPUTS_KEY]
            else:
                return state[OUTPUTS_KEY][current_output_key]

        return RunnableLambda(_select_output_field)

    def _build_prompt_input(
        self, prompt: ChatPromptTemplate, include_history: bool
    ) -> Runnable[RunState, list[BaseMessage]]:
        def _build_prompt(state: RunState) -> list[BaseMessage]:
            current_output_key = state[OUTPUT_VAR_KEY]
            prompt_params = {}

            if include_history:
                prompt_params[HISTORY_KEY] = state[HISTORY_KEY]

            if not current_output_key:
                prompt_params = _StatefulChainBuilder._append_value(
                    prompt_params, state[INPUTS_KEY], TMP_OUTPUT_KEY
                )
            else:
                prompt_params = _StatefulChainBuilder._append_value(
                    prompt_params,
                    state[OUTPUTS_KEY][current_output_key],
                    TMP_OUTPUT_KEY,
                )

            return prompt.invoke(prompt_params).to_messages()

        return RunnableLambda(_build_prompt)

    @classmethod
    def _inject_state(
        cls, initial_state: RunState | None = None
    ) -> Runnable[ChainInputType, RunState]:
        def _inject(inputs: ChainInputType) -> RunState:
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
        runnable: Runnable[list[BaseMessage], T],
        include_history: bool = True,
        append_to_history: bool = True,
        output_field: str | None = None,
    ) -> "_StatefulChainBuilder[ChainInputType, T]":
        prompt = self._append_prompt(
            messages,
            append_to_history=append_to_history,
            include_history=include_history,
        )
        return prompt._append(
            runnable=prompt._build_part_input() | runnable,
            output_field=output_field,
            history_update_mode=(
                HistoryUpdateMode.APPEND
                if append_to_history
                else HistoryUpdateMode.SKIP
            ),
        )

    def _update_history(
        self, messages: ChatPromptTemplate
    ) -> "_StatefulChainBuilder[ChainInputType, LastOutputType]":
        # cast is safe because we are not actually changing the output type
        return cast(
            _StatefulChainBuilder[ChainInputType, LastOutputType],
            self._append_prompt(
                messages=messages,
                append_to_history=True,
                include_history=True,
                update_output=False,
            ),
        )

    def _append_prompt(
        self,
        messages: ChatPromptTemplate,
        append_to_history: bool = True,
        include_history: bool = True,
        update_output: bool = True,
    ) -> "_StatefulChainBuilder[ChainInputType, list[BaseMessage]]":
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
            output_update_mode=(
                OutputUpdateMode.LAST_OUTPUT if update_output else OutputUpdateMode.SKIP
            ),
        )

    def _append(
        self,
        runnable: (
            Runnable[RunState, T] | Runnable[RunState, Runnable[RunState, RunState]]
        ),
        output_field: str | None,
        history_update_mode: HistoryUpdateMode = HistoryUpdateMode.SKIP,
        output_update_mode: OutputUpdateMode = OutputUpdateMode.LAST_OUTPUT,
    ) -> "_StatefulChainBuilder[ChainInputType, T]":
        step = StepRunnable(
            runnable, output_field, output_update_mode, history_update_mode
        )
        self.chain = self.chain | step
        return cast(_StatefulChainBuilder[ChainInputType, T], self)


class StatefulChainBuilder(Generic[T], _StatefulChainBuilder[T, T]):
    pass
