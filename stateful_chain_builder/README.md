# StatefulChainBuilder

A langchain wrapper that makes writing semi-complex chains slightly easier.

## Use Cases 

### Chain-of-thought use with a tool

Sometimes it's useful to let an LLM think out loud before using a tool. Constraining the output too early can make for dumb answers. The flow is:

1. Set up context (like you normally would)
2. Ask the LLM to solve the problem, allowing it to think out loud and use words
3. Then force tool use to get structured output

Here's a toy example:

```python
class Ingredient(BaseModel):
    name: str
    amount: float
    unit: str

class Recipe(BaseModel):
    dish: str
    ingredients: list[Ingredient]
    steps: list[str]

builder = (
    StatefulChainBuilder[dict](ChatOpenAI(model="gpt-3.5-turbo"))
        .system("""
Tell me how to cook the provided dish. 

List concise steps and ingredients we'll need. 

For each ingredient, include a number and a unit.
        """)
        # Output for this prompt will automatically be saved in history and included
        # in subsequent calls.
        .prompt(messages=[("user", "I would like a {dish} please.")])
        .structured_prompt(Recipe)
)

builder.run({"dish": "PB & J"})
```

![alt text](chain-of-thought.png)

### Branching

Writing branching chains can be kind of cumbersome in langchain. It's a little easier with this thing:

```python
class RandomNumber(BaseModel):
    number: int

builder = (
    StatefulChainBuilder(ChatOpenAI(model="gpt-3.5-turbo"))
        .structured_prompt(RandomNumber, "please pick a random number greater than 10")
        .branch(
            condition=lambda x: x.number > 10,
            # the argument here is a cloned instance of the current state
            # of the builder. steps will receive the output from the 
            # step immediately before the branch.
            if_true=lambda b: b.run_lambda(lambda x: x.number / 2),
            if_false=lambda b: (
                b.structured_prompt(RandomNumber, "between 1 and 1000 please")
                .run_lambda(lambda x: x.number / 2)
            )
        )
        .branch(
            condition=lambda x: x > 10,
            # You can also yield constant values with `value_if_true` 
            # or `value_if_false`.
            value_if_true=123,
            if_false=lambda b: b.run_lambda(lambda x: x * 2)
        )
        .run_lambda(lambda x: (
            x + 1
        ))
)

builder.run()
```

![alt text](branching.png)

## Syntax

### Prompt chaining

The following will embed the entire history into each prompt, simulating a ChatGPT-like interface:

```python
builder = (
    StatefulChainBuilder[str](ChatOpenAI(model="gpt-3.5-turbo"))
    .system("You're a friendly assistant")
    .run_lambda(lambda x: {"input": x})
    .prompt(
        # you can use prompt variables by using the `messages` argument
        messages=[("user", "here's my question: {input}")], 
        # you can name the output of this step if you want to use it later
        output_field="wordy_answer")
    .prompt(
        "please reword to be more concise", 
        output_field="concise_answer")
)

# This will run the chain
final_response = builder.run("how do i get to the moon?")
# > Traveling to the moon is currently only possible through space missions conducted by space agencies or private companies. Commercial space tourism programs may offer moon trips in the future, but they will likely be expensive and require preparation.

# this will get a chain that you can call `invoke` or `batch` on
# `.run` uses this internally:
chain = builder.build()
responses = chain.batch(["is the sky green?", "how far away is the moon?"])
#> ['The sky is usually blue, not green.',
#   "The average distance from Earth to the moon is approximately 238,855 miles (384,400 kilometers), but it can vary due to the moon's elliptical orbit."]

# this will build a chain similar to RunnablePassthrough.assign where the provided outputs are present in the input dictionary
passthrough_chain = builder.build_passthrough("concise_answer", "wordy_answer")
passthrough_chain.invoke({"question": "write a sentence about how neat the moon is"})
#> {'question': 'write a sentence about how neat the moon is',
#   'concise_answer': 'The moon is incredibly neat.',
#   'wordy_answer': 'The moon is a mesmerizing celestial body that never fails to captivate with its beauty and mystery.'}
```

### Async iterators

You can use `abatch` to take in an `AsyncIterator` of inputs and emit an `AsyncGenerator` of outputs:

```python
builder = (
    StatefulChainBuilder[str](ChatOpenAI(model="gpt-3.5-turbo"))
    .system("You're a friendly assistant")
    .run_lambda(lambda x: x + 1)
)

async def inputs()
    for i in range(10):
        yield i

outputs = builder.abatch(inputs())

async for o in outputs:
    print(o)

# yields:
#> 8
#  1
#  ...
```

This is useful if you're building a streaming pipeline.

### Calling lambdas

The first argument to a lambda passed to `run_lambda` will always be the output of the previous step.

Sometimes you may want to access intermediate outputs or even the original input to the chain.

If you specify a lambda with two arguments, the current state of the run will be passed to the second argument. Run state has the following shape:

```python
run_state: dict[str, Any] = {
  # will always contain the original input to the chain, exactly as passed in
  "inputs": ..., 

  # a dictionary of outputs from all previous steps
  # unlabeled outputs will have arbitrary prefixes beginning with "_".
  "outputs": {
    "_output_1": null
  },

  # a history of prompts to and responses from LLM calls
  # note that inputs/outputs to lambdas are NOT logged.
  "history": [],
}
```

Example using the run state dictionary:

```python
import random
from dataclasses import dataclass

@dataclass
class CustomInputType:
    number: int

builder = (
    StatefulChainBuilder[CustomInputType](ChatOpenAI(model="gpt-3.5-turbo"))
    # inject a random number
    .run_lambda(lambda _: (random.randint(1, 100)))
    .run_lambda(lambda x: x + 1, output_field="output1")
    # if a lambda with two arguments is given, the second argument will be the
    # RunState dict
    .run_lambda(lambda x, state: x * state['inputs'].number)
    # output from previous steps can be accessed in the 'outputs' dict
    .run_lambda(lambda x, state: x + state['outputs']['output1'])
)

# you can use `build_raw` to get a chain that outputs the final RunState
builder.build_raw().invoke(CustomInputType(number=50))
# > {'inputs': CustomInputType(number=50),
#    'history': [],
#    'outputs': {'__lambda_14d6c4e5-abe0-4bcb-af6d-59705dfeec13': 73,
#     'output1': 74,
#     '__lambda_0fe3f32d-a14d-4e22-b987-e7627eda6fff': 3700,
#     '__lambda_89862b0a-940e-42bd-a54f-e08444b3c4ff': 3774},
#    'last_output_key': '__lambda_89862b0a-940e-42bd-a54f-e08444b3c4ff',
#    'tmp_output': 3774}
```