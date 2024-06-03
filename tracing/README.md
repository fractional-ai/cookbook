# Tracing

The [tracing.py](./tracing.py) file contains a few method to help selectively manage langchain tracing.

The primary function this serves over the langchain primitives is selectively enabling or disabling tracing for a function using an annotation ([existing langchain annotations](https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code) require turning on `LANGCHAIN_TRACING_V2`, which enables tracing globally).

## Usage

#### Enable tracing

```python3
@configure_tracing("My Function")
def my_function():
   ...
```

#### Disable tracing (when enabled globally or higher in the call chain)

```python3
@configure_tracing(enabled=False)
def my_untraced_function():
   ...
```