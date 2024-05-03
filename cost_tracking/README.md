# Cost Tracking

The [cost_tracking.py](./cost_tracking.py) file contains a few helpers to track and manage costs:

## `checkpoint_cost_tracking`

This is a context manager that allows tracking costs within designated sections of code.

```python
@cost_checkpoint("thing1")
def thing1():
    # Some really expensive LLM call here

def thing2():
    with set_cost_checkpoint("thing2"):
        # your code here

@cost_checkpoint("main")
def main():
    thing1()
    thing2()

with checkpoint_cost_tracking() as cb:
    main()
    print(cb)
    print(cb.dict())
```

output will look like this:

```
Model Costs:
  gpt-4-turbo-2024-04-09 = $0.1116
  gpt-3.5-turbo-0125 = $0.0034
Step Cost:
  auth_detection = $0.0025 (2.21%)
  stream_parameter_detection = $0.0976 (84.81%)
  pagination = $0.0088 (7.61%)
** Total Cost: $0.1151
```

`cb.dict()` will return a dictionary with the same information.

There is currently no structure to the checkpoints -- all LLM calls within the context of the checkpoint are tracked. If a checkpointed method calls other checkpointed methods, the results will be accurate but the relationship between them will not be reflected.

## `enforce_budget`

This will create a background thread that monitors costs in real time and hard-quits the process using `os._exit` if a budget is exceeded:

```python
with checkpoint_cost_tracking() as cb:
    async with enforce_budget(0.001, cb):
        # your code here
```