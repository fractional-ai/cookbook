# Classification w/Confidence Scores
Use OpenAI's structured outputs + logprobs to classify a string w/confidence scores.

## Example
Suppose we have news article headlines that we want to categorize into one of several categories:

1. Define your categories as an Enum
    ```python
      class ArticleType(str, Enum):
          SPORTS = "Sports"
          POLITICS = "Politics"
          BUSINESS = "Business"
          TECHNOLOGY = "Technology"
          ENTERTAINMENT = "Entertainment"
          HEALTH = "Health"
          SCIENCE = "Science"
          # ... etc
    ```
2. Pass your input and the enum type to the `classify_with_confidence` function:
    ```python
      openai_client = OpenAI()

      classifications = classify_with_confidence(
        "Scientific breakthrough improves football performance",
        ArticleType,
        openai_client
    )
    ```
3. Now you have an ordered dictionary of classifications with confidence scores:
    ```python
      {
        ArticleType.SCIENCE: 0.5311,
        ArticleType.SPORTS: 0.4687,
      }
    ```

## Options
There are a few options that can be tweaked to improve results:
```python
  classify_with_confidence(
    "Scientific breakthrough improves football performance",
    ArticleType,
    openai_client,

    # Increase/decrease the maximum number of categories returned
    # In practice OpenAI returns a lot of duplicates, especially when confidence is high, so these 10 might still result in 2 unique categorizations
    max_categories=10,

    # Must be a model that supports structured output
    model='gpt-4o-2024-08-06',

    system_prompt="You are an alien who likes to categorize articles but doesn't know about earthling preferences.",
)
```