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