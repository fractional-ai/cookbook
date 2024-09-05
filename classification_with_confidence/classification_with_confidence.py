from openai import OpenAI
from enum import Enum
from typing import TypeVar, Dict, Tuple
import pydantic
import numpy as np
from openai.types.chat import ParsedChatCompletion, ParsedChoice

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that classifies items into categories. You will be given an item to classify, respond with its category. Respond with the appropriate classification."

ClassificationCategory = TypeVar("ClassificationCategory", bound=Enum)


def parse_and_extract_confidence(
    choice: ParsedChoice, enum_type: type[ClassificationCategory]
) -> Tuple[ClassificationCategory, float]:
    """
    Parses a choice (single response item from OpenAI) and extracts the typed classification and its confidence.
    """
    classification = enum_type(choice.message.parsed.type)

    # We have the log probability of each token in the classification. Use them to calculate the probability of the classification
    token_logprobs = [logprobs.logprob for logprobs in choice.logprobs.content]
    probability = np.exp(sum(token_logprobs))

    return classification, probability


def extract_classifications_with_confidence(
    openai_response: ParsedChatCompletion, enum_type: type[ClassificationCategory]
) -> Dict[ClassificationCategory, float]:
    """
    Extracts classifications with confidence from OpenAI response.
    """
    # Create a map where keys are the classifications and values are the probabilities
    # NOTE: OpenAI returns some duplicates (esp when confidence is high), so this will also serve to deduplicate by key
    classification_map: Dict[ClassificationCategory, float] = {}
    for choice in openai_response.choices:
        classification, probability = parse_and_extract_confidence(choice, enum_type)
        classification_map[classification] = probability

    # Sort the map in descending order of probability
    return dict(
        sorted(classification_map.items(), key=lambda item: item[1], reverse=True)
    )


def classify_with_confidence(
    input: str,
    enum_type: type[ClassificationCategory],
    client: OpenAI,
    max_categories: int = 5,
    *,
    model: str = "gpt-4o-2024-08-06",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[ClassificationCategory, float]:
    """
    Given an input string, classifies it into one of the categories provided. Generates multiple options when appropriate and ties each to a confidence score.

    NOTE: The higher the confidence, the fewer results we'll return.
    """
    # OpenAI can't respond directly w/an Enum. We create a wrapper Pydantic model that contains exactly one field called "type" whose type is the enum we're provided.
    classification_type = pydantic.create_model(
        "ClassificationType", type=(enum_type, ...)
    )

    system_prompt += f"\n\nRespond with JSON. Choose only from the following options: {', '.join(enum_type)}"

    resp = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input},
        ],
        response_format=classification_type,
        logprobs=True,
        n=max_categories,
    )

    return extract_classifications_with_confidence(resp, enum_type)


if __name__ == "__main__":

    class ArticleType(str, Enum):
        SPORTS = "Sports"
        POLITICS = "Politics"
        BUSINESS = "Business"
        TECHNOLOGY = "Technology"
        ENTERTAINMENT = "Entertainment"
        HEALTH = "Health"
        SCIENCE = "Science"
        WORLD = "World"
        LOCAL = "Local"
        EDUCATION = "Education"
        ENVIRONMENT = "Environment"
        LIFESTYLE = "Lifestyle"
        CRIME = "Crime"
        TRAVEL = "Travel"
        OPINION = "Opinion"
        FASHION = "Fashion"
        FOOD = "Food"
        AUTOMOTIVE = "Automotive"
        REAL_ESTATE = "Real Estate"
        WEATHER = "Weather"

    classifications = classify_with_confidence(
        "scientific breakthrough improves football performance",
        ArticleType,
        OpenAI(),
        max_categories=10,
    )

    # Print output. Expect it to look something like this:
    #   Science: 53.11%
    #   Sports: 46.87%
    for classification, probability in classifications.items():
        print(f"{classification.value}: {np.round(probability * 100, 2)}%")
