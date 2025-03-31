prompt = """You are a world-class compliance expert with deep expertise in regulatory frameworks, corporate governance, anti-corruption measures, data privacy, and risk management. You have extensive knowledge of and experience with the rules and guidelines established by major payment networks, such as Visa and Mastercard. Your responses must be clear, detailed, and professionally structured.

IMPORTANT: When a question is posed in English, respond in English. If the question is posed in any other language, respond in that same language.

For the purpose of generating fine tuning examples, produce synthetic compliance queries and answers that are directly inspired by the structure, language, and content of Visa/Mastercard documents. Include examples that:

Extract or mimic the style and formatting found in such documents.
Require citing specific sections (e.g., 'Section 1.7.3 – Processing of Authorizations') and interpreting numerical data or tables similar to those in the attached document.
Present realistic scenarios where numerical data and tabular information are analyzed and referenced.
Your generated examples should serve as high-quality fine tuning data, enabling the model to understand and interpret complex compliance regulations with accurate citations and detailed analysis."""
temperature = .1
number_of_examples = 1

import os
from openai import OpenAI
import random
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI(api_key="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")

N_RETRIES = 3

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 8:
            prev_examples = random.sample(prev_examples, 8)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=1000,
    )

    return response.choices[0].message.content

# Generate examples
prev_examples = []
for i in range(number_of_examples):
    print(f'Generating example {i}')
    example = generate_example(prompt, prev_examples, temperature)
    prev_examples.append(example)

print(prev_examples)