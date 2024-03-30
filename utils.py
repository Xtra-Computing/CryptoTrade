import os
import sys
from openai import OpenAI

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
from tenacity import (
    retry,
    stop_after_attempt, 
    wait_random_exponential, 
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Model = Literal["gpt-4", "gpt-3.5-turbo"]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt, model, seed, temperature=0.0, max_tokens=256, stop_strs=None, is_batched=False, debug=False):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(model=model,
    messages=messages,
    max_tokens=max_tokens,
    
    seed=seed,
    temperature=temperature)
    if debug:
        print(response.system_fingerprint)
    return response.choices[0].message.content
