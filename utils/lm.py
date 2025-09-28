# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import openai
from together import Together
from dotenv import load_dotenv

'''
NOTE: 
    Available functions:
        - call_together_api: using TogetherAI hosted models
        - call_vllm_api: using vllm self-served models (기존 함수 유지)
        - openai_generate: using openai models
'''

load_dotenv()
########################################################################################################
def custom_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512):
    raise NotImplementedError()


def generate(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):
    # return custom_api(prompt, model, temperature, top_p, max_tokens, port)
    # return call_vllm_api(prompt, model, temperature, top_p, max_tokens, port, i)
    return call_together_api(prompt, model, temperature, top_p, max_tokens)

together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

together_model_map = {
    'meta-llama/Llama-3.1-405B-Instruct-FP8': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
    'meta-llama/Llama-3.3-70B-Instruct': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'meta-llama/Llama-3.1-70B-Instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    'meta-llama/Llama-3.1-8B-Instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistralai/Mistral-Nemo-Instruct-2407': 'mistralai/Mistral-Nemo-Instruct-2407'
}


########################################################################################################

def call_together_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512):
    try:
        together_model = together_model_map.get(model, model)

        response = together_client.chat.completions.create(
            model=together_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"TogetherAI API 호출 실패: {e}")
        return ""


def call_vllm_api(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):
    CUSTOM_SERVER = "0.0.0.0"

    model_map = {
        'meta-llama/Llama-3.1-405B-Instruct-FP8': {'name': 'llama3.1_405B',
                                                   'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
        'meta-llama/Llama-3.3-70B-Instruct': {'name': 'llama3.3_70B',
                                              'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
        'meta-llama/Llama-3.1-70B-Instruct': {'name': 'llama3.1_70B',
                                              'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
        'meta-llama/Llama-3.1-8B-Instruct': {'name': 'llama3.1_8B',
                                             'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
        'mistralai/Mistral-7B-Instruct-v0.2': {'name': 'mistral7B',
                                               'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]},
        "mistralai/Mistral-Nemo-Instruct-2407": {'name': 'Mistral-Nemo-Instruct-2407',
                                                 'server_urls': [f"http://{CUSTOM_SERVER}:8000/v1"]}
    }

    if port == None:
        port = model_map[model]["server_urls"][i]

    client = openai.OpenAI(
        base_url=f"{port}",
        api_key="NOT A REAL KEY",
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return chat_completion.choices[0].message.content


def openai_generate(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512):
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_KEY"],
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return chat_completion.choices[0].message.content
