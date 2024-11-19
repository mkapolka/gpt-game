import requests
import json

import tiktoken
# TODO: What tokenizer does llama3 use?
TOKEN_ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo')

HOST='http://192.168.0.143:11434'
MODEL='llama3'

# Translate from the OpenAI format to a llama3 payload
def _openai_prompt_to_llama(parts):
    return "<|begin_of_text|>" + "<|eot_id|>".join([
        f"<|start_header_id|>{part['role']}<|end_header_id|>{part['content']}"
        for part in parts
    ]) + "<|eot_id|><|start_header_id|>assistant<end_header_id|>"

def _openai_prompt_to_llama_2(parts):
    return "\n".join(f"### {part['role']}:\n{part['content']}" for part in parts)

def _llama_response_to_openai(response):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response['response']
                }
            }
        ],
        'usage': {
            'prompt_tokens': 0,
            'total_tokens': 0,
        }
    }


class Model():
    def infer(self, parts, stream=False, options=None, regex=None, generation_kwargs={}, max_tokens=None):
        url = f"{HOST}/api/generate"
        prompt = _openai_prompt_to_llama_2(parts)
        response = requests.post(url, json={
            "model": MODEL,
            # "raw": True,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "stop": ["user:"],
                "frequency_penalty": generation_kwargs.get('frequency_penalty', 1.0)
            }
        }, stream=stream)
        response.raise_for_status()
        if stream:
            output = ''
            for line in response.iter_lines():
                decoded = line.decode('utf-8')
                js = json.loads(decoded)
                chunk = js['response']
                print(chunk, end='', flush=True)
                output += chunk
            print()
            return output
        return _llama_response_to_openai(response.json())

    def embed(self, text):
        url = f"{HOST}/api/embeddings"
        response = requests.post(url, json={
            "model": MODEL,
            "raw": True,
            "prompt": text,
            "stream": False
        })
        response.raise_for_status()
        return response.json()['embedding']

    def token_encode(self, text):
        return TOKEN_ENCODING.encode(text)

    def token_decode(self, tokens):
        return TOKEN_ENCODING.decode(tokens)
