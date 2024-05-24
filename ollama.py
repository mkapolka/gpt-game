import requests

import tiktoken
# TODO: What tokenizer does llama3 use?
TOKEN_ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo')

HOST='http://192.168.0.143:11434'
MODEL='llama3'

# Translate from the OpenAI format to a llama3 payload
def _openai_prompt_to_llama(parts):
    turns = "<|eot_id|>".join([
        f"<|start_header_id|>{part['role']}<|end_header_id|>{part['content']}"
        for part in parts
    ])

def _llama_response_to_openai(repsonse):
    return {
        "choices": [
            {
                "message": {
                    "content": response['response']
                }
            }
        ],
        'usage': {
            'prompt_tokens': 0
        }
    }


class Model():
    def infer(self, parts, max_tokens=None):
        url = f"{HOST}/api/generate"
        response = requests.get(url, json={
            "model": MODEL,
            "raw": True,
            "prompt": _translate_format(parts),
            "stream": False
        })
        return _llama_response_to_openai(response.json())

    def embed(self, text):
        url = f"{HOST}/api/embeddings"
        response = requests.get(url, json={
            "model": MODEL,
            "raw": True,
            "prompt": text,
            "stream": False
        })
        return _llama_response_to_openai(response.json())

    def token_encode(self, text):
        return TOKEN_ENCODING.encode(text)

    def token_decode(self, tokens):
        return TOKEN_ENCODING.decode(tokens)
