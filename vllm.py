import os
import requests
import json

import tiktoken
# TODO: What tokenizer does llama3 use?
TOKEN_ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo')

RUNPOD_API_KEY=os.environ['RUNPOD_API_KEY']
SERVERLESS_ID='vllm-tbrjgsydxfykv4'
HOST=f'https://api.runpod.ai/v2/{SERVERLESS_ID}/openai/v1'
MODEL='neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8'

class Model():
    def infer(self, parts, stream=False, options=None, regex=None, max_tokens=None, generation_kwargs={}):
        url = f"{HOST}/chat/completions"
        response = requests.post(url, json={
            "model": MODEL,
            # "raw": True,
            "messages": parts,
            "max_tokens": max_tokens,
            "stream": stream,
            **generation_kwargs,
            **({"guided_choice": options} if options else {}),
            **({"guided_regex": regex} if regex else {}),
        }, headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        }, stream=stream)
        response.raise_for_status()
        if stream:
            output = ''
            for line in response.iter_lines():
                decoded = line.decode('utf-8')
                if decoded.startswith('data: ') and not decoded.endswith('[DONE]'):
                    decoded = decoded.lstrip('data: ')
                    js = json.loads(decoded)
                    chunk = js['choices'][0]['delta']['content'] or ''
                    print(chunk, end='', flush=True)
                    output += chunk
            print()
            return output
        else:
            return response.json()['choices'][0]['message']['content']

    def embed(self, text):
        url = f"{HOST}/embeddings"
        print(url)
        response = requests.post(url, json={
            "model": MODEL,
            "raw": True,
            "prompt": text,
            "stream": False
        }, headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}"
        })
        response.raise_for_status()
        print(response.json())
        return response.json()['embedding']

    def token_encode(self, text):
        return TOKEN_ENCODING.encode(text)

    def token_decode(self, tokens):
        return TOKEN_ENCODING.decode(tokens)
