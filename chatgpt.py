import openai
import os
import tiktoken
import json

openai.api_key = os.environ['OPENAI_API_KEY']
MODEL = 'gpt-4o-mini'
TOKEN_ENCODING = tiktoken.encoding_for_model(MODEL)

class Model():
    def infer(self, parts, stream=False, options=None, regex=None, max_tokens=None, generation_kwargs={}):
        optional_args = {}
        if options:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "options",
                    "description": "Choose one of the given options",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "choice": {
                                "type": "string",
                                "description": "The chosen option",
                                "enum": options
                            },
                        },
                        "required": ["choice"],
                        "additionalProperties": False
                    }
                }
            }
            optional_args['response_format'] = response_format
        response = openai.chat.completions.create(model=MODEL, 
                                                  messages=parts,
                                                  max_tokens=max_tokens,
                                                  stream=stream,
                                                  **optional_args)
        if options:
            return json.loads(response.choices[0].message.content)['choice']
        if stream:
            body = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    c = delta.content
                    print(c, end='', flush=True)
                    body += c
            print()
            return body
        else:
            return response.choices[0].message.content

    def embed(self, text):
        return openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text)['data'][0]['embedding']

    def token_encode(self, text):
        return TOKEN_ENCODING.encode(text)

    def token_decode(self, tokens):
        return TOKEN_ENCODING.decode(tokens)
