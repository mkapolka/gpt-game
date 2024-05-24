import openai
import os
import tiktoken

openai.api_key = os.environ['OPENAI_API_KEY']
MODEL = 'gpt-3.5-turbo'
TOKEN_ENCODING = tiktoken.encoding_for_model(MODEL)

class Model():
    def infer(self, parts, max_tokens=None):
        return openai.ChatCompletion.create(model=MODEL, 
                                            messages=parts,
                                            max_tokens=max_tokens)

    def embed(self, text):
        return openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text)['data'][0]['embedding']

    def token_encode(self, text):
        return TOKEN_ENCODING.encode(text)

    def token_decode(self, tokens):
        return TOKEN_ENCODING.decode(tokens)
