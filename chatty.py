import sqlite3
import argparse
import os
import json
import tempfile

import yaml
import openai
import tiktoken
from numpy import dot
from numpy.linalg import norm

openai.api_key = os.environ['OPENAI_API_KEY']
MODEL = 'gpt-3.5-turbo'

HISTORY = []
PROMPTS = []
CONNECTION = None
REDUCERS = {}
TOKEN_ENCODING = tiktoken.encoding_for_model(MODEL)
DEBUG = False


def debug_print(msg):
    if DEBUG:
        print(msg)


def load_definition(filename):
    with open(filename, 'r') as f:
        y = yaml.load(f, yaml.Loader)
    y['path'] = os.path.abspath(filename)
    return y


def load_database(y):
    global CONNECTION

    db_file = os.path.splitext(y['path'])[0] + '.sqlite'
    print(f"Opening DB file {db_file}")
    to_create = not os.path.exists(db_file)
    CONNECTION = sqlite3.connect(db_file)
    cursor = CONNECTION.cursor()
    if to_create:
        print(f"Initializing database {db_file}...")
        cursor.execute("CREATE TABLE prompts (role, value, embedding)")
        CONNECTION.commit()
    return

def load_prompts_into_memory():
    cursor = CONNECTION.cursor()
    cursor.execute("SELECT role, value, embedding FROM prompts")
    global PROMPTS
    PROMPTS = [(role, value, json.loads(embedding)) for (role, value, embedding) in cursor.fetchall()]

def store_prompt(connection, role, text, embedding = None):
    print(f"Writing {text[:100]} to database...")
    if not embedding:
        embedding = get_embedding(text)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO prompts VALUES (?, ?, ?)", (role, text, json.dumps(embedding)))
    connection.commit()


def get_embedding(text):
    return openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text)['data'][0]['embedding']


def cosine_similarity(a,b):
    return dot(a, b) / (norm(a) * norm(b))


def sort_similarity(prompts, embedding):
    return sorted(prompts, key=lambda p: cosine_similarity(p[2], embedding))


# For supporting the legacy format. Might get rid of this.
def drive_file(filename, cursor):
    with open(filename, 'r') as f:
        contents = f.read()
    entries = contents.split("@@@")
    for entry in entries:
        print(f"Writing {entry[:100]} to database...")
        store_prompt(cursor, "user", entry)
    print("Files driven.")


def num_tokens(text):
    return len(TOKEN_ENCODING.encode(text))


def take_tokens(text, amount):
    return TOKEN_ENCODING.decode(TOKEN_ENCODING.encode(text)[:amount])


# Join list_of_strings into a list of strings comprising at most max_tokens tokens.
# if truncate=True, slice the last entry at the token limit
# otherwise, omit the last entry that would send us over the token limit.
def take_tokens_list(list_of_strings, max_tokens, truncate=True):
    output = []
    i = 0
    while len(list_of_strings) > i and max_tokens > 0:
        next_string = list_of_strings[i]
        next = TOKEN_ENCODING.encode(next_string)
        if len(next) > max_tokens:
            if truncate:
                output.append(TOKEN_ENCODING.decode(next[:max_tokens]))
            else:
                break
        else:
            output.append(next_string)
        i += 1
        max_tokens -= len(next)
    return output

def take_history(max_tokens, truncate=True):
    history_reversed = HISTORY[::-1]
    strings = [h['content'] for h in history_reversed]
    result = take_tokens_list(strings, max_tokens, truncate)
    return [
        {
            "role": history_reversed[i]['role'],
            "content": s
        } for i, s in enumerate(result)
    ][::-1] # re-reverse




def build_part(part_config, prompt):
    type = part_config['type']
    if type == 'const':
        return [{
            "role": part_config.get('role', 'user'),
            "content": part_config['value']
        }]
    elif type == 'search':
        context = []
        use_prompt = part_config['query'].get('prompt')
        use_history = part_config['query'].get('history')
        truncate_results = part_config.get('truncate', True)
        amount = part_config['amount']
        if use_history:
            context.extend("\n".join(c['content'] for c in take_history(use_history)))
        if use_prompt:
            context.append(prompt)
        embedding = get_embedding("\n".join(context))
        results = [value for (_, value, _) in sort_similarity(PROMPTS, embedding)]
        results.reverse()
        payload = take_tokens_list(results, amount, truncate_results)
        return [{
            "role": part_config.get('role', 'user'),
            "content": "\n".join(payload)
        }]
    elif type == "history":
        return take_history(part_config['amount'])
    elif type == "prompt":
        return [{
            "role": part_config.get('role', 'user'),
            "content": prompt
        }]
    elif type == "reducer":
        return [{
            "role": part_config.get('role', 'user'),
            "content": REDUCERS[part_config['reducer']]['value']
        }]
    else:
        raise Exception("Unknown part type: {type}")


def build_payload(parts, prompt):
    return [p for part_config in parts
                for p in build_part(part_config, prompt)]

def initialize_reducers(config):
    for key, reducer in config.get('reducers', {}).items():
        REDUCERS[key] = {
            'budget': reducer['every'],
            'value': reducer.get('initial', '')
        }

def tick_reducers(config, tokens, prompt):
    for key, reducer in config.get('reducers', {}).items():
        budget = REDUCERS[key]['budget']
        budget -= tokens
        REDUCERS[key]['budget'] = budget
        if budget < 0:
            # Update reducer
            debug_print(f"Updating reducer {key}...")
            payload = build_payload(reducer['parts'], prompt)
            response = openai.ChatCompletion.create(model=MODEL, 
                                                    messages=payload,
                                                    max_tokens=reducer['max_output'])
            message = response['choices'][0]['message']
            REDUCERS[key]['value'] = message['content']
            REDUCERS[key]['budget'] = reducer['every']
            if reducer.get('write_to_history'):
                HISTORY.append(message)

def edit_string(s):
    t = tempfile.NamedTemporaryFile()
    with open(t.name, 'w') as f:
        f.write(s)
    os.system("vim %s" % t.name)
    with open(t.name, 'r') as f:
        f.seek(0)
        return f.read()
    t.close()

def do_commands(prompt):
    trim = prompt.strip()
    if trim == "!save":
        store_prompt(CONNECTION, "assistant", HISTORY[-1]['content'])
        return True
    elif trim == "!debug":
        global DEBUG
        print(f"{DEBUG and 'dis' or 'en'}abling debug mode")
        DEBUG = not DEBUG
        return True
    elif trim == '!edit':
        print("Editing last response, then saving to DB")
        entry = HISTORY[-1]['content']
        edited = edit_string(entry)
        store_prompt(CONNECTION, "assistant", edited)
        return True
    return False


def main(filename, mode):
    y = load_definition(filename)
    db = load_database(y)
    load_prompts_into_memory()
    initialize_reducers(y)
    loop_config = y['loops'][mode]
    print(loop_config.get('introduction', "Let's start playing the game!"))
    while True:
        prompt = input(">")
        if do_commands(prompt):
            continue
        else:
            payload = build_payload(loop_config['parts'], prompt)

            debug_print(json.dumps(payload, indent=2))
            response = openai.ChatCompletion.create(model=MODEL, messages=payload, max_tokens=300)

            debug_print("=== DEBUG ===")
            debug_print(f"Cost: {response['usage']['prompt_tokens']} prompt tokens. {response['usage']['total_tokens']} total.")
            HISTORY.append({"role": "user", "content": prompt})
            HISTORY.append(response['choices'][0]['message'])
            print(response['choices'][0]['message']['content'])
            used_tokens = num_tokens("\n".join(h['content'] for h in HISTORY[-2:]))
            tick_reducers(y, used_tokens, prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='main', help="Which loop to execute")
    parser.add_argument("file", type=str, help="Which adventure file to load")
    parser.add_argument("--debug", action='store_true', help="Whether or not to start with debug mode on")

    args = parser.parse_args()

    main(args.file, args.mode)
