import sqlite3
import argparse
import os
import json
import tempfile
import hashlib
import datetime
import logging

from colored import fore, back, style

import yaml
from numpy import dot
from numpy.linalg import norm

import saves

HISTORY = []
PROMPTS = []
CONNECTION = None

SAVES_CONNECTION = None
SAVE_NAME = "default"

REDUCERS = {}
DEBUG = False

import chatgpt
MODEL = chatgpt.Model()
# import ollama
# MODEL = ollama.Model()

logger = logging.getLogger(__name__)
if not os.path.exists("output"):
    os.mkdir('output')
logging.basicConfig(filename='output/log', level=logging.INFO)

DEFAULT_MAX_TOKENS = 300

def debug_print(msg):
    if DEBUG:
        print(msg)

def debug_log(msg):
    logger.info(msg)

def load_definition(game_name):
    filename = f"{game_name}.yaml"
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
        cursor.execute("CREATE TABLE prompts (value)")
        cursor.execute("CREATE TABLE embedding_cache (key, embedding)")
        CONNECTION.commit()
    return

def load_prompts_into_memory():
    cursor = CONNECTION.cursor()
    cursor.execute("SELECT value FROM prompts")
    global PROMPTS
    PROMPTS = [('deprecated', value, get_embedding(value)) for (value,) in cursor.fetchall()]

def store_prompt(connection, role, text, embedding = None):
    print(f"Writing {text[:100]} to database...")
    if not embedding:
        embedding = get_embedding(text)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO prompts VALUES (?, ?, ?)", (role, text, json.dumps(embedding)))
    connection.commit()


def get_embedding(text):
    hash = hashlib.md5(text.encode()).hexdigest()
    cursor = CONNECTION.cursor()
    cursor.execute("SELECT (embedding) FROM embedding_cache WHERE key = ?", (hash,))
    result = cursor.fetchone()
    if result:
        return json.loads(result[0])
    else:
        embedding = MODEL.embed(text)
        cursor.execute("INSERT INTO embedding_cache (key, embedding) VALUES (?, ?)", (hash, json.dumps(embedding)))
        CONNECTION.commit()
        return embedding


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
    return len(MODEL.token_encode(text))


def take_tokens(text, amount):
    return MODEL.token_decode(MODEL.token_encode(text)[:amount])


# Join list_of_strings into a list of strings comprising at most max_tokens tokens.
# if truncate=True, slice the last entry at the token limit
# otherwise, omit the last entry that would send us over the token limit.
def take_tokens_list(list_of_strings, max_tokens, truncate=True):
    output = []
    i = 0
    while len(list_of_strings) > i and max_tokens > 0:
        next_string = list_of_strings[i]
        next = MODEL.token_encode(next_string)
        if len(next) > max_tokens:
            if truncate:
                output.append(MODEL.token_decode(next[:max_tokens]))
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
    # load from save file
    reducers = saves.get_reducers(SAVES_CONNECTION, SAVE_NAME)
    if reducers:
        for reducer in reducers:
            REDUCERS[reducer["key"]] = {
                "value": reducer['value'],
                "budget": reducer['budget']
            }
    else:
        for key, reducer in config.get('reducers', {}).items():
            REDUCERS[key] = {
                'budget': reducer['every'],
                'value': reducer.get('initial', '')
            }

def initialize_history(config):
    history = saves.get_history(SAVES_CONNECTION, SAVE_NAME, 100)
    global HISTORY
    HISTORY = history

def tick_reducers(config, tokens, prompt):
    for key, reducer in config.get('reducers', {}).items():
        budget = REDUCERS[key]['budget']
        budget -= tokens
        REDUCERS[key]['budget'] = budget
        if budget < 0:
            # Update reducer
            debug_print(f"{fore.BLUE_VIOLET}Updating reducer {key}...")
            payload = build_payload(reducer['parts'], prompt)

            debug_log("inference: " + json.dumps(payload))

            response = MODEL.infer(payload, max_tokens=reducer['max_output'])
            message = response['choices'][0]['message']
            REDUCERS[key]['value'] = message['content']

            debug_log(f"New value for reducer {key}:")
            debug_log(f"{message['content']}")

            REDUCERS[key]['budget'] = reducer['every']
            if reducer.get('write_to_history'):
                push_history(message)
        saves.push_reducer(SAVES_CONNECTION, SAVE_NAME, key, REDUCERS[key]['value'], REDUCERS[key]['budget'])

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
    elif trim.startswith("!reducers"):
        print("{fore.MAGENTA}Current reducers:")
        for name, reducer in REDUCERS.items():
            print(f"{fore.RED}{name}:")
            print(f"{fore.DARK_ORANGE}{reducer['value']}{style.RESET}")
            print()
        return True
    elif trim.startswith("!query"):
        rest = " ".join(trim.split(" ")[1:])
        embedding = get_embedding(rest)
        print(f"Getting top prompts for {rest}...")
        results = sort_similarity(PROMPTS, embedding)
        print("\n===\n".join(f"{r}\nSimilarity:{cosine_similarity(e, embedding)}" for (_, r, e) in results[-3:]))
        return True
    return False

def push_history(entry):
    HISTORY.append(entry)
    saves.push_history(SAVES_CONNECTION, SAVE_NAME, entry['role'], entry['content'])


def main(game_name, mode):
    y = load_definition(game_name)
    db = load_database(y)
    load_prompts_into_memory()
    if mode not in y['modes']:
        raise Exception(f"No mode named {mode} found for this adventure.")
    loop_config = y['modes'][mode]
    initialize_reducers(loop_config)
    initialize_history(loop_config)
    print(loop_config.get('introduction', "Let's start playing the game!"))
    # Print the history if there is any
    if len(HISTORY) > 0:
        print("Continuing from last time...")
        for entry in HISTORY[-10:][::-1]:
            if entry['role'] == 'user':
                print(f"{fore.GREEN}> {entry['content']}")
            else:
                print(f"{fore.WHITE}{entry['content']}")
    while True:
        prompt = input(f"{fore.GREEN}>")
        print(style.RESET)
        if do_commands(prompt):
            continue
        else:
            payload = build_payload(loop_config['parts'], prompt)

            debug_log("Main inference: " + json.dumps(payload))

            response = MODEL.infer(payload, max_tokens=loop_config.get('max_tokens', DEFAULT_MAX_TOKENS))

            debug_log(f"Inference Cost: {response['usage']['prompt_tokens']} prompt tokens. {response['usage']['total_tokens']} total.")

            push_history({"role": "user", "content": prompt})
            push_history(response['choices'][0]['message'])

            response = response['choices'][0]['message']['content']
            print(f"{fore.WHITE}{response}{style.RESET}")

            used_tokens = num_tokens("\n".join(h['content'] for h in HISTORY[-2:]))

            tick_reducers(loop_config, used_tokens, prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='main', help="Which loop to execute")
    parser.add_argument("game", type=str, help="Which adventure file to load")
    parser.add_argument("--debug", action='store_true', help="Whether or not to start with debug mode on")
    parser.add_argument("--save-name", type=str, help="Which named save file to use")
    parser.add_argument("--continue", dest="cont", action="store_true", help="Continue with the most recently used save file.")

    args = parser.parse_args()

    DEBUG = args.debug

    SAVES_CONNECTION = saves.open_save_db(args.game)

    if args.save_name and args.cont:
        raise Exception("Use either '--continue' or '--save-name', not both")

    default_save_name = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    save_name = args.cont and saves.get_most_recent_save(SAVES_CONNECTION) or args.save_name or default_save_name
    print(f"Using save file {save_name}")
    if not save_name:
        raise Exception("Couldn't find save with that name")

    SAVE_NAME = save_name

    main(args.game, args.mode)
