import argparse
import yaml
import datetime
import readline
from concurrent.futures import ThreadPoolExecutor

STATE = {}
GAME = None

MODEL = None

THREAD_POOL = None
ARGS = None

def init_model(config):
    model_name = config['name']
    global MODEL
    if model_name == 'chatgpt':
        import chatgpt
        MODEL = chatgpt.Model()
    if model_name == 'ollama':
        import ollama
        MODEL = ollama.Model()
    if model_name == 'vllm':
        import vllm
        MODEL = vllm.Model()


def load_file(filename):
    with open(filename, 'r') as f:
        y = yaml.load(f, yaml.Loader)
    return y

def init_state(game_def):
    global GAME
    global STATE
    GAME = game_def
    STATE = {
        "THINGS": {},
        "HISTORY": [],
        "INPUT": ""
    }
    for id, thing in game_def['things'].items():
        t = thing['type']
        if t == 'reducer':
            initial_value = thing.get('initial', '')
            STATE['THINGS'][id] = {
                'last_run': 0,
                'value': initial_value
            }
        elif t == 'chat':
            initial_value = thing.get('initial', '')
            STATE['THINGS'][id] = {
                'value': initial_value
            }
        elif t == 'state_machine':
            initial_state = thing['initial_state']
            STATE['THINGS'][id] = {
                'state': initial_state
            }

def get_thing(name):
    return STATE['THINGS'][name]


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


def take_history(history, max_tokens, truncate=True):
    history = [h['message'] for h in history]
    history_reversed = history[::-1]
    strings = [h['content'] for h in history_reversed]
    result = take_tokens_list(strings, max_tokens, truncate)
    return [
        {
            "role": history_reversed[i]['role'],
            "content": s
        } for i, s in enumerate(result)
    ][::-1] # re-reverse


def append_history(messages, tags):
    max_idx = 0
    if STATE['HISTORY']:
        max_idx = max(h['idx'] for h in STATE['HISTORY'])
    STATE['HISTORY'] += [{
        "message": message,
        "tags": tags,
        "idx": max_idx + i + 1
    } for i, message in enumerate(messages)]


def fill_context(context_definition):
    output = []
    for element in context_definition:
        assert 'type' in element
        t = element['type']
        if t == 'input':
            output.append({
                "role": element['role'],
                "content": STATE['INPUT']
            })
        elif t == 'const':
            output.append({
                "role": element['role'],
                "content": element['value']
            })
        elif t == 'history':
            tags = set(element.get('tags', ['default']))
            output += take_history(
                (h for h in STATE['HISTORY'] if tags.union(set(h['tags']))),
                element['amount']
            )
        elif t == 'reducer':
            reducer = element['reducer']
            output.append({
                'role': element['role'],
                'content': get_thing(reducer)['value']
            })
        elif t == 'chat':
            chat = element['chat']
            output.append({
                'role': element['role'],
                'content': get_thing(chat)['value']
            })
        elif t == 'state_machine':
            which = element['state_machine']
            machine_state = get_thing(which)['state']
            state_def = GAME['things'][machine_state]
            output += fill_context(state_def.get('body', []))
    return output

def perform_chat(definition, stream=False):
    context = fill_context(definition['body'])
    choices = definition.get('choices', None)
    regex = definition.get('regex', None)
    max_tokens = definition.get('max_tokens')
    response = MODEL.infer(context,
                           max_tokens=max_tokens,
                           options=choices,
                           regex=regex,
                           stream=stream,
                           generation_kwargs=definition.get('generation_kwargs', {}))
    return response


def tick_reducer(name):
    if not STATE['HISTORY']:
        return
    definition = GAME['things'][name]
    reducer = get_thing(name)
    current_idx = STATE['HISTORY'][-1]['idx']
    last_idx = reducer['last_run']
    messages_since_last = [
        h['message']['content']
        for h in STATE['HISTORY']
        if last_idx < h['idx'] <= current_idx
    ]
    tokens_since_last = MODEL.token_encode("".join(messages_since_last))
    if len(tokens_since_last) > definition['every']:
        # Perform the reduction
        payload = fill_context(definition['parts'])
        max_tokens = definition.get('max_tokens')
        value = MODEL.infer(payload, max_tokens=max_tokens)
        reducer['value'] = value
        reducer['last_run'] = current_idx

def perform_input(prompt):
    while True:
        response = input(prompt)
        if response == '!shell':
            from IPython import embed
            embed()
        elif response == '!reload':
            global GAME
            GAME = load_file(ARGS.game)
        else:
            return response


def perform_actions(definition):
    for action in definition:
        t = action['type']
        if t == 'input':
            value = action.get('value')
            prompt = action.get('prompt', '> ')
            response = value
            if not response:
                response = perform_input(prompt)
            STATE["INPUT"] = response
        elif t == 'chat':
            which = action['chat']
            chat = GAME['things'][which]
            value = perform_chat(chat, stream=action.get('stream', False))
            get_thing(which)['value'] = value
        elif t == 'tick':
            reducer = action.get('reducer')
            machine = action.get('machine')
            assert reducer or machine
            assert not (reducer and machine)
            if reducer:
                tick_reducer(reducer)
            elif machine:
                tick_machine(GAME['things'][reducer])
        elif t == 'append_history':
            messages = fill_context(action['body'])
            tags = action.get('tags', ['default'])
            append_history(messages, tags)
        elif t == 'print':
            messages = fill_context(action['body'])
            for m in messages:
                print(m['content'])
        elif t == 'background':
            THREAD_POOL.submit(perform_actions, action['actions'])
        elif t == 'state_machine':
            which = action['state_machine']
            machine_state = get_thing(which)['state']
            state_def = GAME['things'][machine_state]
            perform_actions(state_def.get('actions', []))
        elif t == 'state_transition':
            which = action['state_machine']
            target_state = action['state']
            get_thing(which)['state'] = target_state
        elif t == 'chat_pick':
            payload = fill_context(action['body'])
            options = action['options']
            result = MODEL.infer(payload, options=list(options.keys()))
            print(result)
            perform_actions(options[result])
        elif t == 'loop':
            which = action['loop']
            perform_actions(GAME['things'][which]['actions'])
        else:
            raise Exception("Unrecognized action type: ", t)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='main', help="Which loop to execute")
    parser.add_argument("game", type=str, help="Which adventure file to load")
    parser.add_argument("--debug", action='store_true', help="Whether or not to start with debug mode on")
    parser.add_argument("--save-name", type=str, help="Which named save file to use")
    parser.add_argument("--continue", dest="cont", action="store_true", help="Continue with the most recently used save file.")
    parser.add_argument("--model", type=str, choices=['chatgpt', 'ollama', 'vllm', 'default'], default='default', help="Which model to use")

    args = parser.parse_args()

    DEBUG = args.debug

    if args.save_name and args.cont:
        raise Exception("Use either '--continue' or '--save-name', not both")

    default_save_name = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    save_name = args.cont and saves.get_most_recent_save(SAVES_CONNECTION) or args.save_name or default_save_name
    print(f"Using save file {save_name}")
    if not save_name:
        raise Exception("Couldn't find save with that name")

    SAVE_NAME = save_name

    game_definition = load_file(args.game)

    # Set up the model
    model_definition = {
        'name': args.model
    }
    if args.model == 'default':
        model_definition = game_definition['model']
    init_model(model_definition)

    init_state(game_definition)
    ARGS = args
    done = False
    with ThreadPoolExecutor() as worker:
        perform_actions(GAME['intro']['actions'])
        while not done:
            THREAD_POOL = worker
            perform_actions(GAME['things']['main']['actions'])
