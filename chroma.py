from collections import defaultdict
import argparse
import re
import tempfile
import hashlib
import os

import openai

openai.api_key = os.environ['OPENAI_API_KEY']

HISTORY_SUMMARY_PROMPT = {
    "role": "user",
    "content": "Summarize our adventure up to this point."
}
HISTORY = []


SYSTEM_PROMPT = {"role": "system", "content": 
""" You are a game master, playing a role playing game with the player.
The player will tell you what they want to do, and you will respond with what happens in the game world.
Ignore any references to rolling dice, just focus on telling a compelling story with the player"""}



def connect_db():
    import chromadb
    from chromadb.utils import embedding_functions
    chroma_client = chromadb.Client()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='multi-qa-MiniLM-L6-dot-v1'
    )
    return chroma_client.create_collection("my_collection", embedding_function=embedding_function)


def doc_id(doc):
    return hashlib.md5(doc.encode()).hexdigest()

def load_documents(collection, path):
    with open(path, 'r') as f:
        docs = []
        current_document = []
        for line in f.readlines():
            stripped = line.strip()
            if stripped != "@@@":
                current_document.append(stripped)
            else:
                if current_document:
                    docs.append("\n".join(current_document))
                current_document = []
        ids = [doc_id(doc) for doc in docs]
        collection.add(
            documents=docs,
            ids = ids
        )

def save_document(collection, document):
    collection.add(documents=[document],
                   ids=[doc_id(document)])

def save_pair(collection, question, answer):
    s = f"{question}\n===\n{answer}@@@"
    collection.add(documents=[s],
                   ids=[doc_id(s)])

def query_history(collection, inputs, n_results):
    n_results = min(n_results, collection.count())
    results = collection.query(query_texts=inputs, n_results=n_results)
    sum_scores = defaultdict(float)
    docs = {}
    for i in range(len(inputs)):
        for j in range(n_results):
            sum_scores[results['ids'][i][j]] += results['distances'][i][j]
            docs[results['ids'][i][j]] = results['documents'][i][j]
    choices = sorted(sum_scores.items(), key=lambda p: p[1])[:n_results]
    return [docs[c] for c, _ in choices]

def query_to_prompts(collection, inputs, n_results, flip_querent=False):
    docs = query_history(collection, inputs, n_results)
    output = []
    for context in docs:
        question, answer = re.split("\\s=+\\s", context)
        asker = "assistant" if not flip_querent else "user"
        answerer = "user" if not flip_querent else "assistant"
        if answer:
            output.append({
                "role": asker,
                "content": question
            })
            output.append({
                "role": answerer,
                "content": answer
            })
        else:
            output.append({
                "role": answerer,
                "content": question
            })
    return output

def game_prompt(collection, query):
    global HISTORY
    context_prompt = {
        "role": "user",
        "content": "We are going to play a fantasy role playing game. Ask me questions about the game we are going to play."
    }
    context_entries = query_to_prompts(collection, [m['content'] for m in HISTORY[-4:]] + [query], 10)

    history_prompt = {
        "role": "user",
        "content": "Using the background we've talked about, let's play a role playing game. You be the dungeon master who describes the game environment, and I'll be a player who says what I want my character to do. Narrate directly to me in the second person, and don't take many actions on my character's behalf."
    }

    player_input = {"role": "user", "content": query}

    messages = [
        SYSTEM_PROMPT,
        context_prompt,
        *context_entries,
        history_prompt,
        *HISTORY[-10:],
        player_input
    ]
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, max_tokens=300)

    print("=== DEBUG ===")
    print(f"Cost: {response['usage']['prompt_tokens']} prompt tokens. {response['usage']['total_tokens']} total.")
    print("Msgs:")
    for message in messages:
        print(f"{message['role'][0].upper()}: {message['content'][:50]}...")
    text = response['choices'][0]['message']['content']
    print(text)
    HISTORY.append(player_input)
    HISTORY.append(response['choices'][0]['message'])
    # HISTORY = HISTORY[-10:]

    # Summarize history
    if len(HISTORY) % 10 == 0:
        print("Summarizing history...")
        summary = summarize_history(HISTORY[-10:])
        HISTORY.append(HISTORY_SUMMARY_PROMPT)
        HISTORY.append(summary)
        print("Summary: %s" % summary['content'])


def worldbuilding_prompt(collection, query):
    global HISTORY
    context_prompt = {
        "role": "user",
        "content": "We are going to play a fantasy role playing game. I will ask you questions about this fantasy world, and you will help me imagine that world."
    }
    context_entries = query_to_prompts(collection, [m['content'] for m in HISTORY] + [query], 10, True)

    player_input = {"role": "user", "content": query}

    messages = [
        SYSTEM_PROMPT,
        context_prompt,
        *context_entries,
        *HISTORY[-10:],
        player_input
    ]
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, max_tokens=300)

    print("=== DEBUG ===")
    print(f"Cost: {response['usage']['prompt_tokens']} prompt tokens. {response['usage']['total_tokens']} total.")
    print("Msgs:")
    for message in messages:
        print(f"{message['role'][0].upper()}: {message['content'][:50]}...")
    text = response['choices'][0]['message']['content']
    print("=== DEBUG ===")
    print(text)
    HISTORY.append(player_input)
    HISTORY.append(response['choices'][0]['message'])
    # HISTORY = HISTORY[-10:]

def edit_string(s):
    t = tempfile.NamedTemporaryFile()
    with open(t.name, 'w') as f:
        f.write(s)
    os.system("vim %s" % t.name)
    with open(t.name, 'r') as f:
        f.seek(0)
        return f.read()
    t.close()

def summarize_history(entries):
    payload = entries + [
        HISTORY_SUMMARY_PROMPT
    ]
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=payload, max_tokens=500)
    return response['choices'][0]['message']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("documents_file", type=str)
    parser.add_argument("--build", action='store_true', help="World buidler mode. Don't go into gameplay, just ask questions")
    args = parser.parse_args()

    collection = connect_db()
    load_documents(collection, args.documents_file)
    if not args.build:
        game_prompt(collection, "Okay, I'm ready to start the adventure. Please describe my first scene")

    while True:
        cmd = input(">").strip()
        if cmd[:2] == '!s':
            print("Saving last exchange to DB...")
            question, answer = HISTORY[-2:]
            with open(args.documents_file, 'a') as f:
                f.write(f"{question['content']}\n===\n{answer['content']}\n@@@\n")
            save_pair(collection, question['content'], answer['content'])
        elif cmd[:2] == '!e':
            print("Editing last exchange, then saving to DB...")
            question, answer = HISTORY[-2:]
            content = f"{question['content']}\n===\n{answer['content']}\n@@@\n"
            content = edit_string(content)
            with open(args.documents_file, 'a') as f:
                f.write(content)
            save_document(collection, content)
        else:
            if args.build:
                worldbuilding_prompt(collection, cmd)
            else:
                game_prompt(collection, cmd)
