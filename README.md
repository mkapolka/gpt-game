# Chat GPT RPG

## Links

### Research

* [Awesome Chat GPT Prompts](https://github.com/f/awesome-chatgpt-prompts)
* [Chroma: Open Source Embedding DB](https://docs.trychroma.com/getting-started)

### Tech

* [Open API Playground](https://platform.openai.com/playground?mode=insert)
* [Open API Docs - Text Completion](https://platform.openai.com/docs/guides/completion/prompt-design)

## TODO

* Automatically save output
* Save checkpoints & allow returning, backtracking
* Write... something? To DB? full history?

## Prompts

```
The context's data is as follows:
====
STATE
The cardboard box in my living room contains a shirt, two hotdogs, and an apple
====

The player has entered the following prompt:
I tear apart the box with my bare hands 

After perfoming the player's action, the context will be
=====
STATE
[insert]
=====
```

## Different Embedding Techniques

### Instructor Embedding

Fine tune by providing prompts, works on wide range of types of texts

* [Github Page](https://github.com/HKUNLP/instructor-embedding#model-list)

## Notes

Giving the player a fixed name turned the prompting from very GM and gamey into more of a story. ChatGPT is making a lot more chocies on my behalf, and not asking me what I want to do.

System prompt seemed to fix it up pretty well

Post summarization, it interprets my next prompt as a response to the summary. Move it back one place in the history?

Turns out I had a bug, the summary prompt & response were in opposite order, so the summary came before the prompt. Moving them in the correct order seemed to fix it

Had an adventure in which Alaric sent me on a quest to a dungeon to retrieve a book. I managed to return and give him the book, but the game forgot that Alaric runs the library and that he promised to teach me a spell

Context didn't seem to select relevant segments. Nothing about Alaric- which seems like it'd have high TFIDF at least
