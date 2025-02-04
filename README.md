# Chat GPT RPG

## Usage
`OPENAI_API_KEY=<foo> python3 chatty2.py v2_games/test_town.yaml --model chatgpt`
Either that or replace `test_town.yaml` with `wizard_town_2.yaml`.
Neither are written very well as I've mostly been playing around with the underlying mechanics and seeing 
what kinds of structures I can get up to, but you can read the yaml files to get a sense of what's in the games
to see how what you get up to matches up with what's written.

## Notes
Code might be in random states of workingness, don't expect much. This repo has two swings at the GPT RPG idea:

`chatty.py` was a RAG focused approach. The cool thing about this is you can stuff a lot of data into a database and it
sometimes can pull out relevant data to guide the experience with, but it's hard to get it to select the correct relevant data.

Once I started playing around with structured output, I went more in the direction of `chatty2.py`, which has a more guided structure.
You can use structured output to have the LLM answer questions about the state of the game - did this trigger event happen? Did the player
take damage? Did they move into this location? And then you can update a state machine that provides the context.

The only other kind of novel idea here is the idea of "reducers", which keep track of a little piece of state and you ask the LLM
to run a specified prompt once in a while to update that piece of state. This is particularly good for two things- keeping track of
history on a longer timescale (the prompt would be something like "summarize what's happened in the adventure up to this point", and
you keep the output of that around to put into the context), and keeping track of the player's inventory, quest log, 
or other things that would be too fussy to keep in a general summary but are highly important to the player.
