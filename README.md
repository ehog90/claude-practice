# claude-practice

Small examples of working with the Anthropic API: tool use, evals, prompt caching, embeddings/RAG, image prompts and the text editor tool.

## Setup

Requires Python 3.10+.

```sh
# 1. Create the virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then edit .env and fill in your keys
```

Leave the environment with `deactivate`.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | API key |
| `ANTHROPIC_BASE_URL` | API / gateway base URL |
| `ANTHROPIC_MODEL` | Model ID to use |
| `VOYAGE_API_KEY` | Voyage AI key (embedding examples) |

`.env` is git-ignored. Never commit real keys.

## Managing dependencies

```sh
pip install <package>
pip freeze > requirements.txt   # update the lockfile
```

## Layout

- `main.py`, `utils.py` – basic usage and helpers
- `simple_tool_use/`, `tool_use/`, `tool_loop/`, `multi_turn_tool_use/` – tool use examples
- `eval*.py`, `evaluator.py`, `eval_dataset.json` – evaluation examples
- `prompt_caching/` – prompt caching demo
- `embedding_example/` – embeddings and RAG chat
- `text_editor_example/` – text editor tool
- `imageprompt/` – image prompting
