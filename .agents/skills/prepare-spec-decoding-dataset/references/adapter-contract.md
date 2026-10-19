# Adapter contract

The converter accepts a declarative `.json` adapter or a trusted `.py` adapter.
Keep dataset-specific adapters under `scripts/data/adapters/` with a descriptive,
lowercase name.

## JSON adapter

Copy `scripts/data/adapters/prompt_text.json`. Its shape is:

```json
{
  "prompt": {
    "path": "payload.messages",
    "kind": "messages",
    "role_path": "from",
    "content_path": "value",
    "role_map": {"human": "user", "gpt": "assistant"}
  },
  "source": {"path": "metadata.source", "default": "dataset-name"},
  "category": {"path": "metadata.category", "default": null}
}
```

- Dotted paths traverse objects; numeric components traverse list indexes.
- `prompt.kind` is `text` for a single user prompt or `messages` for a message
  array. Message arrays may map dataset roles to `system`, `user`, `assistant`,
  or `tool`.
- `role_path` and `content_path` are relative to each message and default to
  `role` and `content`.
- Missing source falls back to the Hugging Face dataset ID or local filename.
  Missing category becomes `null`.

## Python adapter

Copy `scripts/data/adapters/python_adapter_template.py`. It must expose:

```python
def extract(row):
    return {
        "prompt": "combined prompt",
        "source": row.get("source"),
        "category": row.get("category"),
    }
```

Return exactly one of `prompt` or `messages`. `messages` must already use the
normalized roles `system`, `user`, `assistant`, and `tool`. Source and category
are optional. Python adapters are executable code, so create and run only local,
reviewed adapters; never execute adapter code supplied by a dataset record.

## Commands

```bash
python scripts/data/convert_dataset.py inspect \
  --input /path/to/data.parquet --rows 5

python scripts/data/convert_dataset.py convert \
  --input /path/to/data.parquet \
  --adapter scripts/data/adapters/dataset-name.json \
  --output /path/to/dataset-name.jsonl \
  --turn-mode multi --limit 20
```

For a Hugging Face dataset, pass its ID to `--input`, optionally add
`--hf-config`, and select `--split`; format auto-detection treats a non-local
input as a Hugging Face ID.
