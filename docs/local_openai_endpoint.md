# Running BooookScore with a self-hosted OpenAI-compatible model

The CLI tools can talk to any chat-completions endpoint that follows the OpenAI API. This guide
walks through pointing BooookScore at a locally hosted service such as the `call_llm` helper
shown earlier.

## 1. Provide an API key value

BooookScore now accepts the API key directly as a string. For local deployments that do not
validate the credential, you can pass a placeholder such as `empty` (the same value used in the
`call_llm` helper). If you prefer managing credentials via files, you can still point `--api_key`
at a text file—the CLI automatically detects whether the argument is a literal key or a path.

## 2. Verify your local endpoint

Ensure your service exposes the OpenAI-compatible chat endpoint at a base URL such as
`http://10.244.2.114:8808/v1`. You should be able to send a request with the Python snippet below
and receive a normal chat completion response:

```python
from openai import OpenAI

client = OpenAI(api_key="empty", base_url="http://10.244.2.114:8808/v1")
reply = client.chat.completions.create(
    model="reportify",
    messages=[{"role": "user", "content": "Ping"}],
)
print(reply.choices[0].message.content)
```

If this round-trip works, BooookScore can use the same endpoint.

## 3. Run BooookScore commands

Every CLI that calls a model accepts the `--base_url` switch. Combine it with `--api openai`
(because the request payload follows the OpenAI schema) and your chosen placeholder API key
string. Below is an end-to-end example that scores a JSON file of summaries:

```bash
python -m booookscore.score \
  --summ_path summaries/chatgpt-2048-hier-cleaned.json \
  --annot_path my_annotations.json \
  --model reportify \
  --api openai \
  --api_key empty \
  --base_url http://10.244.2.114:8808/v1
```

Use the same `--base_url` flag when generating summaries or running post-processing. The `--api`
option selects which built-in provider client BooookScore uses (`openai`, `anthropic`, or
`together`). For any OpenAI-compatible endpoint—self-hosted or not—set it to `openai` so the
request schema matches what your service expects:

```bash
python -m booookscore.summ \
  --book_path data/all_books_chunked.pkl \
  --summ_path my_summaries.json \
  --model reportify \
  --api openai \
  --api_key empty \
  --base_url http://10.244.2.114:8808/v1 \
  --method hier \
  --chunk_size 2048 \
  --max_context_len 4096 \
  --max_summary_len 2048
```

```bash
python -m booookscore.postprocess \
  --input_path my_summaries.json \
  --model reportify \
  --api openai \
  --api_key empty \
  --base_url http://10.244.2.114:8808/v1 \
  --remove_artifacts
```

Each command writes its output in-place (e.g., annotations, cleaned summaries), so you can continue
with the normal workflow once the local model responds successfully.
