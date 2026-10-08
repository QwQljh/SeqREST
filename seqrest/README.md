# SeqREST

SeqREST tests REST APIs using a local LLM, executable request sequences, response-derived resources, and sequence and parameter mutation. It accepts OpenAPI JSON or YAML.

## Quick start

1. Place this directory at `tools/seqrest` in an existing RESTGym checkout.
2. Start a local OpenAI-compatible model service. Edit `model-config.json` with its reachable `LLM_BASE_URL` and actual `LLM_MODEL`. Set optional `LLM_API_KEY` if authentication is enabled; otherwise keep `EMPTY`. An explicitly supplied container environment variable overrides the file. The tool does not start the model service or download weights. The address must be reachable from the tool container.
3. Set `enabled: true` in `restgym-tool-config.yml`; disable other tools if running only SeqREST.
4. From the RESTGym root, run `./restgym.sh b` and select the tool-image build option. The image is named `restgym-seqrest`. Existing images may be skipped; rebuild after changing source or configuration.
5. Select the APIs and time budget in RESTGym, then run `./restgym.sh l`.

Before launching, check that the model can generate a response. Use the URL, model, and `LLM_API_KEY` from `model-config.json`; replace `Bearer EMPTY` when using a real key. A successful response contains non-empty `choices[0].message.content`:

```bash
curl -fsS --max-time 60 http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' -d '{"model":"qwen3-14b","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

To explicitly rebuild from the RESTGym root:

```bash
docker build -t restgym-seqrest -f tools/seqrest/Dockerfile .
```

## Files and output

`src/seqrest` contains the runtime source. `run.sh` automatically starts `python -m seqrest`. `requirements.txt` is the single dependency file and pins the complete tested dependency set. `runtime-defaults.json` contains internal defaults; normal use only requires editing the model configuration and enabling the tool.

RESTGym supplies `API`, `HOST`, `PORT`, `TOOL`, `RUN`, and `TIME_BUDGET` in minutes, and collects HTTP interactions and tool logs. Additional container-local files are disabled by default (`SEQREST_SAVE_ARTIFACTS=false`).

The model server determines the actual context capacity. Internal `LLM_CONTEXT_WINDOW` is a prompt-budget assumption and does not configure the server. Bundle external OpenAPI `$ref` files before use.

[中文说明](README_zh.md)

Keep `EMPTY` in publicly shared packages. Setting a real key in the file includes it in the built image; use an explicitly supplied container environment variable when available to avoid embedding credentials.
