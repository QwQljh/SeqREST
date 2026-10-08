#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults are kept in runtime-defaults.json; model settings in model-config.json.
# Shell values are generated with shlex.quote, including paths and credentials.
cd /tool/src
SESSION_ENV=$(python -m seqrest.runtime)
eval "$SESSION_ENV"

echo "SeqREST starting: API=$CONFIG_SYSTEM_NAME URL=$CONFIG_BASE_URL TIME_BUDGET=$TIME_BUDGET minutes"
echo "Model: $LLM_MODEL at $LLM_BASE_URL; tool context budget=$LLM_CONTEXT_WINDOW"
echo "Specification: $CONFIG_OPENAPI_JSON"
echo "Additional file output: $SEQREST_SAVE_ARTIFACTS"

# Keep stdout available to RESTGym even when additional file output is disabled.
if [ "$SEQREST_SAVE_ARTIFACTS" = "true" ]; then
  mkdir -p "$CONFIG_LOG_PATH"
  exec > >(tee -a "$CONFIG_LOG_PATH/container.log") 2>&1
fi

python -u -m seqrest
echo "SeqREST completed successfully"
if [ "$RESTGYM_KEEPALIVE_AFTER_SUCCESS" = "true" ]; then
  while true; do sleep 3600; done
fi
