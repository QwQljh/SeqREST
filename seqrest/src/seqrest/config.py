"""Session configuration. Docker's launcher supplies the tested runtime defaults."""
import os
from pathlib import Path
from openai import OpenAI
from seqrest.openapi.catalog import APICatalog
from seqrest.openapi.loader import load_specification

CONFIG_LLM_API_KEY = os.getenv("LLM_API_KEY") or "EMPTY"
CONFIG_LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-14b")
CONFIG_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
CONFIG_ABLATION = os.getenv("CONFIG_ABLATION")
ABLATION_NO_RELEVANT_PARAMETER = "No-Relevant-Parameter"
ABLATION_NO_REFLECTION = "No-Reflection"

CONFIG_SYSTEM_NAME = os.getenv("CONFIG_SYSTEM_NAME") or os.getenv("API_NAME") or os.getenv("API") or "scs"
CONFIG_BASE_URL = os.getenv("CONFIG_BASE_URL") or "http://localhost:9090"
CONFIG_OPENAPI_JSON = os.getenv("CONFIG_OPENAPI_JSON") or f"apis/{CONFIG_SYSTEM_NAME}/specifications/openapi.json"
CONFIG_LOG_PATH = os.getenv("CONFIG_LOG_PATH") or "/tool/artifacts"
CONFIG_SAVE_ARTIFACTS = os.getenv("SEQREST_SAVE_ARTIFACTS", "false").lower() in ("1", "true", "yes", "on")
if CONFIG_ABLATION:
    CONFIG_LOG_PATH = str(Path(CONFIG_LOG_PATH)/CONFIG_ABLATION)
if CONFIG_SAVE_ARTIFACTS:
    Path(CONFIG_LOG_PATH).mkdir(parents=True, exist_ok=True)

print(f"TestSystemInfo: {CONFIG_SYSTEM_NAME=} {CONFIG_BASE_URL=}")
print(f"LLMInfo: {CONFIG_LLM_MODEL=} {CONFIG_LLM_BASE_URL=}")
catalog = APICatalog(load_specification(CONFIG_OPENAPI_JSON), CONFIG_BASE_URL)
print(f"OpenAPI catalog: {len(catalog.api_swagger_map)} operations; relationship graph disabled")
openai_raw_client = OpenAI(api_key=CONFIG_LLM_API_KEY, base_url=CONFIG_LLM_BASE_URL, timeout=60)
