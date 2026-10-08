"""Prepare a container session using one set of frozen runtime defaults."""
import json
import math
import os
from pathlib import Path
import shlex
import sys

from seqrest.openapi.loader import find_specification, load_specification

TOOL_ROOT = Path(__file__).resolve().parents[2]


def prepare(environ, tool_root=TOOL_ROOT, specifications=Path('/specifications')):
    defaults = json.loads((tool_root/'runtime-defaults.json').read_text())
    model = json.loads((tool_root/'model-config.json').read_text())
    required_model = {'LLM_BASE_URL', 'LLM_MODEL'}
    allowed_model = required_model | {'LLM_API_KEY'}
    if set(model) - allowed_model:
        raise ValueError('Unsupported model configuration fields: '+str(set(model)-allowed_model))
    if required_model - set(model):
        raise ValueError('Missing model configuration fields: '+str(required_model-set(model)))
    if any(not isinstance(value, str) for value in model.values()):
        raise ValueError('Model configuration values must be strings')
    defaults.update(model)
    defaults['LLM_API_KEY'] = model.get('LLM_API_KEY') or 'EMPTY'
    values = {key: str(environ.get(key) or value) for key, value in defaults.items()}
    for key, default in defaults.items():
        if key in allowed_model:
            continue
        value = values[key]
        if str(default) in ('true', 'false'):
            if value.lower() not in ('true','false','1','0','yes','no','on','off'):
                raise ValueError(f'{key} must be a boolean')
            values[key] = 'true' if value.lower() in ('true','1','yes','on') else 'false'
        elif str(default).isdigit():
            number = float(value) if key == 'TIME_BUDGET' else int(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f'{key} must be nonnegative')
    if float(values['TIME_BUDGET']) <= 0 or int(values['MAX_REQUESTS']) <= 0:
        raise ValueError('TIME_BUDGET and MAX_REQUESTS must be positive')
    for key in ['LLM_CONTEXT_WINDOW','LLM_MAX_OUTPUT_TOKENS','LLM_MAX_INPUT_BYTES','LLM_TIMEOUT_SECONDS']:
        if int(values[key]) <= 0:
            raise ValueError(f'{key} must be positive')
    api = environ.get('API_NAME') or environ.get('CONFIG_SYSTEM_NAME') or environ.get('API') or 'scs'
    tool = environ.get('TOOL') or 'seqrest'
    run = environ.get('RUN') or 'seqrest-output'
    for label, value in [('API',api),('TOOL',tool),('RUN',run)]:
        if not value or value in ('.','..') or '/' in value or '\\' in value:
            raise ValueError(f'{label} must be a single directory name')
    host = environ.get('HOST') or 'localhost'
    port = int(environ.get('PORT') or '9090')
    if not 1 <= port <= 65535:
        raise ValueError('PORT must be between 1 and 65535')
    if ':' in host and not host.startswith('['): host = '['+host+']'
    base_url = environ.get('CONFIG_BASE_URL') or f'http://{host}:{port}'
    if not values.get('LLM_BASE_URL','').startswith(('http://','https://')) or not values.get('LLM_MODEL'):
        raise ValueError('Set a valid LLM_BASE_URL and LLM_MODEL')
    requested = environ.get('CONFIG_OPENAPI_JSON')
    if requested:
        path = Path(requested)
        if not path.is_absolute(): path = tool_root/'src'/path
    else:
        path = find_specification(specifications, api)
    # Validation here gives a clear startup error; the catalog uses the same reader.
    load_specification(path)
    log_dir = environ.get('CONFIG_LOG_PATH') or '/tool/artifacts/'+api+'/'+tool+'/'+run
    values.update(CONFIG_SYSTEM_NAME=api, CONFIG_BASE_URL=base_url,
                  CONFIG_OPENAPI_JSON=str(path), CONFIG_LOG_PATH=log_dir,
                  REST_LEAGUE_SEED=environ.get('REST_LEAGUE_SEED') or api+'-'+run,
                  PYTHONUNBUFFERED='1')
    return values


if __name__ == '__main__':
    try:
        values = prepare(os.environ)
        for key, value in values.items():
            print('export '+key+'='+shlex.quote(value))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print('SeqREST configuration error: '+str(error), file=sys.stderr)
        sys.exit(2)
