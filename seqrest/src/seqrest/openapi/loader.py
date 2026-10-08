"""Read JSON/YAML specifications without changing the downstream catalog."""
import json
from pathlib import Path

import yaml


def load_specification(path):
    path = Path(path)
    try:
        with path.open(encoding='utf-8-sig') as stream:
            if path.suffix.lower() in ('.yaml', '.yml'):
                document = yaml.safe_load(stream)
            else:
                document = json.load(stream)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f'Invalid OpenAPI specification {path}: {error}') from error
    if not isinstance(document, dict) or not isinstance(document.get('paths'), dict):
        raise ValueError('OpenAPI specification must be an object containing paths')
    return document


def find_specification(directory, api):
    directory = Path(directory)
    # Preserve JSON preference and the historical dataset naming convention.
    names = [f'{api}-openapi.json', f'{api}.json', 'openapi.json',
             f'{api}-openapi.yaml', f'{api}-openapi.yml', f'{api}.yaml',
             f'{api}.yml', 'openapi.yaml', 'openapi.yml']
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'No JSON/YAML specification found for {api} in {directory}')
