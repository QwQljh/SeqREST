"""OpenAPI document index only: no edges, embeddings, or graph traversal."""
import copy
import hashlib
import json
import os
import random
import re
from urllib.parse import urlsplit

METHODS = {'get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace'}


class APICatalog:
    def __init__(self, document, base_url):
        self.document = document
        self.base_url = base_url
        self.api_swagger_map = {}
        # Run-scoped state survives per-scenario request-log resets.
        self.covered_2xx = set()
        self.target_counts = {}
        self.scenario_signatures = set()
        self.full_coverage_scenario_texts = set()
        self.required_target = None
        self.required_targets = []
        self.scenario_rejections = 0
        self.coverage_feedback = None
        for path, raw_item in document.get('paths', {}).items():
            item = self.resolve(raw_item)
            for method, operation in item.items():
                if method.lower() not in METHODS or not isinstance(operation, dict):
                    continue
                op = copy.deepcopy(operation)
                parameters = {(p.get('in'), p.get('name')): p
                              for p in item.get('parameters', []) + op.get('parameters', [])}
                op['parameters'] = list(parameters.values())
                op.setdefault('security', document.get('security', []))
                op['securityDefinitions'] = document.get('securityDefinitions',
                    document.get('components', {}).get('securitySchemes', {}))
                self.api_swagger_map[f'{method.upper()} {path}'] = op
        if not self.api_swagger_map:
            raise ValueError('OpenAPI document contains no HTTP operations')

    def resolve(self, value, seen=(), depth=0):
        if depth > 24:
            return {'description': 'Nested schema depth limit reached'}
        if isinstance(value, list):
            return [self.resolve(v, seen, depth + 1) for v in value]
        if not isinstance(value, dict):
            return value
        if '$ref' in value:
            ref = value['$ref']
            if not ref.startswith('#/'):
                raise ValueError(f'External OpenAPI reference must be bundled first: {ref}')
            if ref in seen:
                return {'description': f'Recursive schema {ref}'}
            target = self.document
            for part in ref[2:].split('/'):
                target = target[part.replace('~1', '/').replace('~0', '~')]
            merged = self.resolve(target, seen + (ref,), depth + 1)
            return {**merged, **self.resolve({k: v for k, v in value.items() if k != '$ref'}, seen, depth + 1)}
        return {k: self.resolve(v, seen, depth + 1) for k, v in value.items()}

    def get_base_url(self):
        return self.base_url

    def scenario_context(self, count=10):
        keys = list(self.api_swagger_map)
        uncovered = [key for key in keys if key not in self.covered_2xx]
        if not uncovered:
            self.required_targets = []
            self.required_target = None
            return ('All documented operations have already received a 2XX. '
                    'Design a NEW boundary or unusual-but-valid business scenario; '
                    'prefer different resource states, enum values, optional fields, '
                    'and operation order. An operation sequence may repeat only when its '
                    'parameters or resource-state test is substantively different.\n'
                    'Full operation directory:\n' + '\n'.join(keys)
                    + '\nThis is phase 1: request only the detailed documents needed for the new scenario.')
        feedback = self.coverage_feedback
        unvisited = [key for key in uncovered if not feedback or feedback.status(key) == 'unvisited']
        failed = [key for key in uncovered if feedback and feedback.status(key) == 'coverage_failed']
        ranked_unvisited = sorted(unvisited, key=lambda k: (self.target_counts.get(k, 0), keys.index(k)))
        ranked_failed = sorted(failed, key=lambda k: (self.target_counts.get(k, 0), keys.index(k)))
        # Keep discovering new operations, but reserve a slot for a failed
        # operation once reflection has produced a possible repair.
        self.required_targets = ranked_unvisited[:2]
        if ranked_failed and len(self.required_targets) < 3:
            self.required_targets.append(ranked_failed[0])
        self.required_targets += ranked_unvisited[len(self.required_targets):3]
        self.required_targets = self.required_targets[:3]
        self.required_target = self.required_targets[0] if self.required_targets else None
        for target in self.required_targets:
            self.target_counts[target] = self.target_counts.get(target, 0) + 1
        return ('Full operation directory (any operation may be used):\n' + '\n'.join(keys)
                + '\nAlready covered by 2XX in this run:\n' + '\n'.join(sorted(self.covered_2xx))
                + '\nRecommended uncovered targets (include at least one; you may include several):\n'
                + '\n'.join(self.required_targets)
                + ('\n' + feedback.prompt_context() if feedback else '')
                + '\nInclude at least one recommended target and any necessary setup steps. Do not repeat an earlier ordered operation sequence.\n'
                + 'This is phase 1: request detailed documents before writing the scenario.')

    def detailed_context(self, endpoints):
        selected = []
        for endpoint in endpoints:
            endpoint = str(endpoint).strip().strip('`')
            if endpoint in self.api_swagger_map and endpoint not in selected:
                selected.append(endpoint)
        return json.dumps({k: self.api_swagger_map[k] for k in selected}, ensure_ascii=False)

    def accept_scenario(self, endpoints, scenario_text=None):
        try:
            signature = tuple(self.normalize_endpoint(e)['endpoint'] for e in endpoints)
        except ValueError as exc:
            self.scenario_rejections += 1
            return False, str(exc)
        uncovered = set(self.api_swagger_map) - self.covered_2xx
        reason = None
        if not signature:
            reason = 'No documented operations were parsed.'
        elif uncovered and not uncovered.intersection(signature):
            reason = 'The executable sequence must contain at least one operation not yet covered by 2XX.'
        elif len(signature) > int(os.getenv('SCENARIO_MAX_STEPS', '8')):
            reason = 'Too many steps; generate at most ' + os.getenv('SCENARIO_MAX_STEPS', '8') + ' executable steps.'
        elif uncovered and signature in self.scenario_signatures:
            reason = 'Duplicate ordered operation sequence; generate a different scenario.'
        elif not uncovered and not scenario_text and signature in self.scenario_signatures:
            reason = 'Provide a distinct boundary scenario, not only a repeated operation sequence.'
        elif not uncovered and scenario_text:
            normalized_text = re.sub(r'\s+', ' ', scenario_text).strip().lower()
            scenario_digest = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
            if scenario_digest in self.full_coverage_scenario_texts:
                reason = 'Duplicate boundary scenario; change the tested values or resource state.'
        if reason:
            self.scenario_rejections += 1
            return False, reason
        self.scenario_signatures.add(signature)
        if not uncovered and scenario_text:
            self.full_coverage_scenario_texts.add(scenario_digest)
        return True, 'accepted'

    def matches(self, method, path):
        for signature in self.api_swagger_map:
            verb, template = signature.split(' ', 1)
            pattern = re.sub(r'\\\{[^}]+\\\}', r'[^/]+', re.escape(template))
            if verb == method.upper() and re.fullmatch(pattern, path):
                return True
        return False

    def normalize_endpoint(self, endpoint):
        """Resolve a recorded endpoint without discarding its concrete test values."""
        if not isinstance(endpoint, str):
            raise ValueError('Endpoint must be a METHOD /path string')
        original = endpoint.strip().strip('`').strip().replace('{{', '{').replace('}}', '}')
        parts = original.split(None, 1)
        if len(parts) != 2 or parts[0].lower() not in METHODS:
            raise ValueError('Use METHOD /path for one operation')
        method, supplied = parts[0].upper(), parts[1]
        if not supplied.startswith('/') or supplied.startswith('//') or '#' in supplied:
            raise ValueError('Use a relative endpoint path, not a URL or fragment')
        path = supplied.split('?', 1)[0]

        def candidates(path):
            exact = f'{method} {path}'
            if exact in self.api_swagger_map:
                return [(exact, {})]
            found = []
            for signature in self.api_swagger_map:
                verb, template = signature.split(' ', 1)
                if verb != method:
                    continue
                names = re.findall(r'\{([^{}]+)\}', template)
                chunks = re.split(r'\{[^{}]+\}', template)
                pattern = '([^/]+)'.join(re.escape(chunk) for chunk in chunks)
                match = re.fullmatch(pattern, path)
                if match:
                    found.append((signature, dict(zip(names, match.groups()))))
            return found

        found = candidates(path)
        prefix = urlsplit(self.base_url).path.rstrip('/')
        if not found and prefix and path.startswith(prefix + '/'):
            found = candidates(path[len(prefix):])
        if not found:
            raise ValueError(f'Unknown documented operation: {method} {path}')
        if len(found) != 1:
            raise ValueError('Ambiguous endpoint; specify the documented template: '
                             + ', '.join(signature for signature, _ in found))
        signature, values = found[0]
        return {'endpoint': signature, 'original_endpoint': original,
                'path_values': values,
                'query_string': supplied.partition('?')[2]}
