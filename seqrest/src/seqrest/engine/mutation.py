"""Run-scoped successful seeds and bounded, single-parameter mutations.

No target-specific constants or source-code feedback. A 5XX is a candidate
fault, not a confirmed distinct bug. State is intentionally reset per process.
"""
import copy
import hashlib
import json
import os
from collections import Counter
from urllib.parse import quote

from seqrest.engine.equivalence import MISS_KEY, categories_for_parameter


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def request_args(item, base_url):
    return dict(base_url=base_url, **{key: copy.deepcopy(item[key]) for key in
                ('method', 'api', 'headers', 'params', 'payload', 'payload_type')})


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaves(child, path + (index,))
    else:
        yield path, value


def replace_at(value, path, candidate):
    if not path:
        return copy.deepcopy(candidate)
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    if candidate == MISS_KEY:
        if isinstance(cursor, dict):
            cursor.pop(path[-1], None)
        else:
            del cursor[path[-1]]
    else:
        cursor[path[-1]] = copy.deepcopy(candidate)
    return value


def mutation_sites(operation, seed):
    """Preserve authentication and path resource identifiers by default."""
    for param in operation.get('parameters', []):
        location = param.get('in')
        if location not in ('query', 'header', 'formData'):
            continue
        name = param.get('name', '')
        if any(word in name.lower() for word in ('authorization', 'token', 'cookie', 'api_key', 'apikey')):
            continue
        schema = dict(param.get('schema') or param)
        field = {'query': 'params', 'header': 'headers', 'formData': 'payload'}[location]
        yield field, (name,), schema, bool(param.get('required')), location
    body = operation.get('requestBody', {}).get('content', {}).get(seed['payload_type'], {}).get('schema')
    if body is None:
        body = next((p.get('schema') for p in operation.get('parameters', []) if p.get('in') == 'body'), None)
    if not body:
        return

    def walk(schema, value, path=(), required=True):
        if isinstance(value, dict) and schema.get('properties'):
            for name, child in schema['properties'].items():
                if name in value:
                    yield from walk(child, value[name], path + (name,), name in schema.get('required', []))
                else:
                    yield path + (name,), child, name in schema.get('required', [])
        elif isinstance(value, list) and value:
            yield path, schema, required
            yield from walk(schema.get('items', {}), value[0], path + (0,), True)
        else:
            yield path, schema, required
    for path, schema, required in walk(body, seed['payload']):
        if path and any(word in str(path[-1]).lower() for word in ('token', 'password', 'secret')):
            continue
        yield 'payload', path, schema, required, 'body'


class AdaptiveTesting:
    def __init__(self):
        self.seeds = {}
        self.seen = set()
        self.cursors = Counter()
        self.stagnation = Counter()
        self.mutation_counts = Counter()
        self.observations = set()
        self.phase_counts = Counter()
        self.stops = Counter()
        self.phase = 'scenario'
        self.request_limit = None
        self.deadline = None
        self.sent = 0
        self.epochs = Counter()
        self.epoch_counts = Counter()
        self.fault_requests = Counter()
        self.fault_signatures = set()
        self.fault_followups = Counter()

    def begin_epoch(self, endpoints):
        """Renew bounded budgets, retaining explored candidate history."""
        endpoints = sorted(set(endpoints))
        for endpoint in endpoints:
            self.epochs[endpoint] += 1
            self.epoch_counts[endpoint] = 0
            self.stagnation[endpoint] = 0
        print('MUTATION EPOCH: ' + json.dumps({
            op: self.epochs[op] for op in endpoints
        }))

    def observe(self, item, catalog):
        self.sent += 1
        item['phase'] = self.phase
        self.phase_counts[self.phase] += 1
        try:
            # Use the actual request, never an unrelated current_test_case label.
            endpoint = catalog.normalize_endpoint(item['method'] + ' ' + item['api'])['endpoint']
        except ValueError:
            return
        item['api_endpoint'] = endpoint
        status = item.get('response_code')
        if item.get('is_2xx'):
            catalog.covered_2xx.add(endpoint)
            # Retain only normal successful requests, not increasingly mutated seeds.
            if 'mutation' not in self.phase:
                self.seeds[endpoint] = request_args(item, catalog.base_url)
        observation = (endpoint, status, item.get('fault_signature') if item.get('is_5xx') else None)
        if item.get('is_5xx'):
            self.fault_requests[endpoint] += 1
            # Missing signatures cannot establish distinct faults; count the
            # endpoint/status bucket once rather than inventing new identities.
            fault = (endpoint, status, item.get('fault_signature'))
            if fault not in self.fault_signatures:
                self.fault_signatures.add(fault)
                if endpoint in self.seeds:
                    self.fault_followups[endpoint] = min(2, self.fault_followups[endpoint] + 1)
        if self.phase == 'mutation' or ('mutation' in self.phase and item.get('parameter_categories', {}).get('__sequence__', {}).get('mutated')):
            self.mutation_counts[endpoint] += 1
            self.epoch_counts[endpoint] += 1
            if item.get('is_5xx') and observation not in self.observations:
                # New server fault signature: real progress, keep exploring.
                self.stagnation[endpoint] = 0
            elif item.get('is_2xx'):
                # A 2xx mutation may execute a new application branch even when the
                # HTTP status stays 2xx.  We treat a *repeated* 2xx (same endpoint
                # and response body) as stagnation to avoid burning the whole
                # mutation budget on the same successful request, but a 2xx with a
                # new response body is still progress.
                response_hash = str(item.get('response_data') or '')[:500]
                new_key = (endpoint, '2xx', response_hash)
                if new_key in self.observations:
                    self.stagnation[endpoint] += 1
                else:
                    self.stagnation[endpoint] = 0
                self.observations.add(new_key)
            else:
                self.stagnation[endpoint] += 1
        self.observations.add(observation)

    def next_mutation(self, endpoint, operation, with_edits=False, scope=None):
        seed = self.seeds.get(endpoint)
        if seed is None:
            self.stops['no_successful_seed'] += 1
            return None
        if self.epoch_counts[endpoint] >= int(os.getenv('REST_LEAGUE_MUTATION_MAX_PER_OPERATION', '24')):
            self.stops['operation_budget_exhausted'] += 1
            return None
        if self.stagnation[endpoint] >= int(os.getenv('REST_LEAGUE_MUTATION_STAGNATION', '8')):
            self.stops['no_new_status_or_fault'] += 1
            return None
        choices = []
        for field, path, schema, required, location in mutation_sites(operation, seed):
            name = str(path[-1]) if path else 'body'
            categories = categories_for_parameter(
                name, schema, required, variant=self.epochs[endpoint]
            )
            # Alternate valid and invalid classes rather than exhausting one class first.
            categories.sort(key=lambda category: category.valid, reverse=True)
            valid = [c for c in categories if c.valid]
            invalid = [c for c in categories if not c.valid]
            if os.getenv('REST_LEAGUE_VALID_ONLY_MUTATION', 'false').lower() == 'true':
                invalid = []
            for index in range(max(len(valid), len(invalid))):
                for group in (valid, invalid):
                    if index < len(group):
                        choices.append((field, path, location, group[index]))
        # Round-robin over parameter sites as well as categories.
        grouped = {}
        for choice in choices:
            grouped.setdefault((choice[0], choice[1]), []).append(choice)
        choices = [group[index] for index in range(max((len(g) for g in grouped.values()), default=0))
                   for group in grouped.values() if index < len(group)]
        for _ in range(len(choices)):
            index = self.cursors[endpoint] % len(choices)
            self.cursors[endpoint] += 1
            field, path, location, category = choices[index]
            candidate = copy.deepcopy(seed)
            if category.value == MISS_KEY and not path:
                continue
            try:
                candidate[field] = replace_at(candidate[field], path, category.value)
            except (KeyError, TypeError, IndexError):
                continue
            if field == 'headers' and path and path[0] in candidate[field]:
                candidate[field][path[0]] = str(candidate[field][path[0]])
            edits = [(field, path, category.value)]

            # Rotate paired parameter combinations across epochs.
            if with_edits and self.cursors[endpoint] % 4 == 0:
                others = choices[index+1:] + choices[:index]
                shift = self.epochs[endpoint] % max(1, len(others))
                others = others[shift:] + others[:shift]
                for other_field, other_path, _, other_category in others:
                    if (other_field, other_path) == (field, path):
                        continue
                    if other_field == field and (
                        other_path[:len(path)] == path
                        or path[:len(other_path)] == other_path
                    ):
                        continue
                    if other_category.value == MISS_KEY and not other_path:
                        continue
                    try:
                        candidate[other_field] = replace_at(
                            candidate[other_field],
                            other_path,
                            other_category.value,
                        )
                    except (KeyError, TypeError, IndexError):
                        continue
                    if (
                        other_field == 'headers'
                        and other_path
                        and other_path[0] in candidate[other_field]
                    ):
                        candidate[other_field][other_path[0]] = str(
                            candidate[other_field][other_path[0]]
                        )
                    edits.append(
                        (other_field, other_path, other_category.value)
                    )
                    break

            # Include every edit before deduplication. New runtime IDs alone
            # must not make an old parameter trial appear new.
            digest = (
                fingerprint({'scope': scope, 'edits': edits})
                if with_edits else fingerprint(candidate)
            )
            if (
                fingerprint(candidate) == fingerprint(seed)
                or (endpoint, digest) in self.seen
            ):
                continue
            self.seen.add((endpoint, digest))
            candidate['parameter_categories'] = {
                location + ':' + '.'.join(map(str, path)): {
                    'location': location,
                    'category_id': category.category_id,
                    'category_name': category.name,
                    'valid': category.valid,
                    'reason': category.reason,
                    'source': 'successful_request_parameter_mutation',
                    'seed_fingerprint': fingerprint(seed),
                    'candidate_value': category.value,
                    'edit_count': len(edits),
                }
            }
            if with_edits:
                candidate['_sequence_edits'] = edits
            return candidate
        self.stops['candidates_exhausted'] += 1
        return None

    def summary(self, catalog):
        return {'covered_2xx': sorted(catalog.covered_2xx),
                'accepted_scenarios': len(catalog.scenario_signatures),
                'rejected_scenarios': catalog.scenario_rejections,
                'successful_seed_count': len(self.seeds),
                'phase_request_counts': dict(self.phase_counts),
                'mutation_stop_counts': dict(self.stops),
                'mutation_epochs': dict(self.epochs),
                'requests_5xx': sum(self.fault_requests.values()),
                'fault_signature_count': len(self.fault_signatures),
                'requests_5xx_by_operation': dict(self.fault_requests),
                'successful_sequence_count': len(
                    getattr(getattr(catalog, 'sequence_corpus', None),
                            'plans', {})
                ),
                'unique_mutation_requests': len(self.seen)}


runtime = AdaptiveTesting()
