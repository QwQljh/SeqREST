"""Request plans with response references, related resource records and scoped replay.

No API-specific names, privileged side channel, or live code-coverage claims.
"""
import copy
import hashlib
import json
import os
import re
import time
import uuid
from urllib.parse import quote, unquote


def setting(name, default):
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def key(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def singular(value):
    value = key(value)
    return value[:-3] + 'y' if value.endswith('ies') else value[:-1] if value.endswith('s') else value


class BindingError(ValueError):
    pass


def response_value(value, outputs):
    if isinstance(value, dict) and set(value) == {'$ref'}:
        parts = str(value['$ref']).split('.')
        if len(parts) < 3 or parts[0] not in outputs:
            raise BindingError('Missing successful producer: ' + str(value['$ref']))
        result = outputs[parts[0]]
        try:
            for part in parts[1:]:
                result = result[int(part)] if isinstance(result, list) else result[part]
        except (KeyError, IndexError, TypeError, ValueError):
            raise BindingError('Response reference not found: ' + str(value['$ref']))
        if result is None:
            raise BindingError('Null response reference: ' + str(value['$ref']))
        return copy.deepcopy(result)
    if isinstance(value, dict):
        return {k: response_value(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [response_value(v, outputs) for v in value]
    return copy.deepcopy(value)


class ResourceStore:
    """Entity records keep IDs together with parent IDs and successful state evidence."""
    def __init__(self):
        self.records = {}
        self.serial = 0

    def path_values(self, template, actual):
        names = re.findall(r'\{([^{}]+)\}', template)
        pattern = re.sub(r'\\\{[^}]+\\\}', '([^/]+)', re.escape(template))
        match = re.fullmatch(pattern, actual.split('?')[0])
        return dict(zip(names, [unquote(v) for v in match.groups()])) if match else {}

    def observe(self, item, catalog):
        if not item.get('is_2xx'):
            return
        endpoint = item.get('api_endpoint')
        if endpoint not in catalog.api_swagger_map:
            return
        method, template = endpoint.split(' ', 1)
        parents = self.path_values(template, item.get('api', template))
        route = template.strip('/').split('/')
        entity = singular(route[-2] if route[-1].startswith('{') else route[-1])
        seq = item.get('parameter_categories', {}).get('__sequence__', {}).get('sequence_id')
        self.serial += 1
        if method == 'DELETE' and route[-1].startswith('{'):
            own = route[-1][1:-1]
            for record in self.records.values():
                if record['type'] == entity and str(record['id']) == str(parents.get(own)) and self.matches(record, parents):
                    record['state'] = 'deleted'
            return
        body = item.get('full_response_data', item.get('response_data'))
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                body = None

        def walk(value, kind, ancestors, depth=0):
            if depth > 8:
                return
            if isinstance(value, list):
                for child in value[:50]:
                    walk(child, kind, ancestors, depth + 1)
            elif isinstance(value, dict):
                fields = {key(k): v for k, v in value.items() if isinstance(v, (str, int, float, bool))}
                identity = next((fields[k] for k in (kind+'id', 'id', kind+'name', 'name') if k in fields), None)
                # Generic wrappers (e.g. {response: UUID}) still have a typed source.
                if identity is None and isinstance(value.get('response'), str) and re.fullmatch(r'[0-9a-fA-F-]{36}', value['response']):
                    identity = value['response']
                child_parents = dict(ancestors)
                if identity is not None:
                    aliases = {kind+'id', kind+'name', kind}
                    aliases.update(k for k, v in fields.items() if v == identity and k != 'id')
                    record_id = json.dumps([kind, str(identity), sorted((k, str(v)) for k, v in ancestors.items() if key(k) not in aliases)], sort_keys=True)
                    old = self.records.get(record_id, {})
                    record = dict(old, type=kind, id=identity, aliases=sorted(aliases),
                                  parents=dict(ancestors), source=endpoint, state='live',
                                  observed=self.serial, sequence_id=seq)
                    record['attributes'] = dict(old.get('attributes', {}), **fields)
                    record['evidence'] = {'method': method, 'status': item.get('response_code')}
                    self.records[record_id] = record
                    child_parents[kind+'id'] = identity
                for name, child in value.items():
                    if isinstance(child, (dict, list)):
                        child_kind = kind if name in ('data', 'content', 'items', 'response', '_embedded') else singular(name)
                        walk(child, child_kind, child_parents, depth + 1)

        walk(body, entity, parents)
        location = item.get('response_headers', {}).get('Location')
        if location and method == 'POST':
            identity = unquote(str(location).rstrip('/').split('/')[-1])
            walk({'id': identity}, entity, parents)
        limit = setting('REST_LEAGUE_RESOURCE_RECORD_LIMIT', 1000)
        while len(self.records) > max(1, limit):
            oldest = min(self.records, key=lambda k: self.records[k]['observed'])
            del self.records[oldest]

    @staticmethod
    def matches(record, parents):
        own = set(record['aliases']) | {'id'}
        known = {key(k): str(v) for k, v in record['parents'].items()}
        return all(key(k) in own or (key(k) in known and str(v) == known[key(k)]) for k, v in parents.items())

    def select(self, name, parents, sequence_id=None):
        candidates = [r for r in self.records.values() if key(name) in r['aliases']
                      and r['state'] != 'deleted' and self.matches(r, parents)]
        candidates.sort(key=lambda r: (r.get('sequence_id') == sequence_id if sequence_id else False,
                                       len(set(map(key, parents)) & set(map(key, r['parents']))), r['observed']), reverse=True)
        return candidates[0]['id'] if candidates else None

    def context(self, limit=20):
        rows = sorted(self.records.values(), key=lambda r: r['observed'], reverse=True)
        return [{'type': r['type'], 'id': r['id'], 'parents': r['parents'], 'state': r['state'],
                 'attributes': {k: v for k, v in r.get('attributes', {}).items()
                                if not any(s in k for s in ('token', 'password', 'secret'))}}
                for r in rows if r['state'] != 'deleted'][:limit]


resources = ResourceStore()


def validate_plans(data, catalog, count, max_steps, allowed=None, errors=None):
    accepted = []
    errors = [] if errors is None else errors
    if not isinstance(data, dict) or not isinstance(data.get('scenarios'), list):
        errors.append('Expected a JSON object with a scenarios array')
        return accepted
    if not data['scenarios']:
        errors.append('scenarios array is empty')
    for raw in data.get('scenarios', [])[:count]:
        if not isinstance(raw, dict) or not isinstance(raw.get('steps'), list) or not 1 <= len(raw['steps']) <= max_steps:
            errors.append(f'Each scenario needs 1-{max_steps} steps in an array')
            continue
        steps, previous, bad = [], set(), False
        for index, value in enumerate(raw['steps']):
            if not isinstance(value, dict):
                errors.append(f'Step {index+1} must be an object')
                bad = True
                break
            step = copy.deepcopy(value)
            op = step.get('operation')
            if isinstance(op, str) and op not in catalog.api_swagger_map:
                try:
                    normalized = catalog.normalize_endpoint(op)['endpoint']
                    if normalized in catalog.api_swagger_map and isinstance(step.get('path_params', {}), dict):
                        template = normalized.split(' ', 1)[1]
                        actual = op.split(' ', 1)[1]
                        step['path_params'] = dict(ResourceStore().path_values(template, actual), **step.get('path_params', {}))
                        step['operation'] = op = normalized
                except (ValueError, IndexError):
                    pass
            ident = step.setdefault('id', 's'+str(index+1))
            if op not in catalog.api_swagger_map or (allowed is not None and op not in allowed) or not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', ident) or ident in previous:
                errors.append(f'Step {index+1}: unknown/unselected operation or invalid/duplicate id: {op!r}, {ident!r}')
                bad = True
                break
            if any(not isinstance(step.get(k, {}), dict) for k in ('path_params', 'query', 'headers')):
                errors.append(f'Step {ident}: path_params, query and headers must be objects, not null or arrays')
                bad = True
                break
            def refs_valid(v):
                if isinstance(v, dict):
                    if '$ref' in v:
                        ref = v['$ref']
                        return set(v) == {'$ref'} and isinstance(ref, str) and ref.split('.')[0] in previous
                    return all(refs_valid(x) for x in v.values())
                return all(refs_valid(x) for x in v) if isinstance(v, list) else True
            if not refs_valid({k: step[k] for k in ('path_params', 'query', 'headers', 'body') if k in step}):
                errors.append(f'Step {ident}: response reference must refer to an earlier step')
                bad = True
                break
            previous.add(ident)
            steps.append(step)
        if not bad:
            accepted.append({'goal': str(raw.get('goal', 'scenario'))[:240], 'steps': steps})
    return accepted


class SequenceRunner:
    def __init__(self, catalog, tools, adaptive, drain, deadline, max_requests):
        self.catalog, self.tools, self.adaptive, self.drain = catalog, tools, adaptive, drain
        self.deadline, self.max_requests = deadline, max_requests
        from seqrest.engine.resources import ResourcePreparation
        self.preparation = ResourcePreparation(self)

    def can_send(self):
        return len(self.tools.all_request_sequence) < self.max_requests and (self.deadline is None or time.time() < self.deadline)

    def arguments(self, step, outputs, sequence_id):
        op = step['operation']
        args = self.tools.build_request_from_openapi(op, variant=int(step.get('variant', 0)))
        values = response_value(step.get('path_params', {}), outputs)
        parents = {}
        template = op.split(' ', 1)[1]
        for name in re.findall(r'\{([^{}]+)\}', template):
            kind = name
            if key(name) == 'id':
                prefix = template.split('{'+name+'}')[0].rstrip('/').split('/')[-1]
                kind = singular(prefix)+'id'
            supplied = step.get('path_params', {}).get(name)
            explicit_ref = isinstance(supplied, dict) and '$ref' in supplied
            if not explicit_ref and not step.get('probe_invalid_resource', False):
                value = resources.select(kind, parents, sequence_id)
                if value is not None:
                    values[name] = value
            if name not in values:
                if key(name).endswith('id') or name == 'id':
                    raise BindingError('No live resource for '+name+' in '+op)
                param = next((p for p in self.catalog.api_swagger_map[op].get('parameters', []) if p.get('name') == name), {})
                values[name] = self.tools._schema_example(param.get('schema') or param, name, int(step.get('variant', 0)), True)
            parents[kind] = values[name]
        args['api'] = template
        for name, value in values.items():
            args['api'] = args['api'].replace('{'+name+'}', quote(str(value), safe=''))
        if '{' in args['api']:
            raise BindingError('Unbound path: '+args['api'])
        for field, source in (('params', 'query'), ('headers', 'headers'), ('payload', 'body')):
            if source in step:
                args[field] = response_value(step[source], outputs)
        if 'payload_type' in step:
            args['payload_type'] = step['payload_type']
        if 'body' not in step:
            args['payload'] = self.preparation.bind_body(op, args.get('payload'), parents)
        args['parameter_categories']['__sequence__'] = {'sequence_id': sequence_id, 'step': step['id'], 'operation': op}
        return args

    def execute(self, plan, phase='scenario', edits=None, outputs=None, only=None):
        outputs = {} if outputs is None else outputs
        execution = uuid.uuid4().hex[:12]
        sent, blocked = [], []
        print('SEQUENCE START: '+json.dumps({'id': execution, 'goal': plan['goal'], 'phase': phase}))
        for step in plan['steps']:
            if only is not None and step['id'] not in only:
                continue
            if not self.can_send():
                break
            try:
                try:
                    args = self.arguments(step, outputs, execution)
                except BindingError as exc:
                    if phase == "resource_preparation" or "No live resource" not in str(exc):
                        raise
                    self.preparation.prepare(step["operation"])
                    args = self.arguments(step, outputs, execution)
                if edits and step['id'] in edits:
                    for field, path, candidate in edits[step['id']]:
                        from seqrest.engine.mutation import replace_at
                        args[field] = replace_at(args[field], path, candidate)
                    args['parameter_categories']['__sequence__']['mutated'] = True
                self.adaptive.phase = phase
                self.tools.test_scenario.current_test_case = self.tools.test_scenario.TestCase(plan['goal'], step['operation'], phase, 'Observe response')
                start = len(self.tools.all_request_sequence)
                self.tools.do_request(**args)
                items = self.tools.all_request_sequence[start:]
                sent.extend(items)
                if items:
                    item = items[-1]
                    # Failed producers never provide IDs to dependent steps.
                    if item.get('is_2xx'):
                        body = item.get('full_response_data', item.get('response_data'))
                        if isinstance(body, str):
                            try:
                                body = json.loads(body)
                            except ValueError:
                                pass
                        outputs[step['id']] = {'body': body, 'headers': item.get('response_headers', {})}
                    else:
                        outputs.pop(step['id'], None)
                self.drain('Sequence '+phase)
            except (BindingError, ValueError, TypeError, KeyError) as exc:
                outputs.pop(step['id'], None)
                blocked.append({'step': step['id'], 'operation': step['operation'], 'reason': str(exc)})
                print('SEQUENCE BLOCKED: '+json.dumps(blocked[-1]))
        self.adaptive.phase = 'scenario'
        print('SEQUENCE END: '+json.dumps({'id': execution, 'sent': len(sent), 'blocked': len(blocked)}))
        if (
            only is None and not edits
            and phase in (
                'scenario', 'fallback_sequence',
                'sequence_replay', 'resource_preparation',
            )
        ):
            from seqrest.engine.corpus import get_corpus
            get_corpus(self.catalog).remember(plan, sent)
        return sent, blocked, outputs

    def prepare_resources(self, plan, sent):
        """Prepare missing dependencies; retain successful resources without blind copies."""
        for step in plan['steps']:
            self.preparation.prepare(step['operation'])

    def explore(self, plan):
        # Distinguish different symbolic plans without hashing fresh runtime IDs.
        scope = hashlib.sha256(
            json.dumps(plan['steps'], sort_keys=True).encode()
        ).hexdigest()
        # Valid prefix is recreated on every trial; edit values, never frozen IDs.
        for trial in range(setting('REST_LEAGUE_SEQUENCE_MUTATION_ROUNDS', 8)):
            if not self.can_send():
                return
            edits = {}
            for offset in range(len(plan['steps'])):
                step = plan['steps'][(trial+offset) % len(plan['steps'])]
                candidate = self.adaptive.next_mutation(step['operation'], self.catalog.api_swagger_map[step['operation']], with_edits=True, scope=scope)
                if candidate:
                    changes = candidate.get('_sequence_edits', [])
                    if changes:
                        edits[step['id']] = changes
                        break
            if edits:
                self.execute(plan, phase='sequence_mutation', edits=edits)
        # Independent probes still replay prerequisites, then mutate one target.
        for probe in range(setting('REST_LEAGUE_SCOPED_SINGLE_MUTATION_ROUNDS', 12)):
            if not self.can_send():
                return
            step = plan['steps'][probe % len(plan['steps'])]
            candidate = self.adaptive.next_mutation(step['operation'], self.catalog.api_swagger_map[step['operation']], with_edits=True, scope=scope)
            if candidate and candidate.get('_sequence_edits'):
                index = plan['steps'].index(step)
                prefix = dict(plan, steps=plan['steps'][:index+1])
                self.execute(prefix, phase='single_mutation', edits={step['id']: candidate['_sequence_edits']})

    def fallback(self, plan, sent, blocked, outputs):
        failures = {i.get('api_endpoint') for i in sent if not i.get('is_2xx')}
        failures.update(i['operation'] for i in blocked)
        successes = {i.get('api_endpoint') for i in sent if i.get('is_2xx')}
        failures -= successes
        # Single retries retain current successful producers and their exact IDs.
        for retry in range(setting('REST_LEAGUE_SCOPED_FALLBACK_ROUNDS', 4)):
            if not failures or not self.can_send():
                break
            retry_plan = copy.deepcopy(plan)
            for step in retry_plan['steps']:
                if step['operation'] in failures:
                    step['variant'] = retry
                    # Keep explicit dynamic bindings, use schema variants for ordinary values.
                    for field in ('body', 'query'):
                        if field in step and '$ref' not in json.dumps(step[field]):
                            step.pop(field)
            only = {s['id'] for s in retry_plan['steps'] if s['operation'] in failures}
            items, waiting, outputs = self.execute(retry_plan, 'fallback_single', outputs=outputs, only=only)
            failures -= {i.get('api_endpoint') for i in items if i.get('is_2xx')}
        # Full-sequence retries also reproduce setup and refresh all bindings.
        for retry in range(setting('REST_LEAGUE_SEQUENCE_FALLBACK_ROUNDS', 3)):
            if not failures or not self.can_send():
                break
            retry_plan = copy.deepcopy(plan)
            for step in retry_plan['steps']:
                if step['operation'] in failures:
                    step['variant'] = retry + 1
                    for field in ('body', 'query'):
                        if field in step and '$ref' not in json.dumps(step[field]):
                            step.pop(field)
            items, _, _ = self.execute(retry_plan, 'fallback_sequence')
            failures -= {i.get('api_endpoint') for i in items if i.get('is_2xx')}
        return failures
