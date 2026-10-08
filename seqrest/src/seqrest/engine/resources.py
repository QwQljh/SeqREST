"""OpenAPI producer inference adapted from the WeChat resource bootstrap.

Only prepare dependencies of selected targets; bindings remain parent-aware.
"""
import re
import seqrest.engine.sequence as sequence
from seqrest.engine.sequence import key, singular, setting


def role(operation):
    method, path = operation.split(' ', 1)
    words = set(re.findall(r'[a-z]+', path.lower()))
    if method != 'POST':
        return 'lister' if method == 'GET' and not path.endswith('}') else 'consumer'
    if words & {'search', 'query', 'filter'}:
        return 'search'
    if words & {'login', 'signin', 'token', 'refresh'}:
        return 'authentication'
    if words & {'register', 'signup'}:
        return 'registration'
    return 'creator'


class ResourcePreparation:
    def __init__(self, runner):
        self.runner = runner
        self.catalog = runner.catalog
        self.attempted = set()
        self.remaining = setting('REST_LEAGUE_RESOURCE_BOOTSTRAP_REQUESTS', 8)

    def body_dependencies(self, target):
        cache = getattr(self.catalog, 'body_resource_dependencies', {})
        if target in cache:
            return cache[target]
        operation = self.catalog.resolve(self.catalog.api_swagger_map[target])
        schemas = [media.get('schema', {}) for media in operation.get('requestBody', {}).get('content', {}).values()]
        schemas += [p.get('schema', {}) for p in operation.get('parameters', []) if p.get('in') == 'body']
        dependencies = []
        def walk(schema, path=()):
            for name in schema.get('required', []):
                child = schema.get('properties', {}).get(name, {})
                location = path + (name,)
                if child.get('properties'):
                    walk(child, location)
                else:
                    resource_key = key(name)
                    if resource_key == 'id' and path:
                        resource_key = singular(path[-1]) + 'id'
                    if resource_key != 'id' and resource_key.endswith('id') and self.producers(target, resource_key):
                        dependencies.append((location, resource_key))
        for schema in schemas[:1]:
            walk(schema)
        cache[target] = dependencies
        self.catalog.body_resource_dependencies = cache
        return dependencies

    def send_producer(self, producer, target, stack=()):
        if producer in stack or producer in self.attempted or not self.remaining or not self.runner.can_send():
            return False
        failures = getattr(self.catalog, 'resource_producer_failures', {})
        if failures.get(producer, 0) >= 3:
            return False
        self.attempted.add(producer)
        self.prepare(producer, stack + (target,))
        if not self.remaining or not self.runner.can_send():
            return False
        self.remaining -= 1
        attempts = getattr(self.catalog, 'resource_producer_attempts', {})
        variant = attempts.get(producer, 0)
        attempts[producer] = variant + 1
        self.catalog.resource_producer_attempts = attempts
        plan = {'goal': 'Prepare dependency for ' + target, 'steps': [
            {'id': 'prepare', 'operation': producer, 'variant': variant * 5}]}
        sent, _, _ = self.runner.execute(plan, 'resource_preparation')
        succeeded = any(item.get('is_2xx') for item in sent)
        if succeeded:
            done = getattr(self.catalog, 'resource_bootstrap_success', set())
            done.add(producer)
            self.catalog.resource_bootstrap_success = done
        else:
            failures[producer] = failures.get(producer, 0) + 1
            self.catalog.resource_producer_failures = failures
        return succeeded

    def bind_body(self, target, payload, parents):
        if not isinstance(payload, dict):
            return payload
        for path, resource_key in self.body_dependencies(target):
            value = sequence.resources.select(resource_key, parents)
            if value is None:
                raise sequence.BindingError('No live resource for body ' + '.'.join(path) + ' in ' + target)
            current = payload
            for name in path[:-1]:
                current = current.setdefault(name, {})
            current[path[-1]] = value
        return payload

    def producers(self, target, name):
        """Prefer the exact collection for this path parameter over name guesses."""
        path = target.split(' ', 1)[1]
        parts = path.strip('/').split('/')
        collection = None
        for index, part in enumerate(parts):
            if part == '{' + name + '}' and index:
                collection = '/' + '/'.join(parts[:index])
                break
        candidates = []
        for operation in self.catalog.api_swagger_map:
            method, producer_path = operation.split(' ', 1)
            producer_role = role(operation)
            if operation == target or producer_role not in ('creator', 'lister', 'registration', 'authentication'):
                continue
            exact = producer_path == collection if collection else False
            typed = key(name) in (singular(producer_path.split('/')[-1]) + 'id', singular(producer_path.split('/')[-1]) + 'name')
            if not exact and not typed:
                continue
            candidates.append((int(exact), int(method == 'POST'), operation))
        return [row[2] for row in sorted(candidates, reverse=True)]

    def prepare(self, target, stack=()):
        if len(stack) >= setting('REST_LEAGUE_RESOURCE_DEPENDENCY_DEPTH', 3) or target in stack:
            return False
        path = target.split(' ', 1)[1]
        parents = {}
        changed = False
        parts = path.strip('/').split('/')
        for index, part in enumerate(parts):
            if not part.startswith('{'):
                continue
            name = part[1:-1]
            resource_key = singular(parts[index - 1]) + ('name' if 'name' in name.lower() else 'id')
            value = sequence.resources.select(resource_key, parents)
            if value is None:
                for producer in self.producers(target, name):
                    succeeded = self.send_producer(producer, target, stack)
                    changed = changed or succeeded
                    value = sequence.resources.select(resource_key, parents)
                    if value is not None:
                        break
            if value is None:
                return changed  # Never create a child with a guessed parent ID.
            parents[resource_key] = value
        for _, resource_key in self.body_dependencies(target):
            if sequence.resources.select(resource_key, parents) is None:
                for producer in self.producers(target, resource_key):
                    changed = self.send_producer(producer, target, stack) or changed
                    if sequence.resources.select(resource_key, parents) is not None:
                        break
        return changed

    def bootstrap(self, targets):
        for target in targets:
            if not self.remaining or not self.runner.can_send():
                break
            if target in self.catalog.api_swagger_map:
                self.prepare(target)
        # Proactively populate creators even when targets (e.g. POST search)
        # expose no path IDs. Registration and authentication precede creators.
        priorities = {'registration': 0, 'authentication': 1, 'creator': 2}
        candidates = sorted(self.catalog.api_swagger_map,
                            key=lambda op: (priorities.get(role(op), 9), op.count('{'), op))
        for producer in candidates:
            if not self.remaining or not self.runner.can_send():
                break
            if role(producer) not in priorities:
                continue
            if producer in getattr(self.catalog, 'resource_bootstrap_success', set()):
                continue
            if any(record['source'] == producer and record['state'] != 'deleted'
                   for record in sequence.resources.records.values()):
                continue
            self.send_producer(producer, 'resource bootstrap')
