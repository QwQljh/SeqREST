"""Run-scoped verified sequences and model-independent exploration."""
import copy
import hashlib
import json
from collections import OrderedDict

from seqrest.engine.sequence import setting, validate_plans


def signature(plan):
    return hashlib.sha256(
        json.dumps(plan['steps'], sort_keys=True).encode()
    ).hexdigest()


class SequenceCorpus:
    def __init__(self):
        self.plans = OrderedDict()
        self.cursor = 0
        self.rounds = 0
        self.state_trials = set()

    def remember(self, plan, sent):
        # Save only the contiguous prefix whose individual steps succeeded.
        # A later successful request cannot repair an earlier failed producer.
        prefix = []
        for step, item in zip(plan['steps'], sent):
            meta = item.get('parameter_categories', {}).get('__sequence__', {})
            if (
                meta.get('step') != step['id']
                or item.get('api_endpoint') != step['operation']
                or not item.get('is_2xx')
                or step.get('probe_invalid_resource')
            ):
                break
            prefix.append(copy.deepcopy(step))

        if not prefix:
            return

        verified = {'goal': plan['goal'], 'steps': prefix}
        key = signature(verified)
        if key in self.plans:
            return
        self.plans[key] = verified
        limit = max(1, setting('REST_LEAGUE_SUCCESS_SEQUENCE_LIMIT', 64))
        while len(self.plans) > limit:
            self.plans.popitem(last=False)
        print('SUCCESS SEQUENCE SAVED: ' + json.dumps({
            'signature': key[:12], 'steps': len(prefix),
            'pool_size': len(self.plans),
        }))

    def state_variants(self, plan, catalog):
        candidates = []
        steps = plan['steps']

        # Change adjacent read order only when response references remain valid.
        for index in range(len(steps) - 1):
            if all(
                step['operation'].split(' ', 1)[0] in ('GET', 'HEAD')
                for step in steps[index:index+2]
            ):
                changed = copy.deepcopy(plan)
                changed['steps'][index:index+2] = list(
                    reversed(changed['steps'][index:index+2])
                )
                changed['goal'] = 'Read-order comparison: ' + plan['goal']
                candidates.append(changed)

        # Re-read a previously queried resource after later state changes.
        # Examples include update -> read and delete -> read.
        for index, step in enumerate(steps[:-1]):
            if step['operation'].split(' ', 1)[0] not in ('GET', 'HEAD'):
                continue
            if not any(
                later['operation'].split(' ', 1)[0]
                not in ('GET', 'HEAD', 'OPTIONS')
                for later in steps[index+1:]
            ):
                continue
            changed = copy.deepcopy(plan)
            extra = copy.deepcopy(step)
            names = {s['id'] for s in steps}
            ident = 'state_check'
            while ident in names:
                ident += '_x'
            extra['id'] = ident
            changed['steps'].append(extra)
            changed['goal'] = 'Read after state change: ' + plan['goal']
            candidates.append(changed)

        maximum = max(1, setting('SCENARIO_MAX_STEPS', 16))
        allowed = {step['operation'] for step in steps}
        for candidate in candidates:
            key = signature(candidate)
            if key in self.state_trials:
                continue
            valid = validate_plans(
                {'scenarios': [candidate]},
                catalog, 1, maximum, allowed=allowed,
            )
            if valid:
                self.state_trials.add(key)
                yield valid[0]

    def run(self, runner, feedback):
        self.rounds += 1
        if not self.plans or not runner.can_send():
            return False, 0

        rows = list(self.plans.values())
        count = min(
            len(rows),
            max(1, setting('REST_LEAGUE_SAVED_SEQUENCE_BATCH', 3)),
        )
        # Reserve at most one slot for a newly discovered server fault. The
        # remaining slots rotate normally, so repeated faults cannot monopolize.
        credits = runner.adaptive.fault_followups
        rewarded = max(rows, key=lambda plan: sum(credits[step['operation']] for step in plan['steps']))
        selected = []
        if any(credits[step['operation']] for step in rewarded['steps']):
            selected.append(rewarded)
            for operation in {step['operation'] for step in rewarded['steps']}:
                if credits[operation]:
                    credits[operation] -= 1
        scanned = 0
        while len(selected) < count and scanned < len(rows):
            plan = rows[self.cursor % len(rows)]
            self.cursor = (self.cursor + 1) % len(rows)
            scanned += 1
            if plan not in selected:
                selected.append(plan)
        batch_start = len(runner.tools.all_request_sequence)
        progress = False

        for plan in selected:
            if not runner.can_send():
                break
            start = len(runner.tools.all_request_sequence)

            # Each mutation executes its original setup and resolves fresh IDs.
            runner.explore(plan)
            if (
                len(runner.tools.all_request_sequence) == start
                and runner.can_send()
            ):
                runner.adaptive.begin_epoch(
                    [step['operation'] for step in plan['steps']]
                )
                runner.explore(plan)

            if runner.can_send():
                for changed in self.state_variants(plan, runner.catalog):
                    runner.execute(changed, phase='state_exploration')
                    break

            requests = runner.tools.all_request_sequence[start:]
            if requests:
                progress = feedback.observe_batch(plan, requests) or progress

        sent = len(runner.tools.all_request_sequence) - batch_start
        print('SAVED SEQUENCE EXPLORATION: ' + json.dumps({
            'pool_size': len(self.plans), 'selected': count, 'sent': sent,
        }))
        return progress, sent


def get_corpus(catalog):
    if not hasattr(catalog, 'sequence_corpus'):
        catalog.sequence_corpus = SequenceCorpus()
    return catalog.sequence_corpus
