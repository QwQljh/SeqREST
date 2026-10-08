"""Exploration probability equals successful operation coverage each round."""
import math


class SchedulingPolicy:
    MODES = ('coverage', 'exploration')

    def __init__(self):
        self.rounds = dict.fromkeys(self.MODES, 0)
        self.statuses = set()
        self.faults = set()
        self.last = {}

    def probability(self, coverage):
        if not math.isfinite(coverage):
            return 0.0
        return max(0.0, min(1.0, coverage))

    def choose(self, coverage, draw):
        probability = self.probability(coverage)
        mode = 'exploration' if probability == 1 or draw < probability else 'coverage'
        self.last = {'mode': mode, 'exploration_probability': probability,
                     'coverage_ratio': coverage}
        return mode

    def observe(self, mode, requests, new_operations, elapsed_seconds):
        new_statuses = 0
        new_faults = 0
        for item in requests:
            endpoint = item.get('api_endpoint')
            status = item.get('response_code')
            if not endpoint or not isinstance(status, int):
                continue
            # Diagnostics do not affect the coverage-based mode probability.
            status_key = (endpoint, status)
            if status_key not in self.statuses:
                self.statuses.add(status_key)
                if 200 <= status < 300 or 500 <= status < 600:
                    new_statuses += 1
            signature = item.get('fault_signature')
            if 500 <= status < 600 and signature:
                fault = (endpoint, status, signature)
                if fault not in self.faults:
                    self.faults.add(fault)
                    new_faults += 1
        self.rounds[mode] += 1
        return dict(self.last, rounds=dict(self.rounds),
                    new_operations=new_operations, new_faults=new_faults,
                    new_statuses=new_statuses, requests=len(requests),
                    elapsed_seconds=round(elapsed_seconds, 3))


def get_policy(catalog):
    if not hasattr(catalog, 'scheduling_policy'):
        catalog.scheduling_policy = SchedulingPolicy()
    return catalog.scheduling_policy
