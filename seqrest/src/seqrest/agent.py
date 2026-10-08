import json
import random
import re
import time
import uuid
import sys
import os
import ast
import hashlib

import autogen
from autogen import ConversableAgent
from autogen.oai.client import OpenAIWrapper

from seqrest.http import do_request, record_result
import seqrest.http as tools
import seqrest.scenario as test_scenario
import seqrest.config as global_vars_funcs_configs
import seqrest.engine.mutation as adaptive_testing
import seqrest.planning.feedback as coverage_feedback


sys.setrecursionlimit(5000)


def env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid integer env ignored: {name}={value}")
        return default


def stable_seed_from_env() -> int:
    seed_text = os.getenv("REST_LEAGUE_SEED") or os.getenv("RUN") or global_vars_funcs_configs.CONFIG_SYSTEM_NAME
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


REST_LEAGUE_MODE = env_flag("REST_LEAGUE_MODE")
REST_LEAGUE_SEED_ALL_ENDPOINTS = env_flag("REST_LEAGUE_SEED_ALL_ENDPOINTS", "false")
REST_LEAGUE_DIRECT_FALLBACK = env_flag("REST_LEAGUE_DIRECT_FALLBACK", "true")
REST_LEAGUE_SKIP_LLM = env_flag("REST_LEAGUE_SKIP_LLM")
REST_LEAGUE_MUTATION_ROUNDS = max(1, env_int("REST_LEAGUE_MUTATION_ROUNDS", 1))
REST_LEAGUE_COVERAGE_RETRY_ROUNDS = max(0, env_int("REST_LEAGUE_COVERAGE_RETRY_ROUNDS", 3))
REST_LEAGUE_FAILED_RETRY_ROUNDS = max(0, env_int("REST_LEAGUE_FAILED_RETRY_ROUNDS", 2))
REST_LEAGUE_LATE_COVERAGE_RETRY_ROUNDS = max(0, env_int("REST_LEAGUE_LATE_COVERAGE_RETRY_ROUNDS", 0))
REST_LEAGUE_NEGATIVE_MUTATION_INTERVAL = max(0, env_int("REST_LEAGUE_NEGATIVE_MUTATION_INTERVAL", 5))
REST_LEAGUE_SKIP_PERSISTENT_4XX = env_flag("REST_LEAGUE_SKIP_PERSISTENT_4XX", "false")
REST_LEAGUE_SUCCESS_REPLAY_ROUNDS = max(0, env_int("REST_LEAGUE_SUCCESS_REPLAY_ROUNDS", 0))
REST_LEAGUE_SCORE_ORDERING = env_flag("REST_LEAGUE_SCORE_ORDERING", "false")
REST_LEAGUE_RESET_RESOURCE_POOL_EACH_LOOP = env_flag("REST_LEAGUE_RESET_RESOURCE_POOL_EACH_LOOP", "false")
REST_LEAGUE_MUTATE_DESTRUCTIVE_ENDPOINTS = env_flag("REST_LEAGUE_MUTATE_DESTRUCTIVE_ENDPOINTS", "false")
REST_LEAGUE_VALID_ONLY_MUTATION = env_flag("REST_LEAGUE_VALID_ONLY_MUTATION", "false")
REST_LEAGUE_VALID_SEED = env_flag("REST_LEAGUE_VALID_SEED", "true")
REST_LEAGUE_REPEAT_5XX_LIMIT = max(0, env_int("REST_LEAGUE_REPEAT_5XX_LIMIT", 25))
REST_LEAGUE_STATIC_GET_5XX_LIMIT = max(0, env_int("REST_LEAGUE_STATIC_GET_5XX_LIMIT", 3))
REST_LEAGUE_FAULT_DIVERSITY_REQUESTS = max(0, env_int("REST_LEAGUE_FAULT_DIVERSITY_REQUESTS", 0))
REST_LEAGUE_FAULT_DIVERSITY_STOP_AFTER_NO_5XX = max(0, env_int("REST_LEAGUE_FAULT_DIVERSITY_STOP_AFTER_NO_5XX", 30))
REST_LEAGUE_VALID_TRAFFIC_ROUNDS = max(0, env_int("REST_LEAGUE_VALID_TRAFFIC_ROUNDS", 0))
REST_LEAGUE_BALANCED_MODE = env_flag("REST_LEAGUE_BALANCED_MODE", "false")
REST_LEAGUE_STATIC_GET_REPLAY_LIMIT = max(0, env_int("REST_LEAGUE_STATIC_GET_REPLAY_LIMIT", 20))
REST_LEAGUE_SEED = stable_seed_from_env()
SCORING_ONLY_MODE = env_flag("SCORING_ONLY_MODE", "true")
random.seed(REST_LEAGUE_SEED)

FULL_COVERAGE_MAX_BATCHES = max(1, env_int('REST_LEAGUE_FULL_COVERAGE_MAX_BATCHES', 12))
FULL_COVERAGE_STAGNATION_BATCHES = max(1, env_int('REST_LEAGUE_FULL_COVERAGE_STAGNATION_BATCHES', 3))
VERIFICATION_MIN_REQUESTS = max(0, env_int('REST_LEAGUE_VERIFICATION_MIN_REQUESTS', 100))
planning_feedback = coverage_feedback.CoverageFeedback(global_vars_funcs_configs.catalog)
global_vars_funcs_configs.catalog.coverage_feedback = planning_feedback


llm_config = {"config_list": [
        {
            "cache_seed": REST_LEAGUE_SEED,
            "model": global_vars_funcs_configs.CONFIG_LLM_MODEL,
            "api_key": global_vars_funcs_configs.CONFIG_LLM_API_KEY,
            "timeout": env_int("LLM_TIMEOUT_SECONDS", 60),
            "max_retries": 1,
            "temperature": 0,
            "max_tokens": env_int("LLM_MAX_OUTPUT_TOKENS", 2048),
        },
    ]
}
if global_vars_funcs_configs.CONFIG_LLM_BASE_URL:
    llm_config["config_list"][0]["base_url"] = global_vars_funcs_configs.CONFIG_LLM_BASE_URL
if 'qwen3' in global_vars_funcs_configs.CONFIG_LLM_MODEL.lower():
    llm_config['config_list'][0]['extra_body'] = {
        'chat_template_kwargs': {'enable_thinking': False}}

test_scenario_response_message = ""

STRICT_JSON_TOOL_CALL_INSTRUCTION = """
CRITICAL: When calling any tool or function, the arguments must be strictly valid JSON.
Use double quotes for all JSON object keys and string values.
Do not use single quotes, Python dict syntax, markdown code fences, comments, trailing commas, or unescaped control characters.
If a string contains quotes, backslashes, or newlines, escape them according to JSON rules.
The function arguments must be a JSON object only.
"""


def _coerce_tool_arguments_to_json(arguments):
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(arguments)
            except (ValueError, SyntaxError):
                print(f"[JSON_FIX] Replaced invalid tool arguments with empty object: {arguments[:200]}")
                return "{}"
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
        print(f"[JSON_FIX] Replaced non-object tool arguments with empty object: {type(parsed).__name__}")
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    print(f"[JSON_FIX] Replaced unsupported tool arguments type with empty object: {type(arguments).__name__}")
    return "{}"


def _sanitize_message_tool_arguments(message):
    if not isinstance(message, dict):
        return message

    function_call = message.get("function_call")
    if isinstance(function_call, dict) and "arguments" in function_call:
        function_call["arguments"] = _coerce_tool_arguments_to_json(function_call.get("arguments"))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict) and "arguments" in function:
                function["arguments"] = _coerce_tool_arguments_to_json(function.get("arguments"))

    return message


def _install_autogen_json_argument_sanitizer():
    if getattr(OpenAIWrapper.create, "_logiagent_json_sanitizer", False):
        return

    original_create = OpenAIWrapper.create

    def create_with_json_sanitizer(self, **config):
        sanitized_config = dict(config)
        messages = sanitized_config.get("messages")
        if isinstance(messages, list):
            sanitized_config["messages"] = [
                _sanitize_message_tool_arguments(dict(message) if isinstance(message, dict) else message)
                for message in messages
            ]
        started = time.monotonic()
        print('LLM REQUEST: prompt_chars=' + str(len(json.dumps(sanitized_config.get('messages', []), ensure_ascii=False))))
        try:
            result = original_create(self, **sanitized_config)
            print(f'LLM RESPONSE: seconds={time.monotonic()-started:.2f} usage={getattr(result, "usage", None)}')
            return result
        except Exception:
            print(f'LLM ERROR: seconds={time.monotonic()-started:.2f}')
            raise

    create_with_json_sanitizer._logiagent_json_sanitizer = True
    OpenAIWrapper.create = create_with_json_sanitizer


_install_autogen_json_argument_sanitizer()


def record_response_with_status_oracle(item: dict, reason: str) -> bool:
    status_code = item.get("response_code")
    align_with_expected = isinstance(status_code, int) and status_code < 500
    request_info = item.get("request_data", "NoRequest")
    response_info = f"status_code={status_code}, response_data={item.get('response_data', '')}"

    if align_with_expected:
        oracle = "The API request should complete without a server-side 5XX failure."
        judge_reason = (
            f"Status-code validation was used for REST League execution: {reason}. "
            f"The response status code was {status_code}, so it is treated as non-fault."
        )
    else:
        oracle = "The API should not return a server-side 5XX failure."
        judge_reason = (
            f"Status-code validation was used for REST League execution: {reason}. "
            f"The response status code was {status_code}, so it is treated as a potential server fault."
        )

    record_result(
        oracle=oracle,
        judge_reason=judge_reason,
        align_with_expected=align_with_expected,
        request_info=request_info,
        response=response_info,
    )
    print(f"FALLBACK VALIDATION: {status_code=} {align_with_expected=}")
    return True


def prepare_rest_league_fallback_queue():
    # Preserve the accepted remaining queue; never reconstruct consumed steps.
    if REST_LEAGUE_SEED_ALL_ENDPOINTS:
        value = os.getenv("REST_LEAGUE_ENDPOINT_SEED_LIMIT")
        limit = None
        if value is not None and value.strip() != "":
            try:
                limit = max(0, int(value))
            except ValueError:
                print(f"Invalid REST_LEAGUE_ENDPOINT_SEED_LIMIT ignored: {value}")
        test_scenario.add_uncovered_api_endpoints_as_test_cases(limit=limit)


def run_rest_league_direct_fallback(max_requests: int, deadline_timestamp=None, skip_gate: bool = False, sustain: bool = False, probes_only: bool = False):
    if not REST_LEAGUE_MODE or not REST_LEAGUE_DIRECT_FALLBACK:
        return
    if not test_scenario.scenario.accepted and not skip_gate:
        # Never execute an LLM plan rejected by the scenario gate. However,
        # do not let a failed plan suppress deterministic OpenAPI coverage:
        # discard its cases and rebuild only the uncovered-operation queue.
        print('SCENARIO GATE: rejected plan discarded; trying deterministic uncovered-operation fallback.')
        test_scenario.scenario.todo_tests.clear()
        if not REST_LEAGUE_SEED_ALL_ENDPOINTS:
            print('SCENARIO GATE: deterministic fallback is disabled; no requests executed.')
            return

    # REST League auth pre-handshake: if the OpenAPI declares a login endpoint,
    # obtain a token before wasting request budget on 401s.  The handshake itself
    # costs at most 2 requests and is skipped when already attempted.
    try:
        tools.perform_auth_handshake()
    except Exception as exc:
        print(f"AUTH HANDSHAKE: initial handshake failed: {exc}")

    prepare_rest_league_fallback_queue()
    planned_seed_endpoints = [
        item.api_endpoint for item in test_scenario.scenario.todo_tests
        if getattr(item, "api_endpoint", None)
    ]
    variant = 0
    covered_seed_endpoints = {
        test_scenario.normalize_api_endpoint(item.get("api_endpoint"))
        for item in tools.all_request_sequence
        if item.get("api_endpoint") and item.get("is_2xx")
    }

    def _can_continue():
        if len(tools.all_request_sequence) >= max_requests:
            print(f"DIRECT FALLBACK: request limit reached: {max_requests}")
            return False
        if deadline_timestamp is not None and time.time() >= deadline_timestamp:
            print("DIRECT FALLBACK: time budget reached")
            return False
        return True

    def _execute_endpoint(api_endpoint: str, variant_id: int, skip_if_attempted: bool = True):
        normalized_endpoint = test_scenario.normalize_api_endpoint(api_endpoint)
        if skip_if_attempted and normalized_endpoint in covered_seed_endpoints:
            print(f"DIRECT FALLBACK: skip endpoint already covered by 2XX {api_endpoint}")
            return False

        test_scenario.current_test_case = test_scenario.TestCase(
            f"RESTgym deterministic execution: {api_endpoint}",
            api_endpoint,
            "Deterministic black-box request generated from the OpenAPI specification.",
            "The request should complete without an unexpected server-side 5XX response.",
        )
        print(f"DIRECT FALLBACK: executing {api_endpoint} variant={variant_id}")
        try:
            tools.do_openapi_request(api_endpoint, variant=variant_id)
        except Exception as exc:
            print(f"DIRECT FALLBACK: failed to execute {api_endpoint}: {exc}")

        while len(test_scenario.scenario.todo_resps) > 0:
            item = test_scenario.scenario.todo_resps.pop(0)
            record_response_with_status_oracle(item, "RESTgym deterministic OpenAPI fallback")
            if item.get("is_2xx"):
                covered_seed_endpoints.add(normalized_endpoint)
        return True

    def _execute_adaptive_candidate(api_endpoint: str, candidate: dict) -> bool:
        """Execute one single-parameter equivalence-class mutation of a 2xx seed."""
        test_scenario.current_test_case = test_scenario.TestCase(
            f"RESTgym adaptive mutation: {api_endpoint}", api_endpoint,
            "Single-parameter mutation derived from a successful request seed.",
            "The mutation should exercise a new valid or invalid parameter class.",
        )
        try:
            adaptive_testing.runtime.phase = "mutation"
            tools.do_request(**candidate)
        except Exception as exc:
            print(f"DIRECT FALLBACK: adaptive mutation failed {api_endpoint}: {exc}")
            adaptive_testing.runtime.phase = "scenario"
            return False
        while test_scenario.scenario.todo_resps:
            item = test_scenario.scenario.todo_resps.pop(0)
            record_response_with_status_oracle(item, "RESTgym adaptive parameter mutation")
        adaptive_testing.runtime.phase = "scenario"
        return True

    def _auth_wall_detected() -> bool:
        """True when the last N requests were all 401 and no token was obtained.

        Used to skip pointless fault probes / mutations that would only hit the
        authentication filter (e.g. flight-search with a fresh server instance).
        """
        if tools.auth_state.get("token"):
            return False
        if not tools.auth_state.get("attempted"):
            return False
        recent = tools.all_request_sequence[-12:]
        if len(recent) < 6:
            return False
        unauthorized = sum(1 for item in recent if item.get("response_code") == 401)
        return unauthorized >= len(recent) - 1

    def _execute_raw_fault_probe(method: str, api_path: str, label: str) -> bool:
        if not _can_continue():
            return False
        if _auth_wall_detected():
            print("DIRECT FALLBACK: skipping fault probe; authentication wall detected (all 401)")
            return False
        api_endpoint = f"{method.upper()} {api_path}"
        test_scenario.current_test_case = test_scenario.TestCase(
            f"RESTgym fault-diversity probe: {api_endpoint}",
            api_endpoint,
            f"Fault-diversity probe generated near documented API paths: {label}.",
            "The API should not return an unexpected server-side 5XX response.",
        )
        print(f"DIRECT FALLBACK: fault probe {api_endpoint} label={label}")
        try:
            tools.do_request(
                base_url=global_vars_funcs_configs.CONFIG_BASE_URL,
                method=method.upper(),
                api=api_path,
                headers={},
                params={},
                payload={},
                payload_type="application/json",
                parameter_categories={
                    "__fault_probe__": {
                        "location": "path",
                        "category_id": "fault_diversity_path_probe",
                        "category_name": "fault-diversity path probe",
                        "valid": False,
                        "reason": "Near-API path perturbation used to diversify server-side 5XX fault signatures.",
                    }
                },
            )
        except Exception as exc:
            print(f"DIRECT FALLBACK: fault probe failed {api_endpoint}: {exc}")
            return False

        while len(test_scenario.scenario.todo_resps) > 0:
            item = test_scenario.scenario.todo_resps.pop(0)
            record_response_with_status_oracle(item, "RESTgym fault-diversity probe")
        return True

    def _covered_2xx_endpoints():
        return {
            test_scenario.normalize_api_endpoint(item.get("api_endpoint"))
            for item in tools.all_request_sequence
            if item.get("api_endpoint") and item.get("is_2xx")
        }

    def _endpoint_status_stats():
        stats = {}
        for item in tools.all_request_sequence:
            endpoint = item.get("api_endpoint")
            if not endpoint:
                continue
            normalized = test_scenario.normalize_api_endpoint(endpoint)
            entry = stats.setdefault(
                normalized,
                {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0, "faults": set(), "fault_counts": {}},
            )
            status = item.get("response_code")
            if isinstance(status, int) and 200 <= status < 300:
                entry["2xx"] += 1
            elif isinstance(status, int) and 400 <= status < 500:
                entry["4xx"] += 1
            elif isinstance(status, int) and 500 <= status < 600:
                entry["5xx"] += 1
                signature = item.get("fault_signature") or str(item.get("response_data", ""))[:300]
                entry["faults"].add(signature)
                entry["fault_counts"][signature] = entry["fault_counts"].get(signature, 0) + 1
            else:
                entry["other"] += 1
        return stats

    def _variant_for_mutation_round(mutation_round: int) -> int:
        if REST_LEAGUE_NEGATIVE_MUTATION_INTERVAL <= 0:
            return mutation_round
        if (
            mutation_round % REST_LEAGUE_NEGATIVE_MUTATION_INTERVAL == 0
        ):
            # Variants 2 and 3 deliberately exercise invalid/missing classes.
            return 2 if (mutation_round // REST_LEAGUE_NEGATIVE_MUTATION_INTERVAL) % 2 else 3
        # Variants congruent to 0 or 1 modulo 5 keep HiREST categories valid,
        # while still changing values across rounds.
        valid_block = mutation_round - (mutation_round // max(REST_LEAGUE_NEGATIVE_MUTATION_INTERVAL, 1))
        return 5 * max(valid_block, 1) + (valid_block % 2)

    def _valid_variant_for_round(round_id: int) -> int:
        return 5 * max(round_id + 1, 1) + (round_id % 2)

    def _endpoint_execution_priority(api_endpoint: str):
        normalized = test_scenario.normalize_api_endpoint(api_endpoint)
        if " " not in normalized:
            return (9, normalized)
        method, path = normalized.split(" ", 1)
        path_param_count = path.count("{")
        collection_path = path_param_count == 0
        if method == "POST" and collection_path:
            rank = 0
        elif method == "POST":
            rank = 1
        elif method == "GET" and collection_path:
            rank = 2
        elif method == "GET":
            rank = 3
        elif method in {"PUT", "PATCH"}:
            rank = 4
        elif method == "DELETE":
            rank = 5
        else:
            rank = 6
        return (rank, path_param_count, normalized)

    def _path_probe_candidates(api_endpoint: str, round_id: int) -> list[tuple[str, str, str]]:
        normalized = test_scenario.normalize_api_endpoint(api_endpoint)
        if " " not in normalized:
            return []
        method, path = normalized.split(" ", 1)
        path = path.split("?", 1)[0]
        parts = [part for part in path.strip("/").split("/") if part]
        if not parts:
            return []

        candidates = []
        marker = f"logi{round_id}"
        concrete_parts = []
        for index, part in enumerate(parts):
            if part.startswith("{") and part.endswith("}"):
                name = part.strip("{}")
                pooled_value = tools._resource_value_for_parameter(name, path, round_id)
                if pooled_value is not None:
                    concrete_parts.append(str(pooled_value))
                else:
                    concrete_parts.append(marker if not name.lower().endswith("id") else str(900000 + round_id + index))
            else:
                concrete_parts.append(part)
        concrete_path = "/" + "/".join(concrete_parts)

        last = concrete_parts[-1]
        candidates.append(("GET", f"{concrete_path}/{marker}", "unknown-child"))
        candidates.append(("GET", "/" + "/".join(concrete_parts[:-1] + [f"{last}-{marker}"]), "unknown-leaf"))
        candidates.append(("GET", "/" + "/".join(concrete_parts + [last]), "duplicated-leaf"))
        if len(concrete_parts) >= 2:
            candidates.append(("GET", "/" + "/".join(concrete_parts[:-1] + ["api", marker]), "api-sibling"))
        for index, part in enumerate(parts):
            if part.startswith("{") and part.endswith("}"):
                bad_parts = list(concrete_parts)
                bad_parts[index] = f"{part.strip('{}')}-{marker}"
                candidates.append(("GET", "/" + "/".join(bad_parts), "path-param-type-mismatch"))
        if method in {"POST", "PUT", "PATCH"}:
            candidates.append((method, concrete_path, "empty-json-body"))
        return candidates

    def _is_destructive_endpoint(api_endpoint: str) -> bool:
        normalized = test_scenario.normalize_api_endpoint(api_endpoint)
        if " " not in normalized:
            return False
        method, path = normalized.split(" ", 1)
        if method == "DELETE":
            return True
        if method in {"PUT", "PATCH"} and path.count("{") >= 1:
            # Keep update-by-id in seeding/coverage, but avoid spending the
            # whole mutation budget repeatedly corrupting existing resources.
            return True
        return False

    def _is_static_get_endpoint(api_endpoint: str) -> bool:
        normalized = test_scenario.normalize_api_endpoint(api_endpoint)
        if " " not in normalized:
            return False
        method, path = normalized.split(" ", 1)
        return method == "GET" and "{" not in path

    def _should_throttle_repeated_5xx(api_endpoint: str, endpoint_stats: dict) -> bool:
        five_xx = endpoint_stats.get("5xx", 0)
        if five_xx <= 0:
            return False
        unique_faults = len(endpoint_stats.get("faults", set()))
        max_repeated_fault = max(endpoint_stats.get("fault_counts", {}).values() or [0])

        if (
            REST_LEAGUE_STATIC_GET_5XX_LIMIT > 0
            and _is_static_get_endpoint(api_endpoint)
            and five_xx >= REST_LEAGUE_STATIC_GET_5XX_LIMIT
            and unique_faults <= 1
        ):
            return True

        if (
            REST_LEAGUE_REPEAT_5XX_LIMIT > 0
            and max_repeated_fault >= REST_LEAGUE_REPEAT_5XX_LIMIT
            and unique_faults <= 2
        ):
            return True

        return False

    def _should_skip_valid_traffic(api_endpoint: str, endpoint_stats: dict) -> bool:
        if not REST_LEAGUE_MUTATE_DESTRUCTIVE_ENDPOINTS and _is_destructive_endpoint(api_endpoint):
            return True
        if _should_throttle_repeated_5xx(api_endpoint, endpoint_stats):
            return True
        if (
            REST_LEAGUE_STATIC_GET_REPLAY_LIMIT > 0
            and _is_static_get_endpoint(api_endpoint)
            and endpoint_stats.get("2xx", 0) >= REST_LEAGUE_STATIC_GET_REPLAY_LIMIT
        ):
            return True
        if endpoint_stats.get("2xx", 0) == 0 and endpoint_stats.get("4xx", 0) >= REST_LEAGUE_COVERAGE_RETRY_ROUNDS + 2:
            return True
        return False

    def _valid_traffic_priority(api_endpoint: str, status_stats: dict):
        normalized = test_scenario.normalize_api_endpoint(api_endpoint)
        endpoint_stats = status_stats.get(normalized, {})
        two_xx = endpoint_stats.get("2xx", 0)
        four_xx = endpoint_stats.get("4xx", 0)
        five_xx = endpoint_stats.get("5xx", 0)
        base_priority = _endpoint_execution_priority(api_endpoint)
        # Favor endpoints that already demonstrate reachability, but still give
        # a chance to low-count reachable endpoints instead of replaying list GETs forever.
        return (
            0 if two_xx > 0 else 1,
            two_xx,
            five_xx + four_xx,
            base_priority,
        )

    def _endpoint_score_for_replay(api_endpoint: str, status_stats: dict):
        normalized = test_scenario.normalize_api_endpoint(api_endpoint)
        endpoint_stats = status_stats.get(normalized, {})
        two_xx = endpoint_stats.get("2xx", 0)
        four_xx = endpoint_stats.get("4xx", 0)
        five_xx = endpoint_stats.get("5xx", 0)
        unique_faults = len(endpoint_stats.get("faults", set()))
        if two_xx <= 0:
            return -1000 - four_xx
        # Favor endpoints that are reachable and have shown diverse faults.
        return (unique_faults * 20) + (five_xx * 2) + two_xx - (four_xx * 3)

    def _retry_uncovered_endpoints(rounds: int, label: str, variant_offset: int = 0):
        for coverage_round in range(rounds):
            covered = _covered_2xx_endpoints()
            uncovered = [
                endpoint for endpoint in ordered_endpoints
                if test_scenario.normalize_api_endpoint(endpoint) not in covered
            ]
            if not uncovered:
                return
            print(
                f"DIRECT FALLBACK: {label} coverage retry round {coverage_round + 1}/"
                f"{rounds}, uncovered={len(uncovered)}"
            )
            for api_endpoint in uncovered:
                if not _can_continue():
                    return
                if REST_LEAGUE_VALID_ONLY_MUTATION:
                    variant_id = _valid_variant_for_round(variant_offset + coverage_round)
                else:
                    variant_id = coverage_round % 2
                _execute_endpoint(api_endpoint, variant_id=variant_id, skip_if_attempted=False)

    test_scenario.scenario.todo_tests.sort(key=lambda item: _endpoint_execution_priority(item.api_endpoint))
    while len(test_scenario.scenario.todo_tests) > 0:
        if not _can_continue():
            break

        test_case = test_scenario.scenario.todo_tests.pop(0)
        seed_variant = 0 if REST_LEAGUE_VALID_SEED else variant
        if _execute_endpoint(test_case.api_endpoint, variant_id=seed_variant, skip_if_attempted=True):
            variant += 1

    # ---- Phase order (REST League) ----
    # 1. seed / planned scenarios + early coverage retry
    # 2. success replay (controlled replay of 2xx seeds -> branch diversity)
    # 3. adaptive mutation (single-parameter equivalence mutations of 2xx seeds)
    # 4. late coverage retry for still-uncovered endpoints
    # 5. fault-diversity path probes (only when no auth wall)
    # 6. optional valid traffic / balanced mode
    ordered_endpoints = sorted(global_vars_funcs_configs.catalog.api_swagger_map.keys(), key=_endpoint_execution_priority)

    # Fault-diversity path probes: pure noise on authentication walls, so they
    # are gated off when a wall blocks the API.  One-shot per process; used by
    # both the normal fallback path and (via probes_only) by structured /
    # sequence mode, which otherwise never reaches this fallback.
    def _run_fault_diversity_probes(endpoints):
        if getattr(_run_fault_diversity_probes, "_done", False):
            return
        _run_fault_diversity_probes._done = True
        if REST_LEAGUE_FAULT_DIVERSITY_REQUESTS <= 0 or _auth_wall_detected():
            return
        probes_sent = 0
        misses_since_5xx = 0
        probe_seen = set()
        print(
            "DIRECT FALLBACK: fault diversity exploration start, "
            f"budget={REST_LEAGUE_FAULT_DIVERSITY_REQUESTS}"
        )
        for round_id in range(REST_LEAGUE_FAULT_DIVERSITY_REQUESTS):
            for api_endpoint in endpoints:
                for method, api_path, label in _path_probe_candidates(api_endpoint, round_id):
                    if probes_sent >= REST_LEAGUE_FAULT_DIVERSITY_REQUESTS:
                        break
                    key = (method, api_path)
                    if key in probe_seen:
                        continue
                    probe_seen.add(key)
                    before_5xx = sum(1 for item in tools.all_request_sequence if item.get("is_5xx"))
                    if not _execute_raw_fault_probe(method, api_path, label):
                        continue
                    probes_sent += 1
                    after_5xx = sum(1 for item in tools.all_request_sequence if item.get("is_5xx"))
                    if after_5xx > before_5xx:
                        misses_since_5xx = 0
                    else:
                        misses_since_5xx += 1
                    if (
                        REST_LEAGUE_FAULT_DIVERSITY_STOP_AFTER_NO_5XX > 0
                        and misses_since_5xx >= REST_LEAGUE_FAULT_DIVERSITY_STOP_AFTER_NO_5XX
                    ):
                        print(
                            "DIRECT FALLBACK: fault diversity exploration stopped after "
                            f"{misses_since_5xx} probes without 5XX"
                        )
                        probes_sent = REST_LEAGUE_FAULT_DIVERSITY_REQUESTS
                        break
                if probes_sent >= REST_LEAGUE_FAULT_DIVERSITY_REQUESTS:
                    break
            if probes_sent >= REST_LEAGUE_FAULT_DIVERSITY_REQUESTS:
                break

    if probes_only:
        # Structured/sequence mode: send the one-shot fault-diversity probes
        # without re-running coverage or mutation phases.
        _run_fault_diversity_probes(ordered_endpoints)
        return
    if REST_LEAGUE_SEED_ALL_ENDPOINTS:
        _retry_uncovered_endpoints(REST_LEAGUE_COVERAGE_RETRY_ROUNDS, "early")
    elif REST_LEAGUE_FAILED_RETRY_ROUNDS > 0:
        # Retry only endpoints that the generated scenario explicitly asked for
        # and that still have no 2xx response.  This is deliberately narrower
        # than synthesizing every uncovered operation.
        for retry_round in range(REST_LEAGUE_FAILED_RETRY_ROUNDS):
            covered = _covered_2xx_endpoints()
            failed = [
                endpoint for endpoint in planned_seed_endpoints
                if test_scenario.normalize_api_endpoint(endpoint) not in covered
            ]
            if not failed:
                break
            print(
                f"DIRECT FALLBACK: scenario failure retry {retry_round + 1}/"
                f"{REST_LEAGUE_FAILED_RETRY_ROUNDS}, endpoints={len(failed)}"
            )
            for endpoint in failed:
                if not _can_continue():
                    return
                _execute_endpoint(endpoint, variant_id=0, skip_if_attempted=False)

    # Success replay: repeat 2xx-reachable endpoints with the next equivalence
    # variant.  This is the pet-clinic pattern: reuse working requests to
    # exercise more branches instead of spraying invalid payloads.
    for replay_round in range(REST_LEAGUE_SUCCESS_REPLAY_ROUNDS):
        status_stats = _endpoint_status_stats()
        replay_endpoints = [
            endpoint for endpoint in ordered_endpoints
            if status_stats.get(test_scenario.normalize_api_endpoint(endpoint), {}).get("2xx", 0) > 0
        ]
        replay_endpoints.sort(
            key=lambda endpoint: (
                -_endpoint_score_for_replay(endpoint, status_stats),
                _endpoint_execution_priority(endpoint),
            )
        )
        print(
            f"DIRECT FALLBACK: success replay round {replay_round + 1}/"
            f"{REST_LEAGUE_SUCCESS_REPLAY_ROUNDS}, seeds={len(replay_endpoints)}"
        )
        for api_endpoint in replay_endpoints:
            if not _can_continue():
                return
            normalized_endpoint = test_scenario.normalize_api_endpoint(api_endpoint)
            endpoint_stats = status_stats.get(normalized_endpoint, {})
            if _should_throttle_repeated_5xx(api_endpoint, endpoint_stats):
                print(
                    "DIRECT FALLBACK: throttle repeated 5XX endpoint during replay: "
                    f"{api_endpoint} stats={endpoint_stats}"
                )
                continue
            if (
                REST_LEAGUE_SKIP_PERSISTENT_4XX
                and endpoint_stats.get("2xx", 0) == 0
                and endpoint_stats.get("4xx", 0) >= REST_LEAGUE_COVERAGE_RETRY_ROUNDS + 2
            ):
                print(f"DIRECT FALLBACK: skip persistent 4XX endpoint during replay: {api_endpoint}")
                continue
            variant_id = _variant_for_mutation_round(replay_round + 1)
            _execute_endpoint(api_endpoint, variant_id=variant_id, skip_if_attempted=False)

    if REST_LEAGUE_BALANCED_MODE and REST_LEAGUE_VALID_TRAFFIC_ROUNDS > 0:
        print("DIRECT FALLBACK: balanced mode finished after valid traffic phase.")
        return

    def _adaptive_mutation_phase(endpoints):
        nonlocal variant
        for mutation_round in range(1, REST_LEAGUE_MUTATION_ROUNDS):
            status_stats = _endpoint_status_stats()
            if REST_LEAGUE_SCORE_ORDERING:
                mutation_endpoints = sorted(
                    endpoints,
                    key=lambda endpoint: (
                        -_endpoint_score_for_replay(endpoint, status_stats),
                        _endpoint_execution_priority(endpoint),
                    ),
                )
            else:
                mutation_endpoints = endpoints
            mutation_endpoints = [
                endpoint for endpoint in mutation_endpoints
                if status_stats.get(test_scenario.normalize_api_endpoint(endpoint), {}).get("2xx", 0) > 0
            ]
            for api_endpoint in mutation_endpoints:
                if not _can_continue():
                    return
                if not REST_LEAGUE_MUTATE_DESTRUCTIVE_ENDPOINTS and _is_destructive_endpoint(api_endpoint):
                    continue
                normalized_endpoint = test_scenario.normalize_api_endpoint(api_endpoint)
                endpoint_stats = status_stats.get(normalized_endpoint, {})
                if _should_throttle_repeated_5xx(api_endpoint, endpoint_stats):
                    print(
                        "DIRECT FALLBACK: throttle repeated 5XX endpoint during mutation: "
                        f"{api_endpoint} stats={endpoint_stats}"
                    )
                    continue
                if (
                    REST_LEAGUE_SKIP_PERSISTENT_4XX
                    and endpoint_stats.get("2xx", 0) == 0
                    and endpoint_stats.get("4xx", 0) >= REST_LEAGUE_COVERAGE_RETRY_ROUNDS + 2
                ):
                    print(f"DIRECT FALLBACK: skip persistent 4XX endpoint during mutation: {api_endpoint}")
                    continue
                if REST_LEAGUE_VALID_ONLY_MUTATION:
                    variant_id = _valid_variant_for_round(mutation_round)
                else:
                    variant_id = _variant_for_mutation_round(mutation_round)
                adaptive_candidate = adaptive_testing.runtime.next_mutation(
                    api_endpoint,
                    global_vars_funcs_configs.catalog.api_swagger_map.get(normalized_endpoint, {}),
                )
                if adaptive_candidate is not None:
                    executed = _execute_adaptive_candidate(api_endpoint, adaptive_candidate)
                else:
                    # None is a stop decision, not permission to use legacy mutations.
                    executed = False
                if executed:
                    variant += 1

    # Adaptive mutation of successful seeds.  Skipped in sustain mode: after
    # full coverage we go straight to valid-traffic replay to fill the budget.
    if not sustain:
        _adaptive_mutation_phase(ordered_endpoints)

    # Late coverage retry: give every uncovered endpoint one more shot with a
    # fresh variant after mutations had time to populate the resource pool.
    if REST_LEAGUE_SEED_ALL_ENDPOINTS:
        _retry_uncovered_endpoints(
            REST_LEAGUE_LATE_COVERAGE_RETRY_ROUNDS,
            "late",
            variant_offset=REST_LEAGUE_COVERAGE_RETRY_ROUNDS + 1,
        )

    # Fault-diversity path probes last (one-shot per process; defined above so
    # that structured/sequence mode can also trigger them via probes_only).
    _run_fault_diversity_probes(ordered_endpoints)

    if REST_LEAGUE_VALID_TRAFFIC_ROUNDS > 0:
        print(
            "DIRECT FALLBACK: valid traffic phase start, "
            f"rounds={REST_LEAGUE_VALID_TRAFFIC_ROUNDS} sustain={sustain}"
        )
        for valid_round in range(REST_LEAGUE_VALID_TRAFFIC_ROUNDS):
            if sustain and not _can_continue():
                print("DIRECT FALLBACK: sustain mode ended (request/time budget reached)")
                return
            status_stats = _endpoint_status_stats()
            valid_endpoints = sorted(
                ordered_endpoints,
                key=lambda endpoint: _valid_traffic_priority(endpoint, status_stats),
            )
            executed_in_round = 0
            for api_endpoint in valid_endpoints:
                if not _can_continue():
                    return
                normalized_endpoint = test_scenario.normalize_api_endpoint(api_endpoint)
                endpoint_stats = status_stats.get(normalized_endpoint, {})
                if not sustain and _should_skip_valid_traffic(api_endpoint, endpoint_stats):
                    continue
                variant_id = _valid_variant_for_round(valid_round)
                if _execute_endpoint(api_endpoint, variant_id=variant_id, skip_if_attempted=False):
                    executed_in_round += 1
            print(
                f"DIRECT FALLBACK: valid traffic round {valid_round + 1}/"
                f"{REST_LEAGUE_VALID_TRAFFIC_ROUNDS}, executed={executed_in_round}"
            )
            if executed_in_round == 0 and not sustain:
                break


def main():
    api_retry_in_round_max_time = 3
    api_retry_last_pop_test_case = None  # type: test_scenario.TestCase
    previous_agent_names = []
    last_progress_request_count = 0
    last_recorded_result_count = 0
    scenario_generation_retries = 0
    scenario_detail_round = False

    def _endpoint_seed_limit():
        value = os.getenv("REST_LEAGUE_ENDPOINT_SEED_LIMIT")
        if value is None or value.strip() == "":
            return None
        try:
            return max(0, int(value))
        except ValueError:
            print(f"Invalid REST_LEAGUE_ENDPOINT_SEED_LIMIT ignored: {value}")
            return None

    # Prompts

    full_coverage = set(global_vars_funcs_configs.catalog.api_swagger_map) <= global_vars_funcs_configs.catalog.covered_2xx
    selection_rule = (
        'Every documented operation already has a 2XX. Select a novel, coherent boundary '
        'or unusual-but-valid scenario. An operation sequence may repeat only if the '
        'tested parameter values or resource states are substantively different. '
        'Prefer optional fields, enum alternatives, resource state transitions, and distinct '
        'values that might reach new branches or reveal a new 5XX.'
        if full_coverage else
        'Select at least one [UNCOVERED] operation and its necessary setup operations. '
        'The final scenario must include at least one previously uncovered operation.'
    )
    test_scenario_prompt = f"""
You are a test scenario generate agent.
You generate one detailed and appropriate test scenario in two phases.

PHASE 1: inspect the compact operation directory below. {selection_rule}
Identify the detailed operation documents needed for the scenario and its setup operations.
Reply with a line beginning exactly `DETAIL_REQUEST:` followed by a
comma-separated list of documented METHOD /path signatures. Do not generate the scenario
or call any tools in phase 1.
Request at most 8 detailed documents: selected operations and their necessary setup operations.
Generate at most 12 steps. Do not request or copy the entire API directory as a scenario.

PHASE 2: after the detailed documents are supplied, generate the final scenario and only
then allow the recorder to record it. The final scenario must use concrete path parameters in the documented
path, with values obtained from earlier responses when dependencies require them.

# Requirements for Test Scenario
- Each generated test scenario should be appropriately detailed, completed, complex, realistic, and coherent. Should mix different operations across multiple API endpoints, 
- Focus on interactions that could reveal defects such as mismatches (e.g., create, update, retrieve, delete) in data consistency, unexpected error responses, or unexpected dependency between endpoints.
- Design test scenarios with logically connected steps to reveal data inconsistencies, state management flaws, or unexpected behaviors, focusing on extended interaction paths with multiple CRUD operations or endpoint dependencies.
- For REST League execution, generate a flat sequence of independently executable HTTP requests whenever the API surface allows it.
- Each numbered step must contain exactly one concrete HTTP request to exactly one OpenAPI endpoint.
- Do not generate steps that require parallel/concurrent requests, loops, manual assertions, hidden state inspection, source-code access, or a human to transform a previous response.
- Do not describe compound steps such as "call A, then call B" inside one numbered step. Split them into separate numbered steps.
- Prefer requests likely to obtain 2XX responses for operation coverage, while mixing boundary values that may expose unique 5XX server faults.
- Reuse values from earlier responses only when the next request can still be written as one concrete HTTP request.

# Output Format

The output should be formatted as a step-by-step sequence, where each point explains a part of the scenario, including:
- The API endpoints that will be used.
- Description of this API call
- Expected responses.
Each step should be numbered and describe in detail to fully understand the scenario while remaining concise and refined.

# Example
```
**Input:** Provided system APIs: [1. POST /products, 2. GET /products, 3. POST /cart, 4. POST /checkout] 
**Output:**
1. **Title:** Add a Product to the System
    - **API Endpoint:** POST /products
    - **Description:** Adds a new product to the system with details such as name, price, and stock quantity.
   - **Expected Response:** A success response confirming that the product has been added successfully with the provided details.

2. **Title:** Retrieve Product List
    - **API Endpoint:** GET /products
    - **Description:** Fetches the list of all available products to verify that the newly added product appears in the inventory.
    - **Expected Response:** A list of products, including the recently added product with accurate details.
3. ... 

Summary: 
This workflow tests the entire purchasing process, ensuring that product management, cart functionality, and checkout operations work seamlessly together and that inventory updates correctly.
```

The following REST APIs are in the under test system, Please do not use other APIs other than these.
{global_vars_funcs_configs.catalog.scenario_context()}


Previously generated test scenarios:
{test_scenario.get_previous_all_test_scenarios_for_llm()}
Ensure that the new test scenario tests genuinely different values or resource states from previous scenarios.

Start generate new test scenario for this REST API system.
"""

    test_scenario_recorder_prompt = """
First, call `record_test_scenario` once to record a test scenario generated by this REST API system, including a `summary` and `api_sequence`.  
Then, sequentially call `add_test_case` to record all REST test cases derived from the test scenario.

If `record_test_scenario` is called and all test data has been recorded, respond with "STATE-FLOW-MESSAGE:TEST SCENARIO RECORD COMPLETED" and do nothing else.

# Function Details

### **`record_test_scenario`**
- **`summary`**: A description of the test scenario. If the test scenario document already includes a summary, use it as is without modification.
- **`api_sequence`**: A list of items representing the sequence of API calls.  
  Each item must follow the **API Endpoint Rules** (see below).  
  Example: `["POST /pet", "GET /user/{{username}}", "DELETE /order/{{id}}"]`  

### **`add_test_case`**
- **`test_case_title`**: A title describing the specific test case.  
- **`api_endpoint`**: The API endpoint (see **API Endpoint Rules** below).  
- **`description`**: A description of the test case purpose and behavior.  
- **`expected_resp`**: The expected response for this test case.

### **API Endpoint Rules**
- **Format**: Use `<HTTP_METHOD> <API_ENDPOINT>`.  
  Examples:  
  - `POST /pet`  
  - `GET /user/{{username}}`  
  - `DELETE /order/{{id}}`  

- **Placeholders**: Use placeholders (e.g., `{{username}}`, `{{id}}`) for variables in the endpoints instead of hardcoded values.  
  Examples:  
  - Correct: `GET /user/{{username}}`  
  - Incorrect: `GET /user/johnDoe`  

- **Naming**: Always follow the placeholder names as defined in the Swagger documentation.  
  - Do **not** modify placeholder names. For example:  
    - Correct: `DELETE /order/{{id}}` (matches Swagger's definition).  
    - Incorrect: `DELETE /order/{{order_id}}` (modified the placeholder name).  

### Shared Rules
- Follow the API endpoint rules consistently across both `record_test_scenario` and `add_test_case`.
- Each `add_test_case` must represent exactly one concrete HTTP request to exactly one OpenAPI endpoint.
- If a scenario step contains multiple sequential API calls, split it into multiple `add_test_case` calls.
- Do not put query strings or concrete path values in `api_endpoint`; use only the OpenAPI endpoint template.
- For REST League execution, record every concrete HTTP request described by the scenario as an `add_test_case` call.
- Do not skip boundary, negative, or repeated requests from the generated scenario.
- Do not record concurrency or "five parallel requests" as one test case. Convert each intended HTTP request into a separate `add_test_case` call.
- Only emit "STATE-FLOW-MESSAGE:TEST SCENARIO RECORD COMPLETED" after every concrete HTTP request from the scenario has been recorded.
"""

    api_invoke_prompt = f"""
You are API invocation agent.
You generate API request parameters and execute API call.

# Steps
1. **Get next REST API test case**: Using the `get_next_test_case` tool to get next one test api, this api's corresponding swagger api info, and other useful info. 
2. **Generate Parameters AND Send API Requests**: For previous got API, Generate the appropriate parameters according to the information retrieved. Try to generate Then Execute this API request using the `do_request` tool. 

# Notes
- Get one by `get_next_test_case` then call `do_request` for it. Do not call get_next_test_case repeatedly.
- If an API request fails and get_next_test_case returns the same API as the previous one, use do_request to execute the request instead of calling get_next_test_case again.
- Execute exactly one HTTP request after each `get_next_test_case` call.
- Use only the endpoint path in the `api` argument; put query parameters in `params`.
- Prefer generating requests that can obtain 2XX responses, while also exploring boundary values that may expose 5XX server faults.
- In REST League execution, continue until `get_next_test_case` returns "No more test case"; do not stop after one successful request.
- If the test case description contains multiple examples, choose one concrete request for the current test case only.
- Generate parameter values based on Swagger constraints. Ensure values are **valid, diverse, and reproducible**:
  - **Strings**: Use deterministic but varied values derived from the endpoint and test case, such as `"user_001"`, `"file_001.txt"`, `"user001@example.net"`, `"Passw0rd001"`, `"serial_001"`.
  - **Numbers**: Use deterministic values within allowed ranges, including boundaries.
  - **Enums**: Pick deterministic enum values and vary them across repeated runs using the configured seed.
  - **Formats**: Follow required patterns deterministically (e.g., UUIDs, emails, dates).
  - Hint: For variables that given example values in docs, try using example values at first, but avoiding reuse static or predictable values across all test cases/scenarios.
- When and only when `get_next_test_case` return "No more test case", say "STATE-FLOW-MESSAGE:NO MORE REQUESTS" and do nothing else. That means all request have been executed.

# Function Details
**`get_next_test_case` Function**: Do Not have parameters.

**`do_request` Parameters**:
- **`base_url`** (str): Base URL of the server. base_url = {global_vars_funcs_configs.catalog.get_base_url()}
- **`method`** (str): HTTP method to be used (e.g., GET, POST).
- **`api`** (str): API endpoint path.
- **`headers`** (dict): Request headers. 
  - **Note**: If the context includes Auth information such as an `access_token`, ensure the `headers` dictionary contains an `Authorization` key in the format: `"Authorization": "{{access_token_type}} {{access_token}}"`. For example: `{{"Authorization": "Bearer eyJhb..."}}`.
- **`params`** (dict): URL parameters.
- **`payload`** (dict): Request body payload, typically JSON Object: {{"key": "value"}}, do not use list until you are told so.
- **`payload_type`** (str) Payload type of request body, get_next_test_case's return info may have this info. Default is "application/json"
"""

    api_record_prompt = """
You are an API invocation Record agent.  
You should record:
1. Useful items of this API invocation by `record_useful_items` function tool if invocation is success
2. Reflection of this API invocation by `record_api_reflections` function tool if invocation is failed

If there are no items or reflections deserve to be recorded, return:  
`"STATE-FLOW-MESSAGE:NO ITEMS SHOULD RECORD"` and take no further action in such cases.

# Useful items of this API invocation (success invocation)

If the previous response is correct (aligns with the oracle), use the `record_useful_items` function tool to capture all **relevant object attributes** from the previous REST API's request and response.  
You should only record attributes directly related to identifiable objects, such as `id`, `name`, `password`, or other key properties essential for subsequent requests or validations.

### What to Record:
- **Object Attributes**: Attributes that describe identifiable objects within the system under test, typically needed for subsequent operations or validations. These include:
  - Unique identifiers (e.g., `id`, `productID`, `badge_id`)  
  - Descriptive attributes (e.g., `name`, `badge_name`)  
  - Related links or references (e.g., `link_url`, `image_url`)  
  - Relevant object states (e.g., `state: "active"`, `status: "locked"`) if they are meaningful within the system.  

### What NOT to Record:
- **Exclude the following types of information**:  
  1. **Error or Exception Details**:
     - Example: `error_message`, `reason`, `error_detail`.  
  2. **Request/Response Metadata**:
     - Example: `timestamp`, `etag`, `response_code`.  
  3. **Process or Request State**:
     - Example: `request_status`, `retry_count`.  
  4. **Boolean Values**:
     - Unless they are explicitly part of an object's attributes (e.g., `is_active`), do not record standalone booleans.  
  5. **Other Irrelevant Data**:
     - Avoid data that does not describe objects in the system under test.

### Recording Instructions
- Prepare a dictionary with each useful item's name, corresponding value, and description. 
  Each dictionary key should represent the name of an essential attribute (e.g., "productID", "username").
  The value for each key should be another dictionary containing the following fields:
  - `"value"`: The corresponding value of the attribute (e.g., "101", "Max").
  - `"description"`: A brief description of the attribute (e.g., "Product ID of a specific item", "User with admin privileges").
- Pass the dictionary of items directly to the `record_useful_items` function to record them all at once.


# Reflection of this API invocation (success invocation)
If the previous response of this API is failed, think about if previous api invocation params have problem.

### What to Reflect:
1. **Parameter Issues**: Identify any incorrect or missing parameters in the request. Examples include:
   - Invalid or missing `access_token` in headers.
   - Incorrect or missing required fields in the payload (e.g., `email`, `password`).
   - Mistyped endpoint or query parameters.

2. **Authorization/Authentication Errors**:
   - Was the `Authorization` header properly set (e.g., valid `access_token`)?
   - Were user roles/permissions sufficient for this request?

3. **API Contract Violations**:
   - Does the payload or request structure deviate from API specifications?
   - Is the API expecting additional headers or parameters not provided?

4. **Environmental Issues**:
   - Could server configuration (e.g., CORS, rate limiting) or connectivity problems have caused the failure?

### Reflection Recording Instructions:
- Create a dictionary with detailed reflection points, using concise and descriptive keys.
- Each key should highlight an area of concern (e.g., "missing_access_token", "invalid_payload").
- Pass the dictionary to the `record_api_reflections` function to document the reflection.


# Function Details of `record_useful_items` and `record_api_reflections`
**`record_useful_items` Function Parameter**:
- **`items`**: A dictionary where each key is the name of a useful item to be recorded, and each value is a dictionary containing two fields:
  - `"value"`: The corresponding value of the attribute (e.g., `"101"`, `"Max"`).
  - `"description"`: The description of the attribute (e.g., `"Product ID of a specific item"`, `"User with admin privileges"`).
  Example: 
  ```python
  {
      "productID": {"value": "101", "description": "Product ID of a specific item"},
      "username": {"value": "Max", "description": "User with admin privileges"}
  }
  ```

**`record_api_reflections` Function Parameters**:
- **`api_endpoint`** (str): The API endpoint where the issue occurred. see **API Endpoint Rule** below.
- **`issue_title`** (str): A concise title describing the issue (e.g., `"missing_access_token"` or `"invalid_query_params"`).
- **`issue_detail`** (str): A detailed explanation of the issue (e.g., `"Authorization header is missing or improperly formatted in the request headers."`).

### **API Endpoint Rules**
- **Format**: Use `<HTTP_METHOD> <API_ENDPOINT>`.  
  Examples:  
  - `POST /pet`  
  - `GET /user/{{username}}`  
  - `DELETE /order/{{id}}`  

- **Placeholders**: Use placeholders (e.g., `{{username}}`, `{{id}}`) for variables in the endpoints instead of hardcoded values.  
  Examples:  
  - Correct: `GET /user/{{username}}`  
  - Incorrect: `GET /user/johnDoe`  

- **Naming**: Always follow the placeholder names as defined in the Swagger documentation.  
  - Do **not** modify placeholder names. For example:  
    - Correct: `DELETE /order/{{id}}` (matches Swagger's definition).  
    - Incorrect: `DELETE /order/{{order_id}}` (modified the placeholder name).  
"""

    validation_prompt = """
You are an API Validation Agent. You should do your validation for response data in two steps.
Remember you must *Explicitly* reasoning and analyze by output your thoughts and then you can call `record_result` function tool.

# Steps
1. Call `get_next_response_for_validation` once to retrieve API documentation or response info.
2. Compare the expected and actual responses, *explicitly* evaluating them from multiple perspectives, and record by calling `record_result` function.
   - Consider both the response code and key elements in the response body.
   - Provide a detailed evaluation from at least three different perspectives, ensuring that one perspective may contradict the others to offer a balanced assessment.
   - *Explicitly* output your reasoning for each perspective, Output Format: 
     * Thought1: <thoughts>
     * Thought2: <thoughts>
     * Thought3: <thoughts> ...
     You have to output your thoughts following this Output Format before you call record_result function tool.
   - Suggesting the `record_result` function tool to log the evaluation result:Record the evaluation using `record_result`:
     * Include params `align_with_expected`, `judge_reason`, `oracle` , `request_info`, and `response` when calling the `record_result` function
     * For *minor mismatches**, mark `align_with_expected` as `True`, but highlight the discrepancies in `judge_reason`. For significant deviations, mark it as `False` and elaborate.

# Function Details
**`get_next_response_for_validation` Function**: Do Not have parameters.

**`record_result` Function** Parameters:
- **oracle**: The expected response, provided as a string.
- **judge_reason**: 
  - Provide clear reasoning for your judgment, highlighting any deviations and their significance.
  - For `align_with_expected = True`, explicitly state why the response is considered aligned despite the minor deviations. For example:
    - *"The response includes an additional field not specified in the expected result, but it does not affect the primary functionality."*
  - For `align_with_expected = False`, clearly explain how the discrepancies impact functionality or violate the expected behavior. For example:
    - *"The response omits the required 'id' field, which is critical for subsequent operations."*
    - *"The response code is 500, indicating a server error, which does not align with the expected behavior."*
- **align_with_expected**: 
  - Set to `True` if the actual response mainly aligns with the expected result, even if there are minor deviations such as:
    - Minor differences in non-critical fields or message strings, as long as core functionality and intent are preserved.
    - If you believe the mismatch in response is due to your incorrect request parameters or your wrong expectations, set align_with_expected to True.
  - Set to `False` if discrepancies significantly affect functionality, intended behavior, or key data elements.
- **request_info**: Information about the API request, including method, URL, parameters, and other relevant details.
- **response**: The actual response, including both the response code and body.

# Notes

- Perform one validation for one request and response.
- Provide detailed reasoning when evaluating the alignment of responses.
- If request in test scenario, but not actually do request, record it with: align_with_expected=False, request_info=NoRequest, response=NoResponse
"""

    print(f"{test_scenario_prompt}")

    test_scenario_agent = ConversableAgent(name="test_scenario_agent",
                                             system_message=test_scenario_prompt,
                                             llm_config=llm_config,
                                             human_input_mode="NEVER")

    test_scenario_recorder_agent = ConversableAgent(name="test_scenario_recorder_agent",
                                                    system_message=test_scenario_recorder_prompt + "\n\n" + STRICT_JSON_TOOL_CALL_INSTRUCTION,
                                                    llm_config=llm_config,
                                                    human_input_mode="NEVER")
    test_scenario_recorder_agent.register_for_llm(name="add_test_case", description="record rest api test case")(test_scenario.add_test_case)
    test_scenario_recorder_agent.register_for_execution(name="add_test_case")(test_scenario.add_test_case)

    test_scenario_recorder_agent.register_for_llm(name="record_test_scenario", description="record rest api test scenario")(test_scenario.record_test_scenario)
    test_scenario_recorder_agent.register_for_execution(name="record_test_scenario")(test_scenario.record_test_scenario)

    api_invoke_agent = ConversableAgent(name="api_invoke_agent",
                                        system_message=api_invoke_prompt + "\n\n" + STRICT_JSON_TOOL_CALL_INSTRUCTION,
                                        llm_config=llm_config,
                                        human_input_mode="NEVER",
                                        )
    api_invoke_agent.register_for_llm(name="get_next_test_case", description="get next api test info")(test_scenario.get_next_test_case)
    api_invoke_agent.register_for_execution(name="get_next_test_case")(test_scenario.get_next_test_case)

    api_invoke_agent.register_for_llm(name="do_request",description="Do REST API request")(do_request)
    api_invoke_agent.register_for_execution(name="do_request")(do_request)

    # record feed back
    api_recorder_agent = ConversableAgent(name="api_recorder_agent",
                                                    system_message=api_record_prompt + "\n\n" + STRICT_JSON_TOOL_CALL_INSTRUCTION,
                                                    llm_config=llm_config,
                                                    human_input_mode="NEVER")
    api_recorder_agent.register_for_llm(name="record_useful_items", description="record useful item among previous response data")(test_scenario.record_useful_items)
    api_recorder_agent.register_for_execution(name="record_useful_items")(test_scenario.record_useful_items)

    api_recorder_agent.register_for_llm(name="record_api_reflections", description="record reflection for api call failure")(test_scenario.record_api_reflections)
    api_recorder_agent.register_for_execution(name="record_api_reflections")(test_scenario.record_api_reflections)

    validation_agent = ConversableAgent(name="validation_agent",
                                        system_message=validation_prompt + "\n\n" + STRICT_JSON_TOOL_CALL_INSTRUCTION,
                                        llm_config=llm_config,
                                        human_input_mode="NEVER")

    validation_agent.register_for_llm(name="get_next_response_for_validation",
                                      description="get next api request and response for validation")(test_scenario.get_next_response_for_validation)
    validation_agent.register_for_execution(name="get_next_response_for_validation")(test_scenario.get_next_response_for_validation)

    validation_agent.register_for_llm(name="record_result",
                                      description="record result")(record_result)
    validation_agent.register_for_execution(name="record_result")(record_result)

    # ==================== Execution ========================
    def record_pending_response_with_status_oracle(reason: str) -> bool:
        if len(test_scenario.scenario.todo_resps) == 0:
            return False

        item = test_scenario.scenario.todo_resps.pop(0)
        record_response_with_status_oracle(item, reason)
        print(f"FALLBACK VALIDATION QUEUE: responses_left={len(test_scenario.scenario.todo_resps)}")
        return True

    def record_all_pending_responses_with_status_oracle(reason: str) -> int:
        recorded = 0
        while record_pending_response_with_status_oracle(reason):
            recorded += 1
        return recorded

    def state_transition(last_speaker, groupchat):
        nonlocal scenario_generation_retries, scenario_detail_round
        nonlocal last_progress_request_count
        # Rate limit for online systems
        # time.sleep(random.randint(100, 1000) / 1000)
        nonlocal api_retry_in_round_max_time, api_retry_last_pop_test_case, previous_agent_names, last_recorded_result_count
    
        messages = groupchat.messages
        last_msg = messages[-1]
        last_msg_content = last_msg.get('content') or ""
    
        if len(tools.all_request_sequence) != last_progress_request_count:
            previous_agent_names.clear()
            last_progress_request_count = len(tools.all_request_sequence)
        previous_agent_names.append(last_speaker.name)
        if len(previous_agent_names) > 10:
            previous_agent_names = previous_agent_names[-10:]
            if len(set(previous_agent_names)) <= 1:
                repeated_agent_name = previous_agent_names[-1]
                print(f"Agent called repeatedly more than 10 times, trying fallback flow, agent_names: {repeated_agent_name}")
                if repeated_agent_name == "api_invoke_agent" and len(test_scenario.scenario.todo_resps) > 0:
                    if SCORING_ONLY_MODE:
                        if record_pending_response_with_status_oracle("api_invoke_agent repeated too many times"):
                            last_recorded_result_count = len(tools.right_results) + len(tools.wrong_results)
                            return api_invoke_agent
                        return None
                    if REST_LEAGUE_MODE:
                        if record_pending_response_with_status_oracle("api_invoke_agent repeated too many times"):
                            last_recorded_result_count = len(tools.right_results) + len(tools.wrong_results)
                            return api_invoke_agent
                    return validation_agent
                if repeated_agent_name == "validation_agent":
                    current_recorded_result_count = len(tools.right_results) + len(tools.wrong_results)
                    if current_recorded_result_count > last_recorded_result_count:
                        last_recorded_result_count = current_recorded_result_count
                        if len(test_scenario.scenario.todo_resps) > 0:
                            test_scenario.scenario.todo_resps.pop(0)
                        print(
                            f"REMOVE RESPONSE: there are {len(test_scenario.scenario.todo_resps)} responses left (0 is right).")
                        return api_recorder_agent
                    if record_pending_response_with_status_oracle("validation_agent repeated too many times"):
                        last_recorded_result_count = len(tools.right_results) + len(tools.wrong_results)
                        return api_invoke_agent
                if repeated_agent_name == "api_recorder_agent":
                    return api_invoke_agent
                return None
    
        # print(f"{last_msg_content}")
    
        if last_speaker is test_scenario_agent:
            if last_msg_content.lstrip().startswith('DETAIL_REQUEST:') and not scenario_detail_round:
                requested = re.findall(r'\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE)\s+/[^,;\n]+', last_msg_content)
                requested = [item.strip().rstrip('.') for item in requested]
                targets = list(global_vars_funcs_configs.catalog.required_targets)
                requested = list(dict.fromkeys(targets + requested))
                requested = [e for e in requested if e in global_vars_funcs_configs.catalog.api_swagger_map]
                requested = requested[:max(1, env_int('SCENARIO_MAX_DETAIL_OPERATIONS', 8))]
                details = global_vars_funcs_configs.catalog.detailed_context(requested)
                if not details or details == '{}':
                    details = global_vars_funcs_configs.catalog.detailed_context(
                    list(global_vars_funcs_configs.catalog.required_targets)
                    )
                scenario_detail_round = True
                print(f'SCENARIO PLAN: requested {len(requested)} detailed operation documents.')
                groupchat.messages.append({
                    'role': 'user',
                    'content': 'PHASE 2: Here are the requested detailed OpenAPI documents:\n'
                               + details
                               + '\nNow generate the final numbered scenario. Do not output DETAIL_REQUEST again.'
                })
                return test_scenario_agent
            test_scenario.scenario.scenario_text = last_msg_content
            test_scenario.add_missing_test_cases_from_scenario_text(last_msg_content)
            sequence = [case.api_endpoint for case in test_scenario.scenario.todo_tests]
            accepted, reason = global_vars_funcs_configs.catalog.accept_scenario(sequence, last_msg_content)
            if not accepted:
                print(f'SCENARIO REJECTED: {reason}')
                test_scenario.scenario = test_scenario.TestScenario()
                scenario_detail_round = False
                scenario_generation_retries += 1
                if scenario_generation_retries > 2:
                    raise ValueError('Scenario generation exhausted: ' + reason)
                groupchat.messages.append({'role': 'user', 'content': 'Regenerate the scenario. ' + reason})
                return test_scenario_agent
            test_scenario.record_test_scenario(last_msg_content[:500], sequence)
            test_scenario.scenario.accepted = True
            print('SCENARIO ACCEPTED: ' + json.dumps(sequence))
            if SCORING_ONLY_MODE:
                test_scenario.add_missing_test_cases_from_scenario_text(test_scenario.scenario.scenario_text)
                if REST_LEAGUE_MODE and REST_LEAGUE_SEED_ALL_ENDPOINTS:
                    test_scenario.add_uncovered_api_endpoints_as_test_cases(limit=_endpoint_seed_limit())
                return api_invoke_agent
            return test_scenario_recorder_agent
        elif last_speaker is test_scenario_recorder_agent:
            if "STATE-FLOW-MESSAGE:TEST SCENARIO RECORD COMPLETED" in last_msg_content:
                test_scenario.add_missing_test_cases_from_scenario_text(test_scenario.scenario.scenario_text)
                if REST_LEAGUE_MODE and REST_LEAGUE_SEED_ALL_ENDPOINTS:
                    test_scenario.add_uncovered_api_endpoints_as_test_cases(limit=_endpoint_seed_limit())
                return api_invoke_agent
            else:
                return test_scenario_recorder_agent
        elif last_speaker is api_invoke_agent:
            # do_request records successful HTTP responses in todo_resps. This is more reliable than
            # parsing AutoGen's message text, whose tool-response formatting varies by model/client.
            if len(test_scenario.scenario.todo_resps) > 0:
                actual = test_scenario.scenario.todo_resps[-1].get('api_endpoint')
                if (test_scenario.scenario.todo_tests
                        and actual == global_vars_funcs_configs.catalog.normalize_endpoint(
                            test_scenario.scenario.todo_tests[0].api_endpoint)['endpoint']):
                    api_retry_last_pop_test_case = test_scenario.scenario.todo_tests.pop(0)
                    print(f"REMOVE CASE: there are {len(test_scenario.scenario.todo_tests)} test cases now.")
                if SCORING_ONLY_MODE:
                    record_all_pending_responses_with_status_oracle("SCORING_ONLY_MODE records HTTP status immediately")
                    last_recorded_result_count = len(tools.right_results) + len(tools.wrong_results)
                    return api_invoke_agent
                if REST_LEAGUE_MODE:
                    record_pending_response_with_status_oracle("REST_LEAGUE_MODE records HTTP status immediately")
                    last_recorded_result_count = len(tools.right_results) + len(tools.wrong_results)
                    return api_invoke_agent
                return validation_agent
            if "STATE-FLOW-MESSAGE:NO MORE REQUESTS" in last_msg_content:
                if test_scenario.scenario.todo_tests:
                    groupchat.messages.append({'role': 'user', 'content':
                        'The executable queue is not empty. Call get_next_test_case and execute it.'})
                    return api_invoke_agent
                print("End of this test scenario.")
                return None
            else:
                return api_invoke_agent
        elif last_speaker is validation_agent:
            # After successful record, call next API
            current_recorded_result_count = len(tools.right_results) + len(tools.wrong_results)
            if current_recorded_result_count > last_recorded_result_count:
                last_recorded_result_count = current_recorded_result_count
                if len(test_scenario.scenario.todo_resps) > 0:
                    test_scenario.scenario.todo_resps.pop(0)
                print(
                    f"REMOVE RESPONSE: there are {len(test_scenario.scenario.todo_resps)} responses left (0 is right).")
    
                # Request failed, decide whether to retry based on remaining api_retry_in_round_max_time
                if 'align_with_expected=False' in last_msg_content:
                    print(f"Find align_with_expected=False. {api_retry_in_round_max_time=} {api_retry_last_pop_test_case.api_endpoint=}")
                    if api_retry_in_round_max_time > 0 and type(api_retry_last_pop_test_case) is test_scenario.TestCase:
                        api_retry_in_round_max_time -= 1
                        test_scenario.add_test_case_object(api_retry_last_pop_test_case)
                        api_retry_last_pop_test_case = None
                        return api_invoke_agent
                    else:
                        print(f"{api_retry_in_round_max_time=} Exceeded maximum failure attempts. return api_recorder_agent")
    
                return api_recorder_agent
            else:
                return validation_agent
        elif last_speaker is api_recorder_agent:
            # record_useful_items is returned when call is successful, continue with next API
            if "finished record_useful_items" in last_msg_content and len(last_msg.get('tool_responses', [])) > 0:
                return api_invoke_agent
            # Nothing worth recording, continue to next round
            elif "STATE-FLOW-MESSAGE:NO ITEMS SHOULD RECORD" in last_msg_content:
                return api_invoke_agent
            # Stuck in continuous api_recorder_agent unable to record state, proceed to next request
            elif "Error: Function" in last_msg_content and "not found" in last_msg_content:
                if len(previous_agent_names) > 5 and all(name == "api_recorder_agent" for name in previous_agent_names[-5:]):
                    print("api_recorder_agent called continuously more than 5 times, switching to api_invoke_agent")
                    return api_invoke_agent
                else:
                    return api_recorder_agent
            # record_api_reflections is returned when call fails, end this round of testing
            elif "finished record_api_reflections" in last_msg_content and len(last_msg.get('tool_responses', [])) > 0:
                return None
            else:
                return api_recorder_agent
        else:
            return None

    if SCORING_ONLY_MODE:
        group_agents = [test_scenario_agent, api_invoke_agent]
        manager_system_message = """Agents and its tools:
* test_scenario_agent:
    + generate a concrete REST API test scenario

* api_invoke_agent:
    + get_next_test_case: get next api test info
    + do_request: Do REST API request

Scoring mode:
* 2XX responses contribute to operation coverage.
* Unique 5XX responses contribute to fault detection.
* Responses are recorded with a status-code oracle; no LLM validation agent is used.
"""
    else:
        group_agents = [test_scenario_agent, test_scenario_recorder_agent, api_invoke_agent, api_recorder_agent, validation_agent]
        manager_system_message = """Agents and its tools:
* test_scenario_recorder_agent:
    + add_test_case: record rest api test case
    + record_test_scenario: record rest api test scenario

* api_invoke_agent:
    + get_next_test_case: get next api test info
    + do_request: Do REST API request

* api_recorder_agent:
    + record_useful_items: record useful item among previous response data
    + record_api_reflections: record reflection for api call failure

* validation_agent:
    + get_next_response_for_validation: get next api request and response for validation
    + record_result: record result
"""

    groupchat = autogen.GroupChat(
        agents=group_agents,
        messages=[],
        max_round=int(os.getenv("GROUPCHAT_MAX_ROUND", "400")),
        speaker_selection_method=state_transition,
    )
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config,
                                                  system_message=manager_system_message)

    group_chat_manager.initiate_chat(
        test_scenario_agent,
        message="Start LLM based REST API Test loop",
        # summary_method="reflection_with_llm",
    )

    global test_scenario_response_message
    test_scenario_response_message = test_scenario.scenario.scenario_text

    if SCORING_ONLY_MODE or REST_LEAGUE_MODE:
        while len(test_scenario.scenario.todo_resps) > 0:
            record_pending_response_with_status_oracle("status-oracle final drain")
        return


    test_scenario_summary_prompt = f"""
Summarize a REST API Test Scenario based on its description and execution data. Highlight key outcomes, identify errors or mismatches, and provide actionable suggestions for improvement.

# Inputs
## Test Scenario Description by another agent
{test_scenario_response_message}

## Execution Data: Specifics on correct ("Align with expectation") and incorrect ("Not align with expectation") outcomes.

### Right results:
{tools.right_results}

### Wrong results:
{tools.wrong_results}

# Steps
1. **Overall Success/Failure**: Provide a brief statement on the overall success or failure of the scenario.
2. **Error Identification**: Summarize incorrect outcomes and analyze their causes.
3. **Recommendations**: Suggest actionable improvements or next steps, such as adding tests or addressing missing validations.

# Output Format
- A clear and concise paragraph (50 words) summarizing key outcomes, including test failure, errors, and recommendations.

# Examples
## Example 1
The REST API test scenario executed successfully without errors, validating user lifecycle operations, item creation, password updates, and data consistency. Key points: ensure thorough validation for edge cases in password recovery and health checks in production environments.

## Example 2
Test Scenario failed at POST /admin/users due to insufficient permissions. Ensure access-token validation is correctly configured. Recommend adding role and permission checks in future Test Scenario generation to prevent such issues.

# Notes
- Recommendations should be concise, realistic, and actionable.
- Once the summary is created, proceed to call the `record_test_scenario_result_summary` tool to record the result summary.
"""
    test_scenario_summary_agent = ConversableAgent(name="test_scenario_recorder_agent",
                                                    system_message=test_scenario_summary_prompt + "\n\n" + STRICT_JSON_TOOL_CALL_INSTRUCTION,
                                                    llm_config=llm_config,
                                                    human_input_mode="NEVER")
    test_scenario_summary_agent.register_for_llm(name="record_test_scenario_result_summary", description="record test scenario result summary")(test_scenario.record_test_scenario_result_summary)
    test_scenario_summary_agent.register_for_execution(name="record_test_scenario_result_summary")(test_scenario.record_test_scenario_result_summary)

    dummy_proxy = ConversableAgent(name="mentor_agent",
                                   system_message="You are the mentor agent of test_scenario_summary_agent\n\n" + STRICT_JSON_TOOL_CALL_INSTRUCTION,
                                   llm_config=llm_config,
                                   human_input_mode="NEVER")
    dummy_proxy.register_for_llm(name="record_test_scenario_result_summary",description="record test scenario result summary")(test_scenario.record_test_scenario_result_summary)
    dummy_proxy.register_for_execution(name="record_test_scenario_result_summary")(test_scenario.record_test_scenario_result_summary)

    dummy_proxy.initiate_chat(test_scenario_summary_agent,
                              message="Summary this Test Scenario's execution result",
                              max_turns=2)


completed_request_count = 0


def calculate_request_count():
    # Runtime budgets must not depend on whether optional artifacts are written.
    return completed_request_count


def build_scoring_summary(run_started_at: float, run_error=None) -> dict:
    successful_endpoints = sorted({
        item.get("api_endpoint") or f"{item.get('method')} {item.get('api')}"
        for item in tools.all_request_sequence
        if item.get("is_2xx")
    })
    unique_5xx = sorted({
        f"{item.get('api_endpoint') or item.get('api')}|{item.get('response_code')}|{item.get('fault_signature')}"
        for item in tools.all_request_sequence
        if item.get("is_5xx")
    })
    timeline = []
    for idx, item in enumerate(tools.all_request_sequence, start=1):
        timestamp = item.get("timestamp")
        elapsed_since_start_ms = None
        if isinstance(timestamp, (int, float)):
            elapsed_since_start_ms = int((timestamp - run_started_at) * 1000)
        timeline.append({
            "idx": idx,
            "elapsed_since_run_start_ms": elapsed_since_start_ms,
            "endpoint": item.get("api_endpoint") or f"{item.get('method')} {item.get('api')}",
            "status": item.get("response_code"),
            "is_2xx": item.get("is_2xx"),
            "is_5xx": item.get("is_5xx"),
        })

    return {
        "scoring_only_mode": SCORING_ONLY_MODE,
        "hirest_equiv_mode": os.getenv("HIREST_EQUIV_MODE", "true"),
        "request_count": len(tools.all_request_sequence),
        "operation_coverage_2xx_count": len(successful_endpoints),
        "operation_coverage_2xx_endpoints": successful_endpoints,
        "unique_5xx_count": len(unique_5xx),
        "unique_5xx_signatures": unique_5xx,
        "timeline": timeline,
        "run_error": run_error,
        "scheduling": {
            "rule": "exploration_probability_equals_2xx_operation_coverage",
            "last_decision": dict(getattr(getattr(global_vars_funcs_configs.catalog,
                                                    'scheduling_policy', None), 'last', {})),
        },
    }


def sync_planning_memory() -> None:
    """Carry successful operations and execution feedback into the next LLM round."""
    for item in tools.all_request_sequence:
        if not item.get("is_2xx"):
            continue
        endpoint = item.get("api_endpoint")
        if not endpoint:
            try:
                endpoint = global_vars_funcs_configs.catalog.normalize_endpoint(
                    f"{item.get('method')} {item.get('api')}"
                )["endpoint"]
            except (TypeError, ValueError):
                continue
        global_vars_funcs_configs.catalog.covered_2xx.add(endpoint)


def run_sequence_round(max_requests, deadline_timestamp):
    from seqrest.engine.policy import get_policy
    catalog = global_vars_funcs_configs.catalog
    policy = get_policy(catalog)
    ratio = len(catalog.covered_2xx) / max(1, len(catalog.api_swagger_map))
    mode = policy.choose(ratio, random.random())
    before = set(catalog.covered_2xx)
    start = len(tools.all_request_sequence)
    started = time.monotonic()
    try:
        return _run_sequence_round(max_requests, deadline_timestamp, mode)
    finally:
        summary = policy.observe(mode, tools.all_request_sequence[start:],
                                 len(catalog.covered_2xx - before),
                                 time.monotonic() - started)
        print('DYNAMIC SCHEDULER: ' + json.dumps(summary, sort_keys=True))


def _run_sequence_round(max_requests, deadline_timestamp, mode):
    from seqrest.engine.sequence import SequenceRunner, setting
    from seqrest.engine.corpus import get_corpus
    catalog = global_vars_funcs_configs.catalog
    planning_feedback.deadline = deadline_timestamp
    ratio = len(catalog.covered_2xx) / max(1, len(catalog.api_swagger_map))
    def drain(reason):
        while test_scenario.scenario.todo_resps:
            record_response_with_status_oracle(test_scenario.scenario.todo_resps.pop(0), reason)
    runner = SequenceRunner(catalog, tools, adaptive_testing.runtime, drain, deadline_timestamp,
                            min(max_requests, setting('REST_LEAGUE_SEQUENCE_ROUND_REQUESTS', 1200)))
    runner.preparation.bootstrap(getattr(catalog, "required_targets", []))
    corpus = get_corpus(catalog)
    progress = False
    saved_sent = 0
    if mode == 'exploration':
        progress, saved_sent = corpus.run(runner, planning_feedback)

    plans = planning_feedback.repair_queue[:3]
    del planning_feedback.repair_queue[:len(plans)]
    interval = max(1, setting('REST_LEAGUE_EXPLORATION_PLAN_INTERVAL', 4))
    plan_due = (
        mode == 'coverage'
        or not corpus.plans
        or saved_sent == 0
        or corpus.rounds % interval == 0
    )
    if not plans and plan_due and runner.can_send():
        plans = planning_feedback.plan_requests(
            mode,
            count=(
                setting('REST_LEAGUE_EXPLORATION_BATCH_SIZE', 6)
                if mode == 'exploration' else 1
            ),
        )

    # Rejection/cooldown does not cancel independently verified work.
    if not plans and saved_sent == 0 and runner.can_send():
        local_progress, saved_sent = corpus.run(runner, planning_feedback)
        progress = local_progress or progress
    print(
        f'SEQUENCE SCHEDULER: coverage={ratio:.3f}, mode={mode}, '
        f'plans={len(plans)}, saved_requests={saved_sent}'
    )
    for plan in plans:
        if not runner.can_send():
            break
        test_scenario.scenario.accepted = True
        catalog.scenario_signatures.add(json.dumps(plan['steps'], sort_keys=True))
        start = len(tools.all_request_sequence)
        sent, blocked, outputs = runner.execute(plan)
        sequence = [step['operation'] for step in plan['steps']]
        runner.prepare_resources(plan, sent)
        runner.fallback(plan, sent, blocked, outputs)
        planning_feedback.observe_scenario(sequence, tools.all_request_sequence[start:], tools.all_request_sequence)
        for item in blocked:
            op = item['operation']
            if op not in catalog.covered_2xx:
                planning_feedback.failed[op] = {
                    'scenario_sequence': sequence, 'status': 'binding_blocked',
                    'params': {}, 'body': {}, 'response': item['reason'],
                    'attempts': planning_feedback.failed.get(op, {}).get('attempts', 0) + 1}
                planning_feedback.pending.add(op)
        # Replays always recreate prerequisites; no frozen successful-request IDs.
        if sent and not blocked and all(item.get('is_2xx') for item in sent):
            for _ in range(setting('REST_LEAGUE_SUCCESS_REPLAY_ROUNDS', 1)):
                if not runner.can_send():
                    break
                runner.execute(plan, 'sequence_replay')
        runner.explore(plan)
        progress = planning_feedback.observe_batch(plan, tools.all_request_sequence[start:]) or progress
        if runner.can_send() and planning_feedback.reflection_due():
            planning_feedback.reflect()
    if not tools.all_request_sequence and runner.can_send():
        # No valid local candidate exists: avoid a busy loop during cooldown.
        remaining = (
            max(0, deadline_timestamp - time.time())
            if deadline_timestamp is not None else 0.5
        )
        time.sleep(min(0.5, remaining))
    return progress


def run_full_coverage_batch(max_requests: int, deadline_timestamp=None) -> bool:
    """Execute a small LLM-planned comparison batch without replaying the catalog."""
    plans = planning_feedback.plan_boundary_batch()
    if not plans:
        return False
    test_scenario.scenario.accepted = True
    batch_limit = min(max_requests, max(1, env_int('REST_LEAGUE_FULL_COVERAGE_BATCH_REQUESTS', 36)))
    progress = False
    selected_operations = []

    def can_send():
        return (len(tools.all_request_sequence) < batch_limit
                and (deadline_timestamp is None or time.time() < deadline_timestamp))

    def drain(reason):
        while test_scenario.scenario.todo_resps:
            record_response_with_status_oracle(test_scenario.scenario.todo_resps.pop(0), reason)

    for plan in plans:
        if not can_send():
            break
        start = len(tools.all_request_sequence)
        print('BOUNDARY BATCH: scenario ' + json.dumps(plan, ensure_ascii=False))
        for step in plan['steps']:
            if not can_send():
                break
            endpoint = step['operation']
            selected_operations.append(endpoint)
            test_scenario.current_test_case = test_scenario.TestCase(
                f'Comparative boundary scenario: {plan["goal"]}', endpoint,
                'LLM-selected OpenAPI parameter class for branch and fault exploration.',
                'Observe response status and body diversity.',
            )
            try:
                adaptive_testing.runtime.phase = 'scenario'
                tools.do_openapi_request(endpoint, variant=step['variant'])
            except Exception as exc:
                print(f'BOUNDARY BATCH: request failed {endpoint}: {type(exc).__name__}: {exc}')
            drain('LLM comparative boundary scenario')
        progress = planning_feedback.observe_batch(plan, tools.all_request_sequence[start:]) or progress

    # Mutation remains targeted to operations in the generated sequences.
    # One successful-seed candidate per operation per batch; no catalog-wide spray.
    for endpoint in dict.fromkeys(selected_operations):
        if not can_send():
            break
        candidate = adaptive_testing.runtime.next_mutation(
            endpoint, global_vars_funcs_configs.catalog.api_swagger_map[endpoint])
        if candidate is None:
            continue
        test_scenario.current_test_case = test_scenario.TestCase(
            f'Boundary seed mutation: {endpoint}', endpoint,
            'Single-parameter mutation of a successful scenario request.',
            'Observe new successful response branches and unique server faults.',
        )
        start = len(tools.all_request_sequence)
        try:
            adaptive_testing.runtime.phase = 'mutation'
            tools.do_request(**candidate)
        except Exception as exc:
            print(f'BOUNDARY BATCH: mutation failed {endpoint}: {type(exc).__name__}: {exc}')
        finally:
            adaptive_testing.runtime.phase = 'scenario'
        drain('Targeted boundary mutation')
        progress = planning_feedback.observe_batch({'steps': []}, tools.all_request_sequence[start:]) or progress
    return progress


if __name__ == '__main__':
    if global_vars_funcs_configs.CONFIG_LLM_API_KEY is None and not (REST_LEAGUE_MODE and REST_LEAGUE_DIRECT_FALLBACK):
        print("Please export LLM_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY environment variable")
        sys.exit(1)

    LOOP_CNT = int(os.getenv("LOOP_CNT", "20"))
    MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", "1000"))
    TIME_BUDGET_SECONDS = int(float(os.getenv("TIME_BUDGET", "0")) * 60)
    started_at = time.time()
    full_coverage_batches = 0
    full_coverage_stagnation = 0
    fault_probes_done = False

    # REST League auth pre-handshake: register/login once up front so that the
    # LLM scenario phase (which runs before the deterministic fallback) already
    # sends an Authorization header instead of wasting budget on 401s.
    try:
        tools.perform_auth_handshake()
    except Exception as exc:
        print(f"AUTH HANDSHAKE: initial handshake failed: {exc}")

    for i in range(LOOP_CNT):
        if TIME_BUDGET_SECONDS > 0 and time.time() - started_at >= TIME_BUDGET_SECONDS:
            print(f"time budget reached: {TIME_BUDGET_SECONDS}s, exit...")
            break

        rc = calculate_request_count()
        print(f"request count: {rc}")
        if rc >= MAX_REQUESTS:
            print(f"request count more than or equal to {MAX_REQUESTS}, exit...")
            break
        remaining_request_budget = MAX_REQUESTS - rc

        # reset all global vars
        test_scenario_response_message = ""
        test_scenario.scenario = test_scenario.TestScenario()
        tools.all_request_sequence.clear()
        tools.right_results.clear()
        tools.wrong_results.clear()
        adaptive_testing.runtime.phase = 'scenario'
        if REST_LEAGUE_RESET_RESOURCE_POOL_EACH_LOOP:
            tools.resource_pool.clear()
        seed = REST_LEAGUE_SEED + i
        llm_config["config_list"][0]["cache_seed"] = seed

        print(f"main: {i}, {seed=}, timestamp={int(time.time())}")
        run_started_at = time.time()
        run_error = None
        full_coverage_before = set(global_vars_funcs_configs.catalog.api_swagger_map) <= global_vars_funcs_configs.catalog.covered_2xx
        deadline_timestamp = started_at + TIME_BUDGET_SECONDS if TIME_BUDGET_SECONDS > 0 else None
        batch_progress = False
        planned_sequence = []
        scenario_requests = []
        structured_mode = env_flag('REST_LEAGUE_SEQUENCE_MODE', 'true') and not REST_LEAGUE_SKIP_LLM
        if structured_mode:
            try:
                batch_progress = run_sequence_round(remaining_request_budget, deadline_timestamp)
            except Exception as exc:
                run_error = repr(exc)
                print(f'SEQUENCE ROUND FAILED: {run_error}')
        elif full_coverage_before and not REST_LEAGUE_SKIP_LLM:
            batch_progress = run_full_coverage_batch(remaining_request_budget, deadline_timestamp)
            full_coverage_batches += 1
            full_coverage_stagnation = 0 if batch_progress else full_coverage_stagnation + 1
        elif REST_LEAGUE_SKIP_LLM:
            print("REST_LEAGUE_SKIP_LLM=true, skipping LLM agents and using direct OpenAPI fallback.")
        else:
            try:
                main()
            except Exception as exc:
                run_error = repr(exc)
                print(f"main failed but result will be recorded: {run_error}")
            if test_scenario.scenario.accepted and test_scenario.generated_test_scenarios:
                planned_sequence = test_scenario.generated_test_scenarios[-1].get('api_sequence', [])
            scenario_requests = list(tools.all_request_sequence)

        if not structured_mode and not full_coverage_before:
            run_rest_league_direct_fallback(remaining_request_budget, deadline_timestamp=deadline_timestamp)
        elif structured_mode:
            print('SEQUENCE ROUND: scoped execution completed; no catalog-wide replay.')
        else:
            print('FULL COVERAGE: comparative LLM batch completed; no catalog-wide replay.')

        # One-shot fault-diversity path probes even in structured/sequence mode
        # (the deterministic fallback above is skipped there, so probes would
        # otherwise never run).  Triggered once the run is at least half done,
        # after coverage phases have had time to populate the resource pool.
        if not fault_probes_done and REST_LEAGUE_FAULT_DIVERSITY_REQUESTS > 0:
            if TIME_BUDGET_SECONDS <= 0 or time.time() - started_at >= TIME_BUDGET_SECONDS * 0.5:
                run_rest_league_direct_fallback(
                    remaining_request_budget,
                    deadline_timestamp=deadline_timestamp,
                    skip_gate=True,
                    probes_only=True,
                )
                fault_probes_done = True

        # A timed run must keep exploring after all operations reach 2xx.
        # Batch/stagnation guards are only an exit for runs with no time budget;
        # otherwise a small API such as notebook-manager goes idle early while
        # RESTgym continues to count down the experiment.
        full_coverage_should_stop = full_coverage_before and (
            REST_LEAGUE_SKIP_LLM
            or (TIME_BUDGET_SECONDS <= 0 and (
                full_coverage_batches >= FULL_COVERAGE_MAX_BATCHES
                or full_coverage_stagnation >= FULL_COVERAGE_STAGNATION_BATCHES
            ))
        )
        if full_coverage_should_stop:
            # This configurable minimum retains short-run results; it is not
            # the exploration target. Persist these final
            # requests in the normal round summary before exiting.
            shortfall = max(0, VERIFICATION_MIN_REQUESTS - calculate_request_count()
                            - len(tools.all_request_sequence))
            if shortfall:
                print(f'FULL COVERAGE: filling {shortfall} requests for RESTgym verification only.')
                previous_rounds = REST_LEAGUE_VALID_TRAFFIC_ROUNDS
                REST_LEAGUE_VALID_TRAFFIC_ROUNDS = shortfall
                try:
                    run_rest_league_direct_fallback(
                        len(tools.all_request_sequence) + shortfall,
                        deadline_timestamp=deadline_timestamp,
                        skip_gate=True,
                        sustain=True,
                    )
                finally:
                    REST_LEAGUE_VALID_TRAFFIC_ROUNDS = previous_rounds

        while (
            len(test_scenario.scenario.todo_resps) > 0
            and len(tools.right_results) + len(tools.wrong_results) < len(tools.all_request_sequence)
        ):
            pending_response = test_scenario.scenario.todo_resps.pop(0)
            record_response_with_status_oracle(pending_response, "main returned before validation completed")

        sync_planning_memory()
        if planned_sequence:
            planning_feedback.observe_scenario(planned_sequence, scenario_requests, tools.all_request_sequence)
            if deadline_timestamp is None or time.time() < deadline_timestamp - 15:
                planning_feedback.reflect()
        print('EXECUTION SUMMARY: ' + json.dumps(adaptive_testing.runtime.summary(global_vars_funcs_configs.catalog)))
        test_scenario.record_execution_summary(
            tools.all_request_sequence, tools.right_results, tools.wrong_results
        )

        completed_request_count += len(tools.all_request_sequence)
        if global_vars_funcs_configs.CONFIG_SAVE_ARTIFACTS:
            with open(f"{global_vars_funcs_configs.CONFIG_LOG_PATH}/main_result_{int(time.time()*1000)}_PID_{os.getpid()}.json", "w") as f:
                f.write(
                    json.dumps(
                        {
                            "all_cnt": len(tools.all_request_sequence),
                            "all_request_sequence": tools.all_request_sequence,
                            "successful_sequences": list(
                                getattr(
                                    getattr(global_vars_funcs_configs.catalog,
                                            'sequence_corpus', None),
                                    'plans', {}
                                ).values()
                            ),
                            "right_results": tools.right_results,
                            "wrong_results": tools.wrong_results,
                            "scoring_summary": build_scoring_summary(run_started_at, run_error=run_error),
                            "test_scenario_response_message": test_scenario_response_message,
                            "run_error": run_error,
                            "rest_league_mode": REST_LEAGUE_MODE,
                            "rest_league_seed": seed,
                            "rest_league_mutation_rounds": REST_LEAGUE_MUTATION_ROUNDS,
                        }
                    )
                )

        if full_coverage_should_stop:
            print('FULL COVERAGE: comparative exploration stopped after batch/stagnation guard; leaving keepalive to run.sh.')
            break
        if not full_coverage_before and set(global_vars_funcs_configs.catalog.api_swagger_map) <= global_vars_funcs_configs.catalog.covered_2xx:
            print('FULL COVERAGE: switching to comparative LLM boundary-scenario batches.')
        if run_error is not None:
            time.sleep(1)
