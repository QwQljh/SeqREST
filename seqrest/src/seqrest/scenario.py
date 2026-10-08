import random
import re
import string
import json
from typing import List, Optional

import seqrest.config as global_vars_funcs_configs
import seqrest.http as tools
import seqrest.planning.retrieval as retrieval
from seqrest.config import catalog
from seqrest.openapi.swagger import swagger_to_text


def record_test_scenario(summary: str, api_sequence: List[str]) -> str:
    global generated_test_scenarios
    normalized = []
    for value in api_sequence:
        if isinstance(value, dict):
            value = value.get('api_endpoint') or value.get('endpoint') or value.get('api')
        if not isinstance(value, str):
            continue
        normalized.append(catalog.normalize_endpoint(value)['endpoint'])
    api_sequence = normalized
    if getattr(scenario, 'recorded', False):
        return 'current scenario already recorded'
    generated_test_scenarios.append({
        "summary": summary,
        "api_sequence": api_sequence
    })
    scenario.recorded = True

    if len(generated_test_scenarios) > 10:
        generated_test_scenarios = generated_test_scenarios[1:]

    return "record test_scenario success"


def record_test_scenario_result_summary(result_summary: str) -> str:
    global generated_test_scenarios
    if generated_test_scenarios and getattr(scenario, 'recorded', False):
        generated_test_scenarios[-1]["result_summary"] = result_summary
    return "record current_scenario_result_summary success"


def record_execution_summary(requests: list, right_results: list, wrong_results: list) -> None:
    """Persist compact execution feedback for the next scenario-planning round."""
    if not generated_test_scenarios:
        return
    counts = {"2XX": 0, "4XX": 0, "5XX": 0, "other": 0}
    covered = set()
    failures = []
    for item in requests or []:
        status = item.get("response_code")
        if isinstance(status, int):
            bucket = "2XX" if 200 <= status < 300 else "4XX" if 400 <= status < 500 else "5XX" if status >= 500 else "other"
            counts[bucket] += 1
            endpoint = item.get("api_endpoint")
            if bucket == "2XX" and endpoint:
                covered.add(endpoint)
            if bucket in {"4XX", "5XX"} and len(failures) < 8:
                failures.append(f"{endpoint or item.get('method')} -> {status}")
    generated_test_scenarios[-1]["execution"] = {
        "status_counts": counts,
        "covered_2xx": sorted(covered),
        "failure_examples": failures,
        "request_count": len(requests or []),
    }


def get_previous_all_test_scenarios_for_llm() -> str:
    """
    Generate a textual representation of all previous test scenarios
    in an LLM-friendly format.
    """

    global generated_test_scenarios

    if not generated_test_scenarios or len(generated_test_scenarios) == 0:
        return "No test scenarios have been recorded yet."

    description = "Previous scenario memory (compact; use only to avoid exact full-sequence duplicates):\n"
    for idx, scenario in enumerate(generated_test_scenarios[-6:], 1):
        sequence = scenario.get('api_sequence', [])
        execution = scenario.get('execution') or {}
        description += (
            f"Scenario {idx}:\n"
            f"  Sequence: {' -> '.join(sequence)}\n"
        )
        if execution:
            description += (
                f"  Result: {execution.get('status_counts', {})}, "
                f"covered={len(execution.get('covered_2xx', []))}, "
                f"requests={execution.get('request_count', 0)}\n"
            )

    return description


def reset_useful_items():
    global useful_items
    useful_items = {}


def get_swagger_info_by_api_endpoint(api_endpoint: str) -> str:
    api_endpoint = api_endpoint.replace("{{", "{")
    api_endpoint = api_endpoint.replace("}}", "}")

    swagger_info = catalog.api_swagger_map.get(api_endpoint)
    if swagger_info:
        text_info = swagger_to_text(swagger_info)
        text_info = f"Endpoint: {api_endpoint}\n" + text_info
        return text_info
    else:
        return "No Swagger Info"


def normalize_api_endpoint(api_endpoint: str) -> str:
    endpoint = str(api_endpoint or "").strip()
    endpoint = endpoint.strip("`").strip()
    endpoint = endpoint.replace("{{", "{").replace("}}", "}")
    endpoint = re.sub(r"\s+", " ", endpoint)
    endpoint = endpoint.split("?")[0].rstrip(".,;")
    return endpoint


def _known_endpoint_lookup() -> dict:
    return {normalize_api_endpoint(endpoint): endpoint for endpoint in catalog.api_swagger_map.keys()}


def _endpoint_shape(endpoint: str) -> tuple:
    normalized = normalize_api_endpoint(endpoint)
    if " " not in normalized:
        return "", ()

    method, path = normalized.split(" ", 1)
    parts = []
    for part in path.strip("/").split("/"):
        if re.fullmatch(r"\{[^{}]+\}", part):
            parts.append("{}")
        else:
            parts.append(part)
    return method.upper(), tuple(parts)


def resolve_api_endpoint(api_endpoint: str) -> Optional[str]:
    normalized = normalize_api_endpoint(api_endpoint)
    lookup = _known_endpoint_lookup()
    if normalized in lookup:
        return lookup[normalized]

    candidate_shape = _endpoint_shape(normalized)
    if candidate_shape == ("", ()):
        return None

    matches = [
        original
        for original in catalog.api_swagger_map.keys()
        if _endpoint_shape(original) == candidate_shape
    ]
    if len(matches) == 1:
        print(f"ENDPOINT FIX: {api_endpoint} -> {matches[0]}")
        return matches[0]

    return None


def _extract_field(block: str, field_name: str) -> Optional[str]:
    pattern = re.compile(
        rf"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?{re.escape(field_name)}(?:\*\*)?\s*:\s*(.+)$"
    )
    match = pattern.search(block)
    if match:
        return match.group(1).strip().strip("`").strip()
    return None


def _extract_endpoint_from_text(text: str) -> Optional[str]:
    lookup = _known_endpoint_lookup()
    normalized_text = normalize_api_endpoint(text)

    match = re.search(
        r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(/[^\s,\])`]+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        resolved = resolve_api_endpoint(f"{match.group(1).upper()} {match.group(2)}")
        if resolved:
            return resolved

    for normalized_endpoint, original_endpoint in lookup.items():
        if normalized_endpoint in normalized_text:
            return original_endpoint

    return None


def add_missing_test_cases_from_scenario_text(scenario_text: str) -> int:
    global scenario

    if not isinstance(scenario_text, str) or not scenario_text.strip():
        return 0

    existing_cases = {
        (normalize_api_endpoint(test_case.api_endpoint), str(test_case.title).strip())
        for test_case in scenario.todo_tests
    }
    api_sequence = []
    if generated_test_scenarios and getattr(scenario, 'recorded', False):
        api_sequence = generated_test_scenarios[-1].get("api_sequence", []) or []

    # Both numbered steps and the model's unnumbered Markdown Title blocks.
    blocks = re.split(r"(?m)^\s*\d+[.)]\s+", scenario_text)
    if len(blocks) <= 1:
        blocks = re.split(r"(?im)(?=^\s*(?:\*\*)?Title(?:\*\*)?\s*:)", scenario_text)
    else:
        blocks = blocks[1:]

    added = 0
    for idx, block in enumerate(blocks, start=1):
        endpoint_field = _extract_field(block, "API Endpoint") or _extract_field(block, "Endpoint")
        endpoint = _extract_endpoint_from_text(endpoint_field or block)
        if endpoint is None:
            continue

        title = _extract_field(block, "Title") or f"Parsed scenario step {idx}: {endpoint}"
        case_key = (normalize_api_endpoint(endpoint), title.strip())
        if case_key in existing_cases:
            continue

        description = _extract_field(block, "Description") or block.strip()[:1000]
        expected_resp = (
            _extract_field(block, "Expected Response")
            or "The request should complete without an unexpected server-side 5XX response."
        )

        scenario.todo_tests.append(TestCase(title, endpoint, description, expected_resp))
        existing_cases.add(case_key)
        added += 1

    existing_endpoints = {
        normalize_api_endpoint(test_case.api_endpoint)
        for test_case in scenario.todo_tests
    }
    for idx, endpoint_text in enumerate(api_sequence, start=1):
        endpoint = _extract_endpoint_from_text(str(endpoint_text))
        if endpoint is None or normalize_api_endpoint(endpoint) in existing_endpoints:
            continue
        scenario.todo_tests.append(
            TestCase(
                f"Recorded API sequence fallback {idx}: {endpoint}",
                endpoint,
                "Fallback test case created from record_test_scenario api_sequence.",
                "The request should complete without an unexpected server-side 5XX response.",
            )
        )
        existing_endpoints.add(normalize_api_endpoint(endpoint))
        added += 1

    if added:
        print(f"SCENARIO PARSER: added {added} missing test cases from scenario text/api_sequence.")
    return added


def add_uncovered_api_endpoints_as_test_cases(limit: Optional[int] = None) -> int:
    global scenario

    existing = {
        normalize_api_endpoint(test_case.api_endpoint)
        for test_case in scenario.todo_tests
    }
    added = 0

    for endpoint in catalog.api_swagger_map.keys():
        if limit is not None and added >= limit:
            break
        if normalize_api_endpoint(endpoint) in existing or endpoint in catalog.covered_2xx:
            continue
        scenario.todo_tests.append(
            TestCase(
                f"REST League coverage seed: {endpoint}",
                endpoint,
                (
                    "Generate one black-box REST request for this OpenAPI operation. "
                    "Prefer valid parameters that can obtain a 2XX response; use boundary values only when safe."
                ),
                "The request should complete without an unexpected server-side 5XX response.",
            )
        )
        existing.add(normalize_api_endpoint(endpoint))
        added += 1

    if added:
        print(f"COVERAGE SEED: added {added} uncovered OpenAPI endpoints as test cases.")
    return added


class TestCase:
    def __init__(self, title: str, api_endpoint: str, description: str, expected_resp: str):
        self.title = title
        self.api_endpoint = api_endpoint
        self.description = description
        self.expected_resp = expected_resp
        self.swagger_info = get_swagger_info_by_api_endpoint(api_endpoint)

    def __repr__(self):
        return f"""**TestCase**:{self.title}
**API Endpoint**:{self.api_endpoint}
**Test Description**:{self.description}
**Expected Response**:{self.expected_resp}
**Swagger API Info**:{self.swagger_info}
"""

    def __str__(self):
        return self.__repr__()


class TestScenario:
    def __init__(self, scenario_text=""):
        self.scenario_text = scenario_text
        self.todo_tests = []
        self.todo_resps = []
        self.accepted = False


def add_test_case(test_case_title: str, api_endpoint: str, description: str, expected_resp: str) -> str:
    global scenario
    try:
        resolved_endpoint = resolve_api_endpoint(api_endpoint)
        if resolved_endpoint is None:
            return f"ignored invalid api_endpoint not found in OpenAPI: {api_endpoint}"

        test_case = TestCase(test_case_title, resolved_endpoint, description, expected_resp)
        scenario.todo_tests.append(test_case)
        print(f"ADD CASE: there are {len(scenario.todo_tests)} test cases now.")
        return "success"
    except Exception as e:
        return f"exception: {e}"


def add_test_case_object(test_case: TestCase, idx=0):
    if type(test_case) == TestCase:
        scenario.todo_tests.insert(idx, test_case)
        print(f"ADD CASE: add_test_case_object {idx=} {test_case.title=} there are {len(scenario.todo_tests)} test cases now.")


def get_next_test_case() -> str:
    global scenario, current_test_case

    if len(scenario.todo_tests) == 0:
        return "No more test case"
    else:
        tc = scenario.todo_tests[0]  # type: TestCase
        current_test_case = tc
        print(f"current_test_case changed: {current_test_case.title}")

        tc_info = tc.__repr__()
        useful_item_text = convert_useful_items_to_text()
        if type(useful_item_text) == str and len(useful_item_text) > 0:
            useful_item_text = retrieval.bm25_filter_useful_items(useful_item_text, tc_info)
            tc_info = "\nUseful Information returned by previous REST API:\n" + useful_item_text + "\n\n" + "Test Case Info:" +  tc_info

        reflect_info = get_api_reflections_for_llm(tc.api_endpoint)
        if reflect_info:
            tc_info = tc_info + "\nReflection info for this API:\n" + reflect_info

        return tc_info


def add_next_response_for_validation(item):
    global scenario
    scenario.todo_resps.append(item)
    print(f"ADD RESP: there are {len(scenario.todo_resps)} responses")


def get_next_response_for_validation() -> str:
    """
    :return:
    """
    global scenario

    if len(scenario.todo_resps) > 0:
        item = scenario.todo_resps[0]

        result = ""
        if current_test_case is not None:
            result += "Current Validate Test Case:\n"
            result += current_test_case.__repr__()

        result += f"\nAPI Execution Result:"
        result += f"\tRequest:{item.get('request_data')}\n"
        result += f"\tResponse Code:{item.get('response_code')}\n"
        result += f"\tResponse Data:{item.get('response_data')[:1000] if len(str(item.get('response_data'))) > 1000 else item.get('response_data')}\n"

        return result
    else:
        return "no available response"



def record_useful_items(items: dict) -> str:
    global useful_items  # type: dict[str, list]

    def process_nested(prefix: str, data: dict) -> None:
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                if "value" in value and "description" in value:
                    # This is a leaf node with value and description
                    process_single_item(current_path, value)
                else:
                    # This is a nested dictionary
                    process_nested(current_path, value)
            elif isinstance(value, list):
                # Handle list of dictionaries
                for item in value:
                    if isinstance(item, dict):
                        process_nested(current_path, item)

    def process_single_item(name: str, details: dict) -> None:
        """Helper function to process a single item with value and description"""
        value = details.get('value')
        description = details.get('description', '')

        if name not in useful_items:
            useful_items[name] = []

        existing_entry = next((entry for entry in useful_items[name] if entry['value'] == value), None)

        if existing_entry:
            existing_entry['description'] = description
        else:
            if len(str(useful_items[name])) <= 500 and len(str(value) + str(description)) <= 500:
                useful_items[name].append({
                    'value': value,
                    'description': description
                })

            if len(useful_items[name]) > 5:
                useful_items[name].pop(0)

    # Start processing from root
    process_nested("", items)
    return "finished record_useful_items"


def convert_useful_items_to_text():
    if global_vars_funcs_configs.CONFIG_ABLATION == global_vars_funcs_configs.ABLATION_NO_RELEVANT_PARAMETER:
        print(f"{global_vars_funcs_configs.CONFIG_ABLATION=}, no relevant parameters")
        return None

    global useful_items  # type: dict[str: list]

    if len(useful_items) > 0:
        descriptions = []

        for name, values in useful_items.items():
            value_descriptions = []

            for item in values:
                value = item.get('value', '')
                description = item.get('description', '')
                value_descriptions.append(f"Value: {value}, Description: {description}")

            descriptions.append(f"The param '{name}' previous returned values: " + "; ".join(value_descriptions))

        return "\n".join(descriptions)
    else:
        return None


def record_api_reflections(api_endpoint: str, issue_title: str, issue_detail: str) -> str:
    global api_reflections

    api_endpoint = api_endpoint.replace("{{", "{")
    api_endpoint = api_endpoint.replace("}}", "}")

    # Create the issue entry
    issue_entry = {
        "issue": issue_title,
        "details": issue_detail,
    }

    # Add or update the issues for the given API endpoint
    if api_endpoint not in api_reflections:
        # Initialize a new list with the current issue
        api_reflections[api_endpoint] = [issue_entry]
    else:
        # Check if the same issue already exists for the endpoint
        existing_issues = api_reflections[api_endpoint]
        if issue_entry not in existing_issues:
            # Add the issue if it's unique
            existing_issues.append(issue_entry)
            # Ensure only the latest 3 issues are kept
            if len(existing_issues) > 3:
                existing_issues.pop(0)

    return "finished record_api_reflections"


def get_api_reflections_for_llm(api_endpoint: str) -> str:
    if global_vars_funcs_configs.CONFIG_ABLATION == global_vars_funcs_configs.ABLATION_NO_REFLECTION:
        print(f"{global_vars_funcs_configs.CONFIG_ABLATION=}, No reflections.")
        return None

    global api_reflections
    api_endpoint = api_endpoint.replace("{{", "{")
    api_endpoint = api_endpoint.replace("}}", "}")

    if 'api_reflections' not in globals() or api_endpoint not in api_reflections:
        return None

    reflections = api_reflections.get(api_endpoint, [])
    if not reflections:
        return None

    descriptions = []
    for reflection in reflections:
        issue = reflection.get("issue", "No issue title provided")
        details = reflection.get("details", "No details provided")
        descriptions.append(f"Issue: {issue}. Details: {details}.")

    return "\n".join(descriptions)

# init all vars
'''
Test Scenario recording related.
Example structure:
{
    "summary": "",
    "api_sequence": ["POST"]
}
'''
generated_test_scenarios = []

'''
{
"user_id": [1,2,3],
"username": ["max", "mbx"]
}
'''
useful_items = {}

'''
{
    "GET /api/items": [{
        "issue": "missing_access_token",
        "details": "Authorization header is missing or improperly formatted in the request headers.",
    }]
}
'''
api_reflections = {}


scenario = TestScenario()

current_test_case = None  # type: TestCase
