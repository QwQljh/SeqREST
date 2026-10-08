import json
import base64
import copy
import os
import re
import time
from distutils.util import strtobool
from urllib.parse import quote, urlparse

import requests
from pydantic import BaseModel, Field
from typing import Annotated, Optional, Union

import seqrest.config as global_vars_funcs_configs
import seqrest.engine.equivalence as hirest_equivalence
import seqrest.scenario as test_scenario

all_request_sequence = []
right_results = []
wrong_results = []
resource_pool = {}

# --- Authentication handshake state (REST League) ---
auth_state = {
    "token": None,
    "token_type": "Bearer",
    "refresh_token": None,
    "attempted": False,
    "email": None,
    "password": None,
    "register_endpoint": None,
    "login_endpoint": None,
    "token_acquired_at": None,
    "token_validity_seconds": None,
    "register_payload": None,
    "credentials": [],
    "credential_override": None,
}
_in_handshake = False
_last_401_count = 0
_handshake_401_budget = 8
_AUTH_WALL_MARKERS = (
    "auth error",
    "access denied",
    "not authenticated",
    "unauthorized",
    "authentication required",
    "token expired",
    "invalid token",
    "missing bearer",
    "missing authorization",
    "invalid compact jwt",
)

# Markers that unambiguously mean the presented credential is rejected/invalid
# (hard auth wall: a fresh handshake can help).  Generic 403/404 permission
# denials like "access denied" without these markers are soft: when a token is
# already stored they are usually a role/permission issue, NOT a stale token,
# so re-running the handshake only wastes budget.
_HARD_AUTH_MARKERS = (
    "token expired",
    "invalid token",
    "invalid compact jwt",
    "missing bearer",
    "missing authorization",
    "authentication required",
    "not authenticated",
    "unauthorized",
)


def _is_auth_wall_response(status_code, response_body, has_token=None):
    """A response is an auth wall when it is 401 (always), or 403/404 with
    explicit authentication markers.  When a token is already stored, only
    HARD markers (stale/invalid token) count as a wall so that role/permission
    403s do not trigger useless re-handshakes."""
    if status_code == 401:
        return True
    text = str(response_body or "")
    low = text.lower()
    if status_code not in (403, 404):
        return False
    if has_token:
        return any(marker in low for marker in _HARD_AUTH_MARKERS)
    return any(marker in low for marker in _AUTH_WALL_MARKERS)


def _parse_jwt_validity(token):
    """Return token validity in seconds by decoding a JWT payload (exp - iat).
    Returns None when the token is not a JWT or the claims are missing."""
    try:
        parts = str(token or "").split(".")
        if len(parts) < 2:
            return None
        # base64url decode payload (middle segment)
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload_b64))
        if isinstance(data, dict) and "exp" in data and "iat" in data:
            return int(data["exp"]) - int(data["iat"])
    except Exception:
        pass
    return None


def _auth_enabled() -> bool:
    return _env_flag("REST_LEAGUE_AUTH_HANDSHAKE", "true")


def _find_auth_endpoints():
    """Locate register/login POST endpoints from the OpenAPI catalog."""
    register = None
    login = None
    for endpoint, op in global_vars_funcs_configs.catalog.api_swagger_map.items():
        try:
            method, path = endpoint.split(" ", 1)
        except ValueError:
            continue
        if method != "POST":
            continue
        low = path.lower()
        if not any(k in low for k in ("auth", "login", "signin", "register", "signup", "user", "token")):
            continue
        if any(k in low for k in ("register", "signup")):
            register = endpoint
        elif any(k in low for k in ("login", "signin", "authenticate")):
            login = endpoint
    return register, login


def _admin_role_value(schema):
    """If a register-body field carries an enum containing an ADMIN-like role,
    return (field_name, admin_value) so the handshake can register a privileged
    user.  Returns (None, None) when the API has no role concept."""
    if not isinstance(schema, dict):
        return None, None
    properties = schema.get("properties", {}) or {}
    for field_name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        enum = prop.get("enum")
        if not isinstance(enum, list) or not enum:
            continue
        low_name = str(field_name).lower()
        if not any(k in low_name for k in ("user", "role", "type", "access", "permission", "group")):
            continue
        admin_candidates = [v for v in enum if "ADMIN" in str(v).upper()]
        if admin_candidates:
            return field_name, admin_candidates[0]
    return None, None


def _register_payload_with_role(register_endpoint):
    """Build a register request payload; when the register schema exposes an
    ADMIN role we request it, and we always use a unique email so repeated
    handshakes do not collide with an existing account."""
    payload = {}
    try:
        op = global_vars_funcs_configs.catalog.api_swagger_map.get(register_endpoint, {})
        body_schema = (
            op.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        if not body_schema and "allOf" not in body_schema:
            body_schema = _schema_with_parameter_context({})  # fallback: empty
        # Resolve $ref for the body schema
        try:
            body_schema = _resolve_ref(body_schema)
        except Exception:
            pass
        props = body_schema.get("properties", {}) or {}
        for field_name, prop in props.items():
            prop = _resolve_ref(prop)
            example = _schema_example(prop, field_name, variant=0, required=True)
            payload[field_name] = example
        role_field, admin_value = _admin_role_value(body_schema)
        if role_field and admin_value:
            payload[role_field] = admin_value
            print(f"AUTH HANDSHAKE: register with privileged role {role_field}={admin_value}")
        # Ensure a unique email per handshake so re-handshakes do not 409.
        for field_name, prop in props.items():
            prop = _resolve_ref(prop)
            low = str(field_name).lower()
            if "email" in low:
                payload[field_name] = _unique_register_email()
                break
    except Exception as exc:
        print(f"AUTH HANDSHAKE: role-aware payload build failed, using OpenAPI example: {exc}")
        return None
    return payload if payload else None


def _resolve_ref(schema):
    """Resolve a single-level OpenAPI $ref against the catalog document."""
    if not isinstance(schema, dict) or "$ref" not in schema:
        return schema
    ref = schema.get("$ref", "")
    if not ref.startswith("#/"):
        return schema
    target = global_vars_funcs_configs.catalog.document
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        target = target.get(part, {})
    return target if isinstance(target, dict) else schema


def _extract_auth_token(data):
    """Recursively find an access token inside a response body."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return None
    token_keys = ("access_token", "accesstoken", "accessToken", "jwt", "id_token", "auth_token", "authToken")

    def walk(value, depth=0):
        if depth > 6:
            return None
        if isinstance(value, dict):
            for k, v in value.items():
                kl = str(k).lower()
                if any(tk in kl for tk in token_keys) and isinstance(v, str) and len(v) >= 10:
                    return v
            for v in value.values():
                found = walk(v, depth + 1)
                if found:
                    return found
        elif isinstance(value, list):
            for v in value[:5]:
                found = walk(v, depth + 1)
                if found:
                    return found
        return None

    return walk(data)


def _apply_auth_header(headers):
    """Inject Authorization into a request unless it is the handshake itself.

    The handshake token is authoritative: an LLM might learn a USER-scoped
    token from an earlier register/login it executed itself and embed it into
    the Authorization header of a later business request.  Overwriting with the
    handshake token guarantees business calls carry the privileged account.
    """
    global _in_handshake
    if _in_handshake:
        return headers
    _maybe_refresh_token()
    if auth_state.get("token"):
        headers = dict(headers)
        headers["Authorization"] = f"{auth_state.get('token_type') or 'Bearer'} {auth_state['token']}"
        print(f"AUTH HEADER: injected authoritative {auth_state.get('token_type') or 'Bearer'} token ({len(auth_state['token'])} chars)")
    return headers


def _maybe_refresh_token():
    """Proactively refresh a JWT before it expires so requests do not start
    failing with 401 mid-run.  Only refreshes when the token carries a known
    validity and we are past 60% of its lifetime; otherwise the 401-retry path
    in do_request handles stale tokens."""
    global _in_handshake
    if _in_handshake or not auth_state.get("token"):
        return
    validity = auth_state.get("token_validity_seconds")
    acquired = auth_state.get("token_acquired_at")
    if not validity or not acquired:
        return
    age = time.time() - acquired
    if age < validity * 0.6:
        return
    print(
        f"AUTH REFRESH: token age {int(age)}s >= 60% of {validity}s; re-handshaking"
    )
    try:
        perform_auth_handshake(force=True)
    except Exception as exc:
        print(f"AUTH REFRESH: failed: {exc}")


# --- Generic multi-credential enumeration ------------------------------------
# Field names are hints only; when no name matches we fall back to a structural
# rule (a short upper-case string enum inside a register body is almost always a
# role/type discriminator).  This keeps the mechanism OpenAPI-driven instead of
# hardcoding any single role name.
_ROLE_FIELD_HINTS = (
    "user", "role", "type", "access", "permission", "group",
    "scope", "authority", "account", "kind", "level", "profile", "category",
)
_PRIVILEGED_HINTS = (
    "ADMIN", "SUPER", "ROOT", "OWNER", "MANAGER", "PRIVILEG", "MASTER",
    "STAFF", "OPERATOR", "WRITE", "FULL",
)
_MAX_CREDENTIAL_CANDIDATES = 4
_CREDENTIAL_PROBE_ENABLED = True
_register_email_seq = 0


def _unique_register_email():
    """A register email that cannot collide between two rapid attempts (the
    credential loop registers several accounts within the same millisecond)."""
    global _register_email_seq
    _register_email_seq += 1
    return f"user{int(time.time() * 1000) % 100000000}{_register_email_seq}@example.com"


def _role_field_enums(register_endpoint):
    """[(field_name, [enum values])] for register fields that plausibly encode a
    role / permission discriminator."""
    try:
        op = global_vars_funcs_configs.catalog.api_swagger_map.get(register_endpoint, {})
        body_schema = (
            op.get("requestBody", {}).get("content", {})
            .get("application/json", {}).get("schema", {})
        )
        body_schema = _resolve_ref(body_schema)
    except Exception:
        return []
    props = body_schema.get("properties", {}) or {}
    by_name, by_shape = [], []
    for name, prop in props.items():
        prop = _resolve_ref(prop)
        enum = prop.get("enum")
        if not isinstance(enum, list) or not (1 < len(enum) <= 8):
            continue
        if not all(isinstance(v, str) for v in enum):
            continue
        low = str(name).lower()
        if any(k in low for k in _ROLE_FIELD_HINTS):
            by_name.append((name, enum))
        elif all(len(v) <= 24 and v.upper() == v for v in enum):
            by_shape.append((name, enum))
    return by_name + by_shape


def _role_candidates(register_endpoint):
    """Ordered [(field, value)] role candidates, privileged-looking first.

    Returns [(None, None)] when the API exposes no role concept, which degrades
    gracefully to the previous single-credential behaviour."""
    if not register_endpoint:
        return [(None, None)]
    enums = _role_field_enums(register_endpoint)
    if not enums:
        print("AUTH HANDSHAKE: no role-like enum in register schema; single credential")
        return [(None, None)]
    field, values = enums[0]
    ordered = sorted(
        values,
        key=lambda v: (0 if any(h in str(v).upper() for h in _PRIVILEGED_HINTS) else 1, str(v)),
    )
    candidates = [(field, v) for v in ordered][:_MAX_CREDENTIAL_CANDIDATES]
    print(f"AUTH HANDSHAKE: role candidates for {field} = {[v for _, v in candidates]}")
    return candidates


def _register_payload_for_role(register_endpoint, role_field=None, role_value=None):
    """Build a register payload with a forced role value and a unique email."""
    payload = _register_payload_with_role(register_endpoint)
    if not payload:
        try:
            payload = dict(build_request_from_openapi(register_endpoint, variant=0).get("payload") or {})
        except Exception:
            return None
    payload = dict(payload)
    if role_field and role_value is not None:
        payload[role_field] = role_value
    for key in list(payload.keys()):
        if "email" in str(key).lower():
            payload[key] = _unique_register_email()
            break
    return payload


def _probe_endpoint_for_credentials():
    """A generic write-ish endpoint used to test whether a credential passes the
    authorization layer (prefers POST to a collection path)."""
    catalog = global_vars_funcs_configs.catalog.api_swagger_map
    for want_collection in (True, False):
        for endpoint in catalog:
            try:
                method, path = endpoint.split(" ", 1)
            except ValueError:
                continue
            if method not in ("POST", "PUT", "PATCH", "DELETE"):
                continue
            low = path.lower()
            if any(k in low for k in ("auth", "login", "register", "signup", "token")):
                continue
            if want_collection and "{" in path:
                continue
            return endpoint
    return None


def _probe_credential(token, probe_endpoint):
    """Send one request with an explicit token and score how far it got:
    2xx=3, business 4xx=2 (passed authorization), 401/403=0, other=1.
    token=None sends no Authorization header at all (used by the override
    detector to establish the anonymous baseline)."""
    if not probe_endpoint:
        return None, None
    try:
        args = build_request_from_openapi(probe_endpoint, variant=0)
        headers = dict(args.get("headers") or {})
        headers.pop("Authorization", None)
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        url = remove_duplicate_path_segment(
            global_vars_funcs_configs.CONFIG_BASE_URL, args.get("api", "")
        )
        resp = _send_http(
            args.get("method", "POST"), url, headers,
            args.get("params") or {}, args.get("payload"),
            args.get("payload_type") or "application/json", 10,
        )
        status = resp.status_code
    except Exception as exc:
        print(f"AUTH PROBE: {probe_endpoint} failed: {exc}")
        return None, None
    if 200 <= status < 300:
        score = 3
    elif status in (401, 403):
        score = 0
    elif 400 <= status < 500:
        score = 2
    else:
        score = 1
    return score, status


def _detect_credential_override(probe_endpoint, good_token):
    """Detect whether something between us and the API replaces the credentials
    we send (e.g. a benchmark-side auth addon that injects its own account).

    Purely behavioural, so it generalises to unknown APIs: compare how the same
    endpoint answers (a) a deliberately invalid token, (b) no credentials, and
    (c) our real token.  If the invalid token behaves like the real one, our
    credentials are being substituted.

    Returns True (overridden) / False (we control auth) / None (inconclusive).
    """
    if not probe_endpoint or not good_token:
        return None
    _, bad_status = _probe_credential("not-a-real-token", probe_endpoint)
    _, anon_status = _probe_credential(None, probe_endpoint)
    _, good_status = _probe_credential(good_token, probe_endpoint)
    if bad_status is None or anon_status is None or good_status is None:
        return None
    if bad_status == good_status and bad_status != anon_status:
        return True
    if bad_status == anon_status and bad_status != good_status:
        return False
    return None


def _handshake_once(register, login, payload):
    """Register (when available) + login once; return the obtained token."""
    role_payload = payload
    if register and payload:
        try:
            req = build_request_from_openapi(register, variant=0)
            req["payload"] = dict(payload)
            result = do_request(**req)
            status = result.get("status_code")
            print(f"AUTH HANDSHAKE: register request sent to {register} -> {status}")
            if status == 409:
                print("AUTH HANDSHAKE: register returned 409 (user exists); reusing account for login.")
        except Exception as exc:
            print(f"AUTH HANDSHAKE: register failed: {exc}")
    req = build_request_from_openapi(login, variant=0)
    try:
        login_schema = (
            global_vars_funcs_configs.catalog.api_swagger_map.get(login, {})
            .get("requestBody", {}).get("content", {})
            .get("application/json", {}).get("schema", {})
        )
        login_schema = _resolve_ref(login_schema)
        login_props = login_schema.get("properties", {}) or {}
        if role_payload:
            for key in ("email", "password"):
                if key in login_props and key in role_payload:
                    req["payload"][key] = role_payload[key]
    except Exception as exc:
        print(f"AUTH HANDSHAKE: login payload alignment skipped: {exc}")
    result = do_request(**req)
    print(f"AUTH HANDSHAKE: login request sent to {login} -> {result.get('status_code')}")
    data = result.get("response_data")
    return _extract_auth_token(data), _extract_refresh_token(data)


def perform_auth_handshake(force: bool = False):
    """Register + login to obtain a token; store it for every later request.

    With force=True the handshake re-runs even if it already attempted once
    (used when a stored token stops being accepted, e.g. expiry or a fresh
    server instance)."""
    global _in_handshake
    if auth_state["attempted"] and not force:
        return auth_state.get("token") is not None
    auth_state["attempted"] = True
    if not _auth_enabled():
        print("AUTH HANDSHAKE: disabled by REST_LEAGUE_AUTH_HANDSHAKE")
        return False
    register, login = _find_auth_endpoints()
    auth_state["register_endpoint"], auth_state["login_endpoint"] = register, login
    if not login:
        print("AUTH HANDSHAKE: no login endpoint in OpenAPI; skipping authentication")
        return False

    print(f"AUTH HANDSHAKE: register={register} login={login}")
    _in_handshake = True
    try:
        candidates = _role_candidates(register)
        probe_target = _probe_endpoint_for_credentials()
        if probe_target and _CREDENTIAL_PROBE_ENABLED:
            print(f"AUTH CREDENTIALS: write probe target = {probe_target}")
        credentials = []
        for field, value in candidates:
            payload = _register_payload_for_role(register, field, value) if register else None
            token, refresh = _handshake_once(register, login, payload)
            if not token:
                print(f"AUTH CREDENTIALS: candidate {field}={value} produced no token")
                continue
            score, status = (None, None)
            if _CREDENTIAL_PROBE_ENABLED:
                score, status = _probe_credential(token, probe_target)
            credentials.append({
                "role_field": field,
                "role_value": value,
                "token": token,
                "refresh_token": refresh,
                "probe_score": score,
                "probe_status": status,
                "validity": _parse_jwt_validity(token),
                "payload": dict(payload) if payload else None,
            })
            label = f"{field}={value}" if field else "anonymous"
            print(f"AUTH CREDENTIALS: {label} -> token {len(token)} chars, probe_status={status}, score={score}")
        if not credentials:
            print("AUTH HANDSHAKE: login did not return a usable token")
            return False
        # Highest probe score wins: a credential that passes the authorization
        # layer on a write endpoint is the one worth reusing everywhere.
        credentials.sort(key=lambda c: -(c.get("probe_score") or 0))
        best = credentials[0]
        auth_state["credentials"] = credentials
        auth_state["token"] = best["token"]
        auth_state["refresh_token"] = best.get("refresh_token")
        auth_state["token_acquired_at"] = time.time()
        auth_state["token_validity_seconds"] = best["validity"]
        if best.get("payload"):
            auth_state["register_payload"] = dict(best["payload"])
            auth_state["email"] = best["payload"].get("email")
            auth_state["password"] = best["payload"].get("password")
        # Prefer the scheme declared by the OpenAPI security scheme.
        try:
            schemes = global_vars_funcs_configs.catalog.api_swagger_map.get(login, {}).get("securityDefinitions", {})
            for name, s in schemes.items():
                if isinstance(s, dict) and s.get("type") == "http" and s.get("scheme") == "bearer":
                    auth_state["token_type"] = "Bearer"
                    break
        except Exception:
            pass
        best_label = f"{best['role_field']}={best['role_value']}" if best["role_field"] else "anonymous"
        print(
            f"AUTH HANDSHAKE: selected credential {best_label} "
            f"(probe_status={best['probe_status']}, score={best['probe_score']}, {len(best['token'])} chars)"
        )
        if best.get("probe_score") == 0:
            print(
                "AUTH CREDENTIALS: every candidate was denied on the write probe; this API "
                "most likely does not grant write access to registered users."
            )
        if best["validity"]:
            print(
                f"AUTH HANDSHAKE: JWT validity {best['validity']}s "
                f"(refresh planned at {int(best['validity'] * 0.6)}s)"
            )
        # Detect a platform-side credential substitution (behavioural, generic).
        try:
            override = _detect_credential_override(probe_target, best["token"])
            auth_state["credential_override"] = override
            if override is True:
                print(
                    "AUTH CREDENTIALS: OVERRIDE DETECTED - an invalid token behaves like our "
                    "valid one, so something between the tool and the API is replacing our "
                    "credentials. Our role/token choice cannot affect authorization here; "
                    "stop spending budget on auth and focus on readable coverage + faults."
                )
            elif override is False:
                print("AUTH CREDENTIALS: our credentials are honoured by the API (no override)")
            else:
                print("AUTH CREDENTIALS: override detection inconclusive")
        except Exception as exc:
            print(f"AUTH CREDENTIALS: override detection failed: {exc}")
        return True
    except Exception as exc:
        print(f"AUTH HANDSHAKE: failed: {exc}")
        return False
    finally:
        _in_handshake = False


def _send_http(method, url, headers, params, payload, payload_type, timeout):
    """Single HTTP send used both by do_request and the 401 retry."""
    if payload is not None:
        if payload_type == "application/json":
            return requests.request(method, url, headers=headers, params=params, json=payload,
                                    timeout=timeout, verify=False, proxies={"http": None, "https": None})
        if payload_type == "application/x-www-form-urlencoded":
            return requests.request(method, url, headers=headers, params=params, data=payload,
                                    timeout=timeout, verify=False, proxies={"http": None, "https": None})
        if payload_type == "multipart/form-data":
            return requests.request(method, url, headers=headers, params=params,
                                    files=_prepare_multipart_files(payload),
                                    timeout=timeout, verify=False, proxies={"http": None, "https": None})
        return requests.request(method, url, headers=headers, params=params, data=payload,
                                timeout=timeout, verify=False, proxies={"http": None, "https": None})
    return requests.request(method, url, headers=headers, params=params,
                            timeout=timeout, verify=False, proxies={"http": None, "https": None})


def _env_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def is_rest_league_mode() -> bool:
    value = os.getenv("REST_LEAGUE_MODE", "false")
    try:
        return bool(strtobool(value))
    except ValueError:
        return False


def _schema_example(schema: dict, name: str = "value", variant: int = 0, required: bool = False):
    if hirest_equivalence.enabled():
        return hirest_equivalence.choose_category(name, schema, required=required, variant=variant).value

    if not isinstance(schema, dict):
        return f"{name}_{variant}"

    if "enum" in schema and schema["enum"]:
        return schema["enum"][variant % len(schema["enum"])]

    if "example" in schema and schema["example"] not in (None, ""):
        example = schema["example"]
        if isinstance(example, str) and example.lower() == "abc":
            pass
        else:
            return example

    schema_type = schema.get("type", "string")
    fmt = schema.get("format", "")
    lower_name = name.lower()
    variant_slot = variant % 5

    if schema_type == "integer":
        values = [int(schema.get("minimum", 1)), 0, -1, 2147483647, -2147483648]
        return values[variant_slot]
    if schema_type == "number":
        values = [float(schema.get("minimum", 1.25)), 0.0, -1.0, 1.0e6, -1.0e6]
        return values[variant_slot]
    if schema_type == "boolean":
        return variant % 2 == 0
    if schema_type == "array":
        return [_schema_example(schema.get("items", {}), name, variant, required=True)]
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return {
            prop: _schema_example(prop_schema, prop, variant, required=prop in required)
            for prop, prop_schema in properties.items()
            if prop in required or variant == 0
        }

    if "email" in lower_name:
        return f"user{variant}@example.com"
    if "uuid" in lower_name or fmt == "uuid":
        return f"123e4567-e89b-12d3-a456-42661417{variant:04d}"
    if "datetime" in lower_name or fmt == "date-time":
        return ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00", "2024-01-01", "2024-01-01T00:00:00.000"][variant % 4]
    if "date" in lower_name or fmt == "date":
        return "2024-01-01"
    if "time" in lower_name or fmt == "time":
        return ["00:00:00", "12:00:00", "23:59:59"][variant % 3]
    if lower_name in {"day", "dayname"}:
        return "Monday"
    if lower_name in {"month", "monthname"}:
        return "January"
    if lower_name in {"op", "operator", "operation"}:
        return ["plus", "minus", "multiply", "divide", "unknown"][variant_slot]
    if "file" in lower_name:
        return [f"file{variant}.txt", "README", "archive.tar.gz", "file.", "null"][variant_slot]
    if "dir" in lower_name:
        return [f"tmp{variant}", "var", "a.b", "root", "null"][variant_slot]
    if lower_name in {"pat", "pattern", "regex"}:
        return ["abc.*", "^abc[0-9]+$", ".*", "[a-z]+", "++"][variant_slot]
    if lower_name in {"txt", "text", "word", "word1", "word2", "word3", "s"}:
        return [f"abc{variant}", "12345", "hello", "null", "A" * 128][variant_slot]
    if "url" in lower_name or fmt == "uri":
        return "http://example.com"
    if "phone" in lower_name:
        return "1234567890"
    if "password" in lower_name:
        return f"Passw0rd{variant}"
    if "name" in lower_name or "title" in lower_name:
        return [f"{name}_{variant}", "Mr", "Dr", "null", "A" * 64][variant_slot]

    return [f"{name}_{variant}", "12345", "null", "A" * 64, "special_%21"][variant_slot]


def _schema_valid_value(schema: dict, name: str = "value", variant: int = 0, required: bool = False):
    if not isinstance(schema, dict):
        return f"{name}_{variant}"
    if hirest_equivalence.enabled():
        categories = hirest_equivalence.categories_for_parameter(name, schema, required=required, variant=variant)
        present_valid = [category for category in categories if category.valid and category.value != hirest_equivalence.MISS_KEY]
        non_empty_valid = [category for category in present_valid if category.category_id != "array_empty"]
        if non_empty_valid:
            return non_empty_valid[variant % len(non_empty_valid)].value
        if present_valid:
            return present_valid[variant % len(present_valid)].value
    return _schema_example(schema, name, variant=variant, required=required)


def _schema_valid_payload(schema: dict, name: str = "body", variant: int = 0, required: bool = True):
    if not isinstance(schema, dict):
        return f"{name}_{variant}"
    schema_type = schema.get("type")
    if schema_type is None and "properties" in schema:
        schema_type = "object"
    if schema_type is None and "items" in schema:
        schema_type = "array"
    if schema_type == "object":
        properties = schema.get("properties", {})
        required_props = set(schema.get("required", []))
        include_all = required or not required_props
        value = {}
        for prop, prop_schema in properties.items():
            if include_all or prop in required_props:
                prop_value = _schema_valid_payload(prop_schema, prop, variant=variant, required=prop in required_props or include_all)
                if prop_value != hirest_equivalence.MISS_KEY:
                    value[prop] = prop_value
        return value
    if schema_type == "array":
        return [_schema_valid_payload(schema.get("items", {}), name, variant=variant, required=True)]
    return _schema_valid_value(schema, name, variant=variant, required=required)


def _invalid_value_for_schema(schema: dict, name: str, variant: int):
    if not isinstance(schema, dict):
        return None
    if hirest_equivalence.enabled():
        categories = hirest_equivalence.categories_for_parameter(name, schema, required=True, variant=variant)
        invalid = [category for category in categories if not category.valid and category.value != hirest_equivalence.MISS_KEY]
        if invalid:
            return invalid[variant % len(invalid)].value
    schema_type = schema.get("type", "string")
    if schema_type == "integer":
        return f"{name}_not_int"
    if schema_type == "number":
        return f"{name}_not_number"
    if schema_type == "boolean":
        return "not_boolean"
    if schema_type == "array":
        return "not_array"
    if schema_type == "object":
        return "not_object"
    return _schema_example({"type": "string"}, name, variant=4, required=True)


def _collect_mutation_targets(schema: dict, path=None):
    path = list(path or [])
    if not isinstance(schema, dict):
        return []
    schema_type = schema.get("type")
    if schema_type is None and "properties" in schema:
        schema_type = "object"
    if schema_type is None and "items" in schema:
        schema_type = "array"
    targets = []
    if schema_type == "object":
        for prop, prop_schema in schema.get("properties", {}).items():
            targets.extend(_collect_mutation_targets(prop_schema, path + [prop]))
    elif schema_type == "array":
        targets.append((path, schema))
        targets.extend(_collect_mutation_targets(schema.get("items", {}), path + [0]))
    else:
        targets.append((path, schema))
    return targets


def _set_nested_value(payload, path, value):
    if not path:
        return value
    current = payload
    for part in path[:-1]:
        if isinstance(part, int):
            if not isinstance(current, list) or not current:
                return payload
            current = current[0]
        else:
            if not isinstance(current, dict) or part not in current:
                return payload
            current = current[part]
    last = path[-1]
    if isinstance(last, int):
        if isinstance(current, list) and current:
            current[0] = value
    elif isinstance(current, dict):
        current[last] = value
    return payload


def _delete_nested_key(payload, path):
    if not path:
        return hirest_equivalence.MISS_KEY
    current = payload
    for part in path[:-1]:
        if isinstance(part, int):
            if not isinstance(current, list) or not current:
                return payload
            current = current[0]
        else:
            if not isinstance(current, dict) or part not in current:
                return payload
            current = current[part]
    last = path[-1]
    if isinstance(current, dict) and isinstance(last, str):
        current.pop(last, None)
    return payload


def _apply_field_level_mutation(payload, schema: dict, variant: int, metadata: dict):
    if not _env_flag("REST_LEAGUE_FIELD_MUTATION", "true") or variant < 2:
        return payload
    targets = _collect_mutation_targets(schema)
    if not targets:
        return payload
    path, target_schema = targets[variant % len(targets)]
    field_name = str(path[-1]) if path else "body"
    if variant % 5 == 2:
        payload = _set_nested_value(payload, path, _invalid_value_for_schema(target_schema, field_name, variant))
        mutation_id = "field_invalid_value"
    elif variant % 5 == 3:
        payload = _delete_nested_key(payload, path)
        if payload == hirest_equivalence.MISS_KEY:
            payload = {}
        mutation_id = "field_missing"
    else:
        payload = _set_nested_value(payload, path, _schema_valid_payload(target_schema, field_name, variant=variant, required=True))
        mutation_id = "field_boundary_valid"
    meta_path = "body" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in path)
    metadata[meta_path] = {
        "location": "body",
        "category_id": mutation_id,
        "category_name": "field-level mutation",
        "valid": mutation_id == "field_boundary_valid",
        "reason": "Field-level mutation keeps the surrounding request valid while mutating one parameter.",
    }
    return payload


def _collect_schema_category_metadata(schema: dict, name: str, variant: int, required: bool, path: str) -> dict:
    if not hirest_equivalence.enabled():
        return {}
    if not isinstance(schema, dict):
        schema = {}

    category = hirest_equivalence.choose_category(name, schema, required=required, variant=variant)
    metadata = {
        path: {
            "location": "body",
            "category_id": category.category_id,
            "category_name": category.name,
            "valid": category.valid,
            "reason": category.reason,
        }
    }
    if category.value == hirest_equivalence.MISS_KEY:
        return metadata

    schema_type = schema.get("type")
    if schema_type is None and "properties" in schema:
        schema_type = "object"
    if schema_type is None and "items" in schema:
        schema_type = "array"

    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required_props = set(schema.get("required", []))
        include_all = category.category_id == "object_valid_required"
        for prop, prop_schema in properties.items():
            metadata.update(
                _collect_schema_category_metadata(
                    prop_schema,
                    prop,
                    variant,
                    prop in required_props or include_all,
                    f"{path}.{prop}",
                )
            )
    elif schema_type == "array":
        metadata.update(
            _collect_schema_category_metadata(
                schema.get("items", {}),
                name,
                variant,
                True,
                f"{path}[]",
            )
        )
    return metadata


def _request_body_example(operation: dict, variant: int = 0):
    request_body = operation.get("requestBody") if isinstance(operation, dict) else None
    if not isinstance(request_body, dict):
        return None, "application/json", {}

    content = request_body.get("content", {})
    if not isinstance(content, dict) or not content:
        return None, "application/json", {}

    if "application/json" in content:
        payload_type = "application/json"
    else:
        payload_type = next(iter(content.keys()))

    schema = content.get(payload_type, {}).get("schema", {})
    metadata = _collect_schema_category_metadata(schema, "body", variant, True, "body")
    if _env_flag("REST_LEAGUE_FIELD_MUTATION", "true"):
        metadata["body"] = {
            "location": "body",
            "category_id": "object_valid_required",
            "category_name": "object with valid fields",
            "valid": True,
            "reason": "Valid request body used as the base for field-level mutation.",
        }
        payload = _schema_valid_payload(schema, "body", variant=variant, required=True)
        payload = _inject_resource_values_into_payload(payload, schema, variant)
        payload = _apply_field_level_mutation(payload, schema, variant, metadata)
    else:
        payload = _schema_example(schema, "body", variant, required=True)
    if payload == hirest_equivalence.MISS_KEY:
        payload = {}
    if not _env_flag("REST_LEAGUE_FIELD_MUTATION", "true"):
        payload = _inject_resource_values_into_payload(payload, schema, variant)
    return payload, payload_type, metadata


def _parameter_category_metadata(name: str, schema: dict, required: bool, variant: int, location: str) -> dict:
    if not hirest_equivalence.enabled():
        return {}
    category = hirest_equivalence.choose_category(name, schema, required=required, variant=variant)
    return {
        "location": location,
        "category_id": category.category_id,
        "category_name": category.name,
        "valid": category.valid,
        "reason": category.reason,
    }


def _schema_with_parameter_context(param: dict) -> dict:
    schema = dict(param.get("schema", {}) or {})
    description = param.get("description")
    if description and "description" not in schema:
        schema["description"] = description
    return schema


def _prepare_multipart_files(payload):
    if not isinstance(payload, dict):
        return payload
    files = {}
    for key, value in payload.items():
        if isinstance(value, tuple):
            files[key] = value
        elif isinstance(value, list) and len(value) in {2, 3}:
            files[key] = tuple(value)
        else:
            files[key] = (None, "" if value is None else str(value))
    return files


def _scalar_resource_value(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and value.strip() and len(value) <= 256:
        return value
    return None


def _is_int_like(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, str):
        return bool(re.fullmatch(r"-?\d+", value.strip()))
    return False


def _schema_type(schema: dict) -> str:
    if not isinstance(schema, dict):
        return "string"
    if schema.get("type"):
        return schema.get("type")
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return "string"


def _coerce_resource_value_for_schema(value, schema: dict, name: str):
    if value is None:
        return None
    schema_type = _schema_type(schema)
    normalized_name = _normalize_resource_key(name)
    if schema_type == "integer":
        if not _is_int_like(value):
            return None
        return int(value)
    if schema_type == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return value


def _normalize_resource_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _resource_candidate_keys(name: str) -> list[str]:
    key = _normalize_resource_key(name)
    # A bare `id` has no entity type. Never use the global ID bucket to fill
    # arbitrary path/body parameters; path templates can infer a typed key.
    if key == "id":
        return []
    keys = [key]
    if key.endswith("id") and len(key) > 2:
        keys.append(key[:-2])
    if key in {"username", "user"}:
        keys.extend(["name", "login"])
    if key.endswith("name"):
        keys.append("name")
    return list(dict.fromkeys(k for k in keys if k))


def _put_resource_value(key: str, value, source_endpoint: str = None, state: str = "created"):
    """Store a resource value with provenance and state.

    resource_pool[key] -> [{"value": v, "source": endpoint, "state": state}, ...]
    The bare list-of-scalars format is preserved for backward compatibility, but
    new code prefers _put_resource_entry.
    """
    scalar = _scalar_resource_value(value)
    if scalar is None:
        return
    key = _normalize_resource_key(key)
    entries = resource_pool.setdefault(key, [])
    if not isinstance(entries, list):
        entries = []
        resource_pool[key] = entries
    for entry in entries:
        if isinstance(entry, dict) and entry.get("value") == scalar:
            entry["state"] = state
            if source_endpoint:
                entry["source"] = source_endpoint
            return
        if not isinstance(entry, dict) and entry == scalar:
            # Upgrade legacy scalar entry
            idx = entries.index(entry)
            entries[idx] = {"value": scalar, "source": source_endpoint, "state": state}
            return
    entries.append({"value": scalar, "source": source_endpoint, "state": state})
    if len(entries) > 50:
        del entries[:-50]


def _put_resource_entry(key: str, value, source_endpoint: str = None, state: str = "created"):
    _put_resource_value(key, value, source_endpoint=source_endpoint, state=state)


def _get_resource_value(name: str, variant: int = 0):
    entries = _get_resource_entries(name)
    if entries:
        entry = entries[variant % len(entries)]
        return entry.get("value") if isinstance(entry, dict) else entry
    return None


def _get_resource_entries(name: str) -> list:
    for key in _resource_candidate_keys(name):
        entries = [entry for entry in resource_pool.get(key, [])
                   if not (isinstance(entry, dict) and entry.get("state") == "deleted")]
        if entries:
            return entries
    return []


def _resource_latest_state(name: str):
    """Return ('state', value) of the most recent usable entry for name."""
    for key in _resource_candidate_keys(name):
        entries = resource_pool.get(key)
        if not entries:
            continue
        if isinstance(entries[0], dict):
            # Most recently created first
            return entries[0].get("state"), entries[0].get("value")
        return "created", entries[0]
    return None, None


def mark_resource_invalid(key: str, value):
    """Mark a resource as deleted/invalid after a destructive 2xx response."""
    key = _normalize_resource_key(key)
    entries = resource_pool.get(key)
    if not entries:
        return
    for index, entry in enumerate(entries):
        entry_value = entry.get("value") if isinstance(entry, dict) else entry
        if entry_value == value:
            if isinstance(entry, dict):
                entry["state"] = "deleted"
            else:
                entries[index] = {"value": entry_value, "state": "deleted"}
            print(f"RESOURCE STATE: marked {key}={value} as deleted")
            return


def _singularize_resource_name(name: str) -> str:
    name = _normalize_resource_key(name)
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("s") and len(name) > 3:
        return name[:-1]
    return name


def _resource_value_for_parameter(name: str, path_template: str, variant: int = 0):
    # Prefer live (non-deleted) values.
    live = [
        entry.get("value") if isinstance(entry, dict) else entry
        for entry in _get_resource_entries(name)
        if not (isinstance(entry, dict) and entry.get("state") == "deleted")
    ]
    if live:
        return live[variant % len(live)]
    value = _get_resource_value(name, variant)
    if value is not None:
        return value

    placeholder = "{" + name + "}"
    parts = [part for part in path_template.strip("/").split("/") if part]
    for index, part in enumerate(parts):
        if part == placeholder or part == "{{" + name + "}}":
            if index > 0:
                resource_name = _singularize_resource_name(parts[index - 1])
                for key in (f"{resource_name}id", resource_name):
                    entries = resource_pool.get(key)
                    if entries:
                        if isinstance(entries[0], dict):
                            live = [e.get("value") for e in entries if not e.get("state") == "deleted"]
                            if live:
                                return live[variant % len(live)]
                            continue
                        return entries[variant % len(entries)]
    return None


def _endpoint_resource_names(api_endpoint: str, api_path: str) -> list[str]:
    text = f"{api_endpoint or ''} {api_path or ''}".lower()
    names = []
    for part in re.split(r"[^a-zA-Z0-9{}]+", text):
        part = part.strip("{}")
        if not part or part in {"api", "v1", "v2", "rest", "get", "post", "put", "patch", "delete"}:
            continue
        if part.endswith("ies"):
            names.append(part[:-3] + "y")
        elif part.endswith("s") and len(part) > 3:
            names.append(part[:-1])
        names.append(part)
    return list(dict.fromkeys(_normalize_resource_key(name) for name in names if name))


def _primary_endpoint_resource_name(api_endpoint: str = None, api_path: str = None) -> str:
    """Infer one likely resource type for an unqualified response `id`.

    Prefer the last collection/resource noun in the documented route. This is
    intentionally conservative and generic; explicit response keys remain the
    primary source of typed IDs.
    """
    endpoint_parts = str(api_endpoint or "").split(None, 1)
    path = endpoint_parts[1] if len(endpoint_parts) == 2 else ""
    if not path.startswith("/"):
        path = str(api_path or "").split("?", 1)[0]
    action_words = {
        "create", "delete", "update", "replace", "patch", "get", "list",
        "search", "find", "register", "login", "logout", "authenticate",
        "refresh", "token", "validate", "verify", "activate", "deactivate",
    }
    candidates = []
    for part in path.strip("/").split("/"):
        if not part or (part.startswith("{") and part.endswith("}")):
            continue
        normalized = _normalize_resource_key(part)
        if normalized in {"api", "rest", "v1", "v2", "v3", "v4"} or re.fullmatch(r"v\d+", normalized):
            continue
        if normalized in action_words:
            continue
        candidates.append(_singularize_resource_name(normalized))
    return candidates[-1] if candidates else ""


def update_resource_pool_from_response(response_data, api_endpoint: str = None, api_path: str = None):
    if isinstance(response_data, str):
        try:
            response_data = json.loads(response_data)
        except ValueError:
            return
    # The current test-case label can lag behind direct fallback requests.
    # Attribute response IDs to the URL that was actually sent.
    primary_resource = _primary_endpoint_resource_name(None, api_path) if api_path else _primary_endpoint_resource_name(api_endpoint)
    if isinstance(response_data, dict) and primary_resource:
        # Some create operations return the new ID as a bare UUID inside a
        # generic response wrapper: {"response": "<uuid>"}. Preserve its
        # resource type so a later /resources/{id} request can reuse it.
        created_id = response_data.get("response")
        if isinstance(created_id, str) and re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", created_id
        ):
            _put_resource_value(f"{primary_resource}id", created_id,
                                source_endpoint=api_endpoint)

    def visit(value, parent_key: str = ""):
        if isinstance(value, dict):
            for key, child in value.items():
                scalar = _scalar_resource_value(child)
                if scalar is not None:
                    _put_resource_value(key, scalar, source_endpoint=api_endpoint)
                    normalized_key = _normalize_resource_key(key)
                    if normalized_key == "id":
                        if primary_resource:
                            _put_resource_value(f"{primary_resource}id", scalar, source_endpoint=api_endpoint)
                    if parent_key:
                        _put_resource_value(f"{parent_key}{key}", scalar, source_endpoint=api_endpoint)
                visit(child, _normalize_resource_key(key))
        elif isinstance(value, list):
            for child in value[:10]:
                visit(child, parent_key)

    visit(response_data)

    # Explicit names such as cluster_id were already recorded above. Do not
    # synthesize every response ID under every noun in a nested endpoint path.


def update_resource_pool_from_response_headers(headers, api_endpoint: str = None, api_path: str = None):
    if not _env_flag("REST_LEAGUE_RESOURCE_POOL", "true"):
        return
    if not headers:
        return
    location = None
    try:
        location = headers.get("Location") or headers.get("location")
    except AttributeError:
        if isinstance(headers, dict):
            location = headers.get("Location") or headers.get("location")
    if not location:
        return

    parsed_path = urlparse(str(location)).path or str(location)
    parts = [part for part in parsed_path.strip("/").split("/") if part]
    if not parts:
        return

    primary_resource = _primary_endpoint_resource_name(api_endpoint, api_path)
    for index, part in enumerate(parts):
        # Location paths contain both route words and IDs. Only numeric-like
        # segments are reliable generic resource identifiers.
        if not _is_int_like(part):
            continue
        scalar = int(part)
        if scalar is None:
            continue
        previous = parts[index - 1] if index > 0 else ""
        next_part = parts[index + 1] if index + 1 < len(parts) else ""
        previous_key = _singularize_resource_name(previous)
        if previous_key and previous_key not in {"api", "rest"} and not re.fullmatch(r"v\d+", previous_key):
            _put_resource_value(previous_key, scalar)
            _put_resource_value(f"{previous_key}id", scalar)
        if next_part:
            next_key = _singularize_resource_name(next_part)
            if next_key:
                _put_resource_value(f"{next_key}parentid", scalar)
        if primary_resource:
            _put_resource_value(f"{primary_resource}id", scalar, source_endpoint=api_endpoint)


def _record_automatic_useful_items(data, api_endpoint: str = None, api_path: str = None):
    if not _env_flag("REST_LEAGUE_AUTO_USEFUL_ITEMS", "true"):
        return
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return

    endpoint_names = _endpoint_resource_names(api_endpoint or "", api_path or "")
    useful = {}
    interesting_names = {
        "id", "name", "username", "password", "email", "status", "type",
        "ownerid", "petid", "visitid", "userid", "orderid",
    }

    def put(name: str, value):
        scalar = _scalar_resource_value(value)
        if scalar is None:
            return
        normalized = _normalize_resource_key(name)
        if not normalized:
            return
        if (
            normalized not in interesting_names
            and not normalized.endswith("id")
            and "name" not in normalized
            and normalized not in {"status", "type"}
        ):
            return
        useful[normalized] = {
            "value": scalar,
            "description": f"Automatically extracted from successful {api_endpoint or api_path or 'request'}."
        }
        if normalized == "id":
            for endpoint_name in endpoint_names:
                useful[f"{endpoint_name}id"] = {
                    "value": scalar,
                    "description": f"ID associated with {endpoint_name} from successful {api_endpoint or api_path or 'request'}."
                }

    def visit(value, parent_key: str = ""):
        if isinstance(value, dict):
            for key, child in value.items():
                current_key = f"{parent_key}{key}" if parent_key else key
                put(key, child)
                put(current_key, child)
                visit(child, _normalize_resource_key(key))
        elif isinstance(value, list):
            for child in value[:10]:
                visit(child, parent_key)

    visit(data)
    if useful:
        try:
            test_scenario.record_useful_items(useful)
            print(f"AUTO USEFUL ITEMS: recorded {sorted(useful.keys())[:20]}")
        except Exception as exc:
            print(f"AUTO USEFUL ITEMS: failed to record useful items: {exc}")


def _extract_refresh_token(data):
    """Find a refresh token inside a response body."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return None

    def walk(value, depth=0):
        if depth > 6:
            return None
        if isinstance(value, dict):
            for k, v in value.items():
                if "refresh" in str(k).lower() and isinstance(v, str) and len(v) >= 10:
                    return v
            for v in value.values():
                found = walk(v, depth + 1)
                if found:
                    return found
        elif isinstance(value, list):
            for v in value[:5]:
                found = walk(v, depth + 1)
                if found:
                    return found
        return None

    return walk(data)


def _token_for_body_field(low_name):
    """The credential that belongs in a token-bearing request-body field."""
    if "refresh" in low_name:
        return auth_state.get("refresh_token") or auth_state.get("token")
    return auth_state.get("token")


def _inject_resource_values_into_payload(payload, schema: dict, variant: int = 0):
    if not _env_flag("REST_LEAGUE_RESOURCE_POOL", "true"):
        return payload
    if not isinstance(schema, dict):
        return payload
    schema = _resolve_ref(schema)
    schema_type = schema.get("type")
    if schema_type is None and "properties" in schema:
        schema_type = "object"
    if schema_type is None and "items" in schema:
        schema_type = "array"
    if schema_type == "object" and isinstance(payload, dict):
        for prop, prop_schema in schema.get("properties", {}).items():
            if prop in payload:
                low_prop = str(prop).lower()
                # Token-bearing body fields (logout / refresh-token endpoints)
                # must carry a REAL credential; the schema example only yields a
                # placeholder like "accessToken_0", which the server rejects with
                # "Invalid compact JWT string".
                if "token" in low_prop:
                    real = _token_for_body_field(low_prop)
                    if real:
                        payload[prop] = real
                        continue
                pooled = _get_resource_value(prop, variant)
                normalized_prop = _normalize_resource_key(prop)
                should_reuse = (
                    (normalized_prop.endswith("id") and normalized_prop != "id")
                    or normalized_prop in {"username"}
                )
                coerced = _coerce_resource_value_for_schema(pooled, prop_schema, prop)
                if coerced is not None and should_reuse:
                    payload[prop] = coerced
                else:
                    payload[prop] = _inject_resource_values_into_payload(payload[prop], prop_schema, variant)
    elif schema_type == "array" and isinstance(payload, list):
        for index, item in enumerate(payload):
            payload[index] = _inject_resource_values_into_payload(item, schema.get("items", {}), variant)
    return payload


def _flatten_query_object(prefix, value, out=None):
    """Flatten a nested value into Spring-style dotted query keys, e.g.
    ('request', {'pagination': {'pageNumber': 1}})
      -> {'request.pagination.pageNumber': 1}
    This is the binding style Spring MVC uses for a @ModelAttribute query bean,
    which a JSON string does not satisfy."""
    if out is None:
        out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten_query_object(f"{prefix}.{key}", child, out)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _flatten_query_object(f"{prefix}[{index}]", child, out)
    else:
        out[prefix] = value
    return out


def _reconstruct_body_from_params(params):
    """Rebuild a JSON body from the query parameters that were just sent.

    Used when a service documents a compound parameter as a query parameter but
    its implementation actually requires a request body (the server answers
    "Required request body is missing"): the same data is re-sent in the body.
    Handles both 'request={"json"}' and 'request.pagination.pageNumber=1'."""
    body = {}
    for key, value in (params or {}).items():
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                body.update(parsed)
                continue
        if "." in str(key):
            cursor = body
            parts = str(key).split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
                if not isinstance(cursor, dict):
                    break
            else:
                cursor[parts[-1]] = value
                continue
        body[key] = value
    return body


def build_request_from_openapi(api_endpoint: str, variant: int = 0) -> dict:
    normalized_endpoint = test_scenario.normalize_api_endpoint(api_endpoint)
    if " " not in normalized_endpoint:
        raise ValueError(f"Invalid API endpoint: {api_endpoint}")

    method, path_template = normalized_endpoint.split(" ", 1)
    operation = global_vars_funcs_configs.catalog.api_swagger_map.get(normalized_endpoint)
    if not operation:
        raise ValueError(f"No Swagger info for API endpoint: {api_endpoint}")

    params = {}
    headers = {}
    parameter_categories = {}
    rendered_path = path_template
    # Some services document a compound parameter as a query parameter but the
    # implementation actually requires it in a GET request body ("Required
    # request body is missing").  Collect such a value so it can be re-sent as
    # the body on one of the encodings.
    query_body_value = None

    for param in operation.get("parameters", []):
        name = param.get("name")
        if not name:
            continue
        schema = _schema_with_parameter_context(param)
        location = param.get("in")
        required = bool(param.get("required", False)) or location == "path"
        category = hirest_equivalence.choose_category(name, schema, required=required, variant=variant) if hirest_equivalence.enabled() else None
        value = category.value if category else _schema_example(schema, name, variant, required=required)
        if _env_flag("REST_LEAGUE_RESOURCE_POOL", "true") and location == "path":
            pooled_value = _resource_value_for_parameter(name, path_template, variant)
        else:
            pooled_value = _get_resource_value(name, variant) if _env_flag("REST_LEAGUE_RESOURCE_POOL", "true") else None
        pooled_value = _coerce_resource_value_for_schema(pooled_value, schema, name)
        if pooled_value is not None and (required or variant % 5 not in {2, 4}):
            value = pooled_value
            category = None
            parameter_categories[name] = {
                "location": location,
                "category_id": "resource_pool_value",
                "category_name": "resource value reused from earlier successful request",
                "valid": True,
                "reason": "Resource-pool value selected to improve stateful operation coverage.",
            }
        elif category:
            parameter_categories[name] = {
                "location": location,
                "category_id": category.category_id,
                "category_name": category.name,
                "valid": category.valid,
                "reason": category.reason,
            }
        location = param.get("in")
        if location == "path":
            if value == hirest_equivalence.MISS_KEY:
                value = f"missing_{name}"
            encoded = quote(str(value), safe="")
            rendered_path = rendered_path.replace("{" + name + "}", encoded)
            rendered_path = rendered_path.replace("{{" + name + "}}", encoded)
        elif location == "query":
            if value != hirest_equivalence.MISS_KEY:
                # Compound query parameters (object schemas, e.g. a paging request
                # bean) are bound differently by different frameworks, so cycle
                # through the three common encodings by variant:
                #   mod 0 -> JSON string:        request={"pagination":{...}}
                #   mod 1 -> prefixed dotted:    request.pagination.pageNumber=1
                #   mod 2 -> bare dotted:        pagination.pageNumber=1
                #            (Spring @ModelAttribute binds the bean from bare keys)
                resolved = _resolve_ref(schema) if isinstance(schema, dict) else schema
                is_object = isinstance(resolved, dict) and (
                    resolved.get("type") == "object" or resolved.get("properties")
                )
                if is_object and isinstance(value, (dict, list)):
                    mode = variant % 3
                    if mode == 2 and method == "GET":
                        # Documented as a query param, but this API wants it as
                        # the GET request body -> send the object itself.
                        query_body_value = copy.deepcopy(value)
                        parameter_categories[name] = {
                            "location": "body",
                            "category_id": "compound_param_as_get_body",
                            "category_name": "compound parameter re-sent as GET body",
                            "valid": True,
                            "reason": "OpenAPI documents this as a query parameter but the "
                                      "service requires a request body on GET.",
                        }
                    elif mode == 1:
                        params.update(_flatten_query_object(name, value))
                    elif mode == 2:
                        flat = _flatten_query_object("", value)
                        params.update({k.lstrip("."): v for k, v in flat.items()})
                    else:
                        params[name] = json.dumps(value, separators=(",", ":"))
                else:
                    params[name] = value
        elif location == "header":
            if value != hirest_equivalence.MISS_KEY:
                headers[name] = str(value)

    payload, payload_type, body_categories = _request_body_example(operation, variant)
    if payload is None:
        payload = {}
    if _env_flag("REST_LEAGUE_RESOURCE_POOL", "true") and payload_type == "application/json":
        body_schema = operation.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
        payload = _inject_resource_values_into_payload(payload, body_schema, variant)
    if query_body_value is not None and not payload:
        # Re-send the compound parameter as the request body (see above).
        payload = query_body_value
        payload_type = "application/json"
    parameter_categories.update(body_categories)

    return {
        "base_url": global_vars_funcs_configs.CONFIG_BASE_URL,
        "method": method,
        "api": rendered_path,
        "headers": headers,
        "params": params,
        "payload": payload,
        "payload_type": payload_type,
        "parameter_categories": parameter_categories,
    }


def _find_create_endpoint_for(param_name: str, target_endpoint: str):
    """Find a POST collection endpoint that likely creates the resource type of param_name.

    Example: param_name='cluster_id', target='POST /v3/clusters/{cluster_id}/acls'
    -> returns 'POST /v3/clusters'.
    """
    normalized = test_scenario.normalize_api_endpoint(param_name)
    name = _singularize_resource_name(normalized)
    if name.endswith("id"):
        name = name[:-2]
    if not name:
        return None
    target_low = target_endpoint.lower()
    for endpoint, op in global_vars_funcs_configs.catalog.api_swagger_map.items():
        try:
            method, path = endpoint.split(" ", 1)
        except ValueError:
            continue
        if method != "POST" or "{segment" in path or not path.strip("/"):
            continue
        path_low = path.lower()
        tail = path_low.rstrip("/").split("/")[-1]
        if tail in {name, name + "s", name + "es"} or _singularize_resource_name(tail) == name:
            # Avoid endpoints that are mutations of other resources.
            if "/" + name in target_low or (name and f"/{name}s" in target_low):
                pass
            return endpoint
    return None


def _ensure_resource_for_path_params(api_endpoint: str):
    """If a path parameter has no pool value, first create/list the parent resource.

    Returns True if the pool is now populated for at least one missing path param.
    """
    if not _env_flag("REST_LEAGUE_RESOURCE_POOL", "true"):
        return False
    if not _env_flag("REST_LEAGUE_AUTO_CREATE_RESOURCES", "true"):
        return False
    normalized = test_scenario.normalize_api_endpoint(api_endpoint)
    if " " not in normalized:
        return False
    method, path_template = normalized.split(" ", 1)
    path_params = re.findall(r"\{([^{}]+)\}", path_template)
    if not path_params:
        return False
    missing = [name for name in path_params if _get_resource_value(name) is None]
    if not missing:
        return False
    for name in missing:
        create_endpoint = _find_create_endpoint_for(name, normalized)
        if create_endpoint is None:
            continue
        print(f"RESOURCE AUTO-CREATE: preparing {name} via {create_endpoint} before {api_endpoint}")
        try:
            req = build_request_from_openapi(create_endpoint, variant=0)
            do_request(**req)
        except Exception as exc:
            print(f"RESOURCE AUTO-CREATE: failed {create_endpoint}: {exc}")
        if _get_resource_value(name) is not None:
            print(f"RESOURCE AUTO-CREATE: pool now has {name} after {create_endpoint}")
            return True
    return False


def do_openapi_request(api_endpoint: str, variant: int = 0) -> dict:
    _ensure_resource_for_path_params(api_endpoint)
    request_args = build_request_from_openapi(api_endpoint, variant=variant)
    # Any register call (including LLM-driven ones) must use the privileged
    # role account, otherwise the scenario creates a low-privilege user whose
    # token gets rejected with 403 on write operations.
    normalized = test_scenario.normalize_api_endpoint(api_endpoint)
    if normalized == auth_state.get("register_endpoint"):
        if auth_state.get("register_payload"):
            # Reuse the exact account the handshake registered (409-safe:
            # same email/password; the repeated register will 409 but login
            # with these credentials still returns the privileged token).
            request_args["payload"] = dict(auth_state["register_payload"])
        else:
            role_payload = _register_payload_with_role(api_endpoint)
            if role_payload:
                request_args["payload"] = role_payload
    return do_request(**request_args)


class DoRequestsRequestParams(BaseModel):
    base_url: Annotated[str, Field(description="REST API System Base URL")]
    method: Annotated[str, Field(description="HTTP Method")]
    api: Annotated[str, Field(description="REST API URL")]
    headers: Annotated[Optional[dict], Field(description="API Request Headers")] = Field(default_factory=dict)
    params: Annotated[Optional[dict], Field(description="API Request URL Params")] = Field(default_factory=dict)
    payload: Annotated[Optional[Union[dict, list]], Field(description="API Request Payload Body")] = Field(default_factory=dict)
    payload_type: Annotated[
        str,
        Field(description="Payload Content-Type", default="application/json")
    ]


def remove_duplicate_path_segment(base: str, api_path: str) -> str:
    """
    Remove duplicate segments between base_url.rstrip's end and api.lstrip's beginning if they match specified patterns.
    Use single forward slash when joining paths.
    """
    base_cleaned = base.rstrip('/')
    api_cleaned = api_path.lstrip('/')

    overlap_types = ["api", r"api/v\d+", r"v\d+"]

    for pattern in overlap_types:
        match_base = re.search(pattern + "$", base_cleaned)
        match_api = re.search("^" + pattern, api_cleaned)

        if match_base and match_api:
            base_segment = match_base.group(0)
            api_segment = match_api.group(0)

            if base_segment == api_segment:
                api_cleaned = api_cleaned[len(api_segment):].lstrip('/')
                break

    if base_cleaned and api_cleaned:
        return base_cleaned + "/" + api_cleaned  # Use single forward slash
    elif base_cleaned:
        return base_cleaned + "/"
    elif api_cleaned:
        return "/" + api_cleaned
    else:
        return "/" # or "", depending on requirements


def do_request(
        base_url: Annotated[str, "REST API System Base URL"],
        method: Annotated[str, "HTTP Method"],
        api: Annotated[str, "REST API URL"],
        headers: Annotated[Optional[dict], "API Request Headers"] = None,
        params: Annotated[Optional[dict], "API Request URL Params"] = None,
        payload: Annotated[Optional[Union[dict, list]], "API Request Payload Body"] = None,
        payload_type: Annotated[str, "Payload Content-Type"] = "application/json",
        parameter_categories: Annotated[Optional[dict], "HiREST-style parameter equivalence class metadata"] = None,
) -> dict:
    """
    :param base_url: REST API System Base URL
    :param method: HTTP Method
    :param api: REST API URL
    :param headers: API Request Headers
    :param params: API Request URL Params
    :param payload: API Request Payload Body
    :param payload_type: Payload Content-Type
    :return: API Response
    """
    global all_request_sequence

    headers = dict(headers or {})
    params = dict(params or {})
    parameter_categories = dict(parameter_categories or {})
    if payload is None:
        payload = {}
    timeout = 10
    proxies = {"http": None, "https": None}

    if is_rest_league_mode():
        base_url = global_vars_funcs_configs.CONFIG_BASE_URL

    # LLM calls may provide a template path plus its path parameters separately.
    # Render those parameters before URL construction so stateful IDs really flow
    # from an earlier response into the next request.
    path_template = api
    for path_name in re.findall(r"\{([^{}]+)\}", path_template):
        value = params.pop(path_name, None)
        if value is None and _env_flag("REST_LEAGUE_RESOURCE_POOL", "true"):
            value = _resource_value_for_parameter(path_name, path_template, 0)
        if value is not None:
            api = api.replace("{" + path_name + "}", quote(str(value), safe=""))
    if api != path_template:
        print(f"RESOURCE BINDING: rendered path {path_template} -> {api}")

    url = remove_duplicate_path_segment(base_url, api)
    request_started_at = time.time()
    method = str(method).upper()

    # Set Content-Type header
    if payload_type == "multipart/form-data":
        headers.pop("Content-Type", None)
    elif "Content-Type" not in headers:
        headers['Content-Type'] = payload_type

    # REST League authentication: automatically inject a token obtained by the
    # auth handshake so protected APIs (e.g. flight-search) stop returning 401.
    if is_rest_league_mode():
        headers = _apply_auth_header(headers)

    try:
        if method not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response = _send_http(method, url, headers, params, payload, payload_type, timeout)

        # Auth-wall retry: a 401 (or a 403/404 carrying explicit authentication
        # markers) triggers the register/login handshake exactly once, then
        # replays the request with the obtained token.  If a token was already
        # stored but the server now rejects it, force a fresh handshake (up to
        # the budget) instead of keeping the stale token.
        global _last_401_count
        try:
            _raw_status = response.status_code
            _raw_body = response.text
        except Exception:
            _raw_body = ""
        if is_rest_league_mode() and _is_auth_wall_response(
            _raw_status, _raw_body, has_token=bool(auth_state.get("token"))
        ):
            _last_401_count += 1
            print(
                f"AUTH RETRY: auth-wall response {_raw_status} "
                f"({_last_401_count}/{_handshake_401_budget})"
            )
            if _last_401_count <= _handshake_401_budget and not _in_handshake and _auth_enabled():
                force = bool(auth_state.get("token"))  # stale token -> re-handshake
                if perform_auth_handshake(force=force):
                    headers = _apply_auth_header(headers)
                    response = _send_http(method, url, headers, params, payload, payload_type, timeout)
                    print(f"AUTH RETRY: replayed request after handshake -> {response.status_code}")

        # Body-on-GET retry: some services document a compound parameter as a
        # query parameter while the implementation actually needs it in a GET
        # request body.  Trigger when we sent a compound query parameter and the
        # call failed as a client error (or the server said the body is missing),
        # then re-send the same data as a JSON body.
        try:
            _body_text = response.text
        except Exception:
            _body_text = ""
        if method == "GET" and not payload and params:
            looks_compound = any(
                "." in str(key) or (isinstance(value, str) and value.strip().startswith("{"))
                for key, value in params.items()
            )
            body_missing = "request body is missing" in str(_body_text).lower()
            status_now = getattr(response, "status_code", 0)
            if looks_compound and (body_missing or 400 <= status_now < 500):
                retry_body = _reconstruct_body_from_params(params)
                if retry_body:
                    print(f"GET BODY RETRY: resending parameters as request body -> {api}")
                    try:
                        response = _send_http(method, url, headers, {}, retry_body, "application/json", timeout)
                        print(f"GET BODY RETRY: replay status -> {response.status_code}")
                    except Exception as exc:
                        print(f"GET BODY RETRY failed: {exc}")

        elapsed_ms = int((time.time() - request_started_at) * 1000)

        try:
            response_data = response.json()
        except ValueError:
            response_data = response.text
        if isinstance(response_data, dict) or isinstance(response_data, list):
            response_data = json.dumps(response_data, separators=(',', ':'))
        full_response_data = response_data
        if len(str(response_data)) > 3500:
            response_data = str(response_data)[:3500] + '... [JSON TOO LONG, TRUNCATED]'

        print(f"DO REQUEST: {method=} {url=} {headers=} {params=} {payload=} {payload_type=} {response_data=}")
        is_2xx = 200 <= response.status_code < 300
        is_5xx = 500 <= response.status_code < 600
        if is_2xx:
            if method == "DELETE":
                # Track which resource id was deleted so stale ids are not reused.
                for path_name in re.findall(r"\{([^{}]+)\}", path_template):
                    matched = re.search(re.escape("{" + path_name + "}"), path_template)
                    if matched:
                        deleted_value = _resource_value_for_parameter(path_name, path_template)
                        if deleted_value is not None:
                            resource_key = _normalize_resource_key(path_name)
                            if resource_key == "id":
                                route_parts = [part for part in path_template.strip("/").split("/") if part]
                                placeholder_index = route_parts.index("{" + path_name + "}")
                                if placeholder_index > 0:
                                    resource_key = _singularize_resource_name(route_parts[placeholder_index - 1]) + "id"
                            mark_resource_invalid(resource_key, deleted_value)
            update_resource_pool_from_response(
                payload,
                api_endpoint=getattr(test_scenario.current_test_case, "api_endpoint", None),
                api_path=api,
            )
            update_resource_pool_from_response(
                full_response_data,
                api_endpoint=getattr(test_scenario.current_test_case, "api_endpoint", None),
                api_path=api,
            )
            update_resource_pool_from_response_headers(
                response.headers,
                api_endpoint=getattr(test_scenario.current_test_case, "api_endpoint", None),
                api_path=api,
            )
            _record_automatic_useful_items(
                payload,
                api_endpoint=getattr(test_scenario.current_test_case, "api_endpoint", None),
                api_path=api,
            )
            _record_automatic_useful_items(
                response_data,
                api_endpoint=getattr(test_scenario.current_test_case, "api_endpoint", None),
                api_path=api,
            )
        item = {
            "timestamp": request_started_at,
            "elapsed_ms": elapsed_ms,
            "method": method,
            "api": api,
            "url": url,
            "api_endpoint": getattr(test_scenario.current_test_case, "api_endpoint", None),
            "headers": headers,
            "params": params,
            "payload": payload,
            "payload_type": payload_type,
            "parameter_categories": parameter_categories,
            "request_data": f"{method=} {api=} {params=} {payload=}",
            "response_code": response.status_code,
            "response_data": response_data,
            "full_response_data": full_response_data,
            "response_headers": dict(response.headers),
            "is_2xx": is_2xx,
            "is_5xx": is_5xx,
            "fault_signature": (
                re.sub(r"\s+", " ", str(response_data)).strip()[:1000]
                if is_5xx else None
            ),
            "request_error": None,
            "resource_pool_keys": sorted(resource_pool.keys()),
        }
        # for recording json
        all_request_sequence.append(item)
        import seqrest.engine.mutation as adaptive_testing
        adaptive_testing.runtime.observe(item, global_vars_funcs_configs.catalog)
        import seqrest.engine.sequence as sequence_testing
        sequence_testing.resources.observe(item, global_vars_funcs_configs.catalog)
        # for agent flow control (skip during the auth handshake: those requests
        # are setup, not scenario steps, and must not enter the validation queue)
        if not _in_handshake:
            test_scenario.add_next_response_for_validation(item)

        return {
            'status_code': response.status_code,
            'response_data': response_data,
        }
    except Exception as e:
        if is_rest_league_mode():
            elapsed_ms = int((time.time() - request_started_at) * 1000)
            item = {
                "timestamp": request_started_at,
                "elapsed_ms": elapsed_ms,
                "method": method,
                "api": api,
                "url": url,
                "api_endpoint": getattr(test_scenario.current_test_case, "api_endpoint", None),
                "headers": headers,
                "params": params,
                "payload": payload,
                "payload_type": payload_type,
                "parameter_categories": parameter_categories,
                "request_data": f"{method=} {api=} {params=} {payload=}",
                "response_code": None,
                "response_data": f"Request failed: {e}",
                "is_2xx": False,
                "is_5xx": False,
                "fault_signature": None,
                "request_error": str(e),
            }
            all_request_sequence.append(item)
            if not _in_handshake:
                test_scenario.add_next_response_for_validation(item)
            print(f"DO REQUEST ERROR RECORDED: {method=} {url=} error={e}")
            return {
                "status_code": None,
                "response_data": item["response_data"],
                "request_error": str(e),
            }
        raise RuntimeError(f"Request failed: {e}")


def record_result(oracle: str, judge_reason: str, align_with_expected: bool, request_info: str, response: str) -> str:
    """
    :param align_with_expected: if the oracle is aligned with response
    :param request_info: request info
    :param judge_reason: the reason of the judgement
    :param oracle: oracle string, expected output
    :param response: actual response code and body
    """
    global all_request_sequence, right_results, wrong_results  # Declare global variables

    if align_with_expected:
        right_results.append({
            "request_info": request_info,
            "oracle": oracle,
            "judge_reason": judge_reason,
            "response": response
        })
    else:
        wrong_results.append({
            "request_info": request_info,
            "oracle": oracle,
            "judge_reason": judge_reason,
            "response": response
        })

    print(f"[Invoke record_result] {align_with_expected=} {len(right_results)=} {len(wrong_results)=}")

    return f"finished record_result: {align_with_expected=}"
