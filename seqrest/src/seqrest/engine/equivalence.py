import os
import re
from dataclasses import dataclass
from typing import Any, Optional


MISS_KEY = "<MISS_KEY>"


@dataclass(frozen=True)
class ParameterCategory:
    category_id: str
    name: str
    valid: bool
    value: Any
    reason: str


def enabled() -> bool:
    return os.getenv("HIREST_EQUIV_MODE", "true").strip().lower() in {"1", "true", "yes", "y", "on"}


def _schema_type(schema: dict) -> str:
    schema_type = schema.get("type")
    if schema_type:
        return schema_type
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return "string"


def _first_present(schema: dict, keys: list[str]) -> Optional[Any]:
    for key in keys:
        if key in schema and schema[key] not in (None, ""):
            return schema[key]
    return None


def _valid_string_value(name: str, schema: dict, variant: int) -> str:
    fmt = schema.get("format", "")
    lower_name = name.lower()
    if "email" in lower_name or fmt == "email":
        return f"user{variant}@example.com"
    if "uuid" in lower_name or fmt == "uuid":
        return f"123e4567-e89b-12d3-a456-42661417{variant:04d}"
    if "datetime" in lower_name or fmt == "date-time":
        # Different services accept different datetime shapes; some even reject a
        # full timestamp and only want the date (their parser appends "T00:00:00"
        # itself), so the cycle includes a bare date.
        return [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00",
            "2024-01-01",
            "2024-01-01T00:00:00.000",
        ][variant % 4]
    if "date" in lower_name or fmt == "date":
        return "2024-01-01"
    if "time" in lower_name or fmt == "time":
        return ["00:00:00", "12:00:00", "23:59:59"][variant % 3]
    if "url" in lower_name or fmt in {"uri", "url"}:
        return "http://example.com"
    if "password" in lower_name:
        return f"Passw0rd{variant}"
    if lower_name in {"language", "lang"}:
        return ["en", "en-US", "auto", "de-DE"][variant % 4]
    if "country" in lower_name:
        return ["US", "GB", "DE", "CN"][variant % 4]
    if "status" in lower_name:
        return ["available", "active", "open", "enabled"][variant % 4]
    if "tag" in lower_name:
        return ["friendly", "playful", "cute", "smart"][variant % 4]
    if "file" in lower_name:
        return f"file{variant}.txt"
    if "name" in lower_name or "title" in lower_name:
        return f"{name}_{variant}"
    return f"{name}_{variant}"


def _invalid_string_value(name: str, schema: dict, variant: int) -> str:
    fmt = schema.get("format", "")
    lower_name = name.lower()
    if "email" in lower_name or fmt == "email":
        return "not-an-email"
    if "uuid" in lower_name or fmt == "uuid":
        return "not-a-uuid"
    if "date" in lower_name or fmt in {"date", "date-time"}:
        return "not-a-date"
    if "url" in lower_name or fmt in {"uri", "url"}:
        return "not a url"
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and max_length >= 0:
        return "X" * (max_length + 1)
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and min_length > 0:
        return ""
    return ["", "null", "INVALID_%21", "A" * 256][variant % 4]


def _description_samples(schema: dict) -> list[Any]:
    description = schema.get("description")
    if not isinstance(description, str) or not description.strip():
        return []

    samples: list[str] = []
    quoted = re.findall(r"['\"]([^'\"]{1,80})['\"]", description)
    samples.extend(quoted)

    brace_match = re.search(r"\{([^{}]{1,200})\}", description)
    if brace_match:
        samples.extend(part.strip() for part in re.split(r"[,/|]", brace_match.group(1)))

    use_match = re.search(r"\buse\s+([^.;。]+)", description, flags=re.IGNORECASE)
    if use_match:
        samples.extend(part.strip() for part in re.split(r",|\bor\b|\band\b", use_match.group(1), flags=re.IGNORECASE))

    token_candidates = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{1,30}\b", description)
    semantic_tokens = [
        token for token in token_candidates
        if re.search(r"\d", token) or token.lower() in {"available", "pending", "sold", "active", "open", "enabled"}
    ]
    samples.extend(semantic_tokens)

    cleaned = []
    seen = set()
    stop_words = {"the", "for", "and", "or", "use", "using", "testing", "parameter", "value", "values"}
    for sample in samples:
        sample = sample.strip(" `[](){}<>:;.,")
        if not sample or sample.lower() in stop_words:
            continue
        if sample not in seen:
            seen.add(sample)
            cleaned.append(sample)
    return cleaned[:6]


def _choose_present_valid_category(name: str, schema: dict, required: bool, variant: int) -> ParameterCategory:
    categories = categories_for_parameter(name, schema, required=required, variant=variant)
    present = [category for category in categories if category.valid and category.value != MISS_KEY]
    non_empty_present = [category for category in present if category.category_id != "array_empty"]
    if non_empty_present:
        return non_empty_present[variant % len(non_empty_present)]
    if present:
        return present[variant % len(present)]
    return choose_category(name, schema, required=required, variant=variant, prefer_valid=True)


def categories_for_parameter(name: str, schema: dict, required: bool = False, variant: int = 0) -> list[ParameterCategory]:
    if not isinstance(schema, dict):
        schema = {}

    categories: list[ParameterCategory] = []
    if not required:
        categories.append(
            ParameterCategory(
                "missing_optional",
                "optional parameter omitted",
                True,
                MISS_KEY,
                "HiREST missing-key category for optional parameters.",
            )
        )
    else:
        categories.append(
            ParameterCategory(
                "missing_required",
                "required parameter omitted",
                False,
                MISS_KEY,
                "HiREST missing-key category for required parameters.",
            )
        )

    example = _first_present(schema, ["example", "default"])
    if example is not None:
        categories.append(ParameterCategory("schema_example", "schema example value", True, example, "OpenAPI example/default."))

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        # Every declared enum member is a meaningful valid equivalence class.
        # The per-operation mutation budget bounds execution; truncating here
        # silently makes later members unreachable even when budget remains.
        for index, enum_value in enumerate(enum):
            categories.append(
                ParameterCategory(
                    f"enum_{index}",
                    "declared enum value",
                    True,
                    enum_value,
                    "OpenAPI enum member.",
                )
            )
        categories.append(
            ParameterCategory(
                "enum_outside",
                "value outside declared enum",
                False,
                f"invalid_{name}_{variant}",
                "Value deliberately outside OpenAPI enum.",
            )
        )
        return categories

    schema_type = _schema_type(schema)
    fmt = schema.get("format", "")
    if schema_type == "string" and fmt == "binary":
        categories.extend([
            ParameterCategory(
                "binary_file_valid",
                "valid multipart file",
                True,
                (f"file{variant}.txt", f"sample file content {variant}", "text/plain"),
                "Multipart file value with filename, content, and media type.",
            ),
            ParameterCategory(
                "binary_file_empty",
                "empty multipart file",
                False,
                (f"empty{variant}.txt", "", "text/plain"),
                "Empty file boundary class.",
            ),
            ParameterCategory(
                "binary_wrong_type",
                "non-file multipart value",
                False,
                f"not_a_file_{variant}",
                "Type-violating file upload class.",
            ),
        ])
        return categories

    if schema_type == "integer":
        minimum = schema.get("minimum", 1)
        maximum = schema.get("maximum")
        valid_values = [minimum, 0, 1, variant + 1]
        if isinstance(maximum, int):
            valid_values.append(maximum)
        for value in dict.fromkeys(int(v) for v in valid_values):
            categories.append(ParameterCategory(f"int_valid_{value}", "valid integer class", True, value, "Integer value class."))
        categories.extend([
            ParameterCategory("int_negative_boundary", "negative integer boundary", False, -1, "Invalid or boundary integer class."),
            ParameterCategory("int_large_boundary", "large integer boundary", False, 2147483647, "Large integer boundary class."),
            ParameterCategory("int_wrong_type", "non-integer value", False, f"{name}_not_int", "Type-violating integer class."),
        ])
    elif schema_type == "number":
        minimum = schema.get("minimum", 1.25)
        categories.extend([
            ParameterCategory("num_minimum", "minimum numeric value", True, float(minimum), "Numeric minimum class."),
            ParameterCategory("num_zero", "zero numeric value", True, 0.0, "Numeric zero class."),
            ParameterCategory("num_large", "large numeric boundary", False, 1.0e9, "Large numeric boundary class."),
            ParameterCategory("num_wrong_type", "non-numeric value", False, f"{name}_not_number", "Type-violating numeric class."),
        ])
    elif schema_type == "boolean":
        categories.extend([
            ParameterCategory("bool_true", "true boolean value", True, True, "Boolean true class."),
            ParameterCategory("bool_false", "false boolean value", True, False, "Boolean false class."),
            ParameterCategory("bool_wrong_type", "non-boolean value", False, "not_boolean", "Type-violating boolean class."),
        ])
    elif schema_type == "array":
        item_schema = schema.get("items", {})
        item_required = True
        samples = _description_samples(schema)
        if samples:
            categories.append(
                ParameterCategory(
                    "array_description_samples",
                    "array values inferred from description",
                    True,
                    samples[: min(3, len(samples))],
                    "Representative array values inferred from OpenAPI description.",
                )
            )
        item_value = choose_category(name, item_schema, item_required, variant, prefer_valid=True).value
        min_items = schema.get("minItems", 0)
        categories.append(ParameterCategory("array_single_valid", "single valid item", True, [item_value], "Array with one valid item."))
        categories.append(ParameterCategory("array_empty", "empty array", (not required) and min_items == 0, [], "Empty-array boundary class."))
        invalid_item = choose_category(name, item_schema, item_required, variant, prefer_valid=False).value
        categories.append(ParameterCategory("array_invalid_item", "array with invalid item", False, [invalid_item], "Array item violates item schema."))
    elif schema_type == "object":
        properties = schema.get("properties", {})
        required_props = set(schema.get("required", []))
        include_all = not required_props
        value = {}
        for prop, prop_schema in properties.items():
            category = _choose_present_valid_category(prop, prop_schema, prop in required_props or include_all, variant)
            if category.value != MISS_KEY:
                value[prop] = category.value
        categories.append(ParameterCategory("object_valid_required", "object with valid fields", True, value, "Object satisfying required fields."))
        if required_props:
            missing_value = dict(value)
            missing_value.pop(next(iter(required_props)), None)
            categories.append(ParameterCategory("object_missing_required", "object missing a required field", False, missing_value, "Object violates required field rule."))
    else:
        categories.append(
            ParameterCategory(
                "string_valid_semantic",
                "valid string or semantic value",
                True,
                _valid_string_value(name, schema, variant),
                "String category inferred from name, format, and schema.",
            )
        )
        samples = _description_samples(schema)
        for index, sample in enumerate(samples):
            categories.append(
                ParameterCategory(
                    f"description_sample_{index}",
                    "value inferred from description",
                    True,
                    sample,
                    "Representative value inferred from OpenAPI description.",
                )
            )
        categories.append(
            ParameterCategory(
                "string_invalid_format",
                "invalid string format or boundary",
                False,
                _invalid_string_value(name, schema, variant),
                "String category deliberately violating format, length, or semantic hints.",
            )
        )

    return categories


def choose_category(
    name: str,
    schema: dict,
    required: bool = False,
    variant: int = 0,
    prefer_valid: Optional[bool] = None,
) -> ParameterCategory:
    categories = categories_for_parameter(name, schema, required=required, variant=variant)
    if prefer_valid is None:
        prefer_valid = variant % 5 not in {2, 4}

    filtered = [category for category in categories if category.valid == prefer_valid]
    if not filtered:
        filtered = categories
    return filtered[variant % len(filtered)]
