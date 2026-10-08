
def _string_list(value):
    """Return only string items from a list; tolerate malformed OpenAPI where
    `required` may contain dicts or other non-string entries."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _example_schema(schema: dict) -> dict:
    """Pick a usable branch and merge allOf for example generation."""
    if not isinstance(schema, dict):
        return {}
    merged = dict(schema)
    parts = schema.get("allOf") or []
    if parts:
        merged.pop("allOf", None)
        merged.setdefault("properties", {})
        required = _string_list(merged.get("required", []))
        for part in parts:
            part = _example_schema(part)
            merged.update({k: v for k, v in part.items() if k not in {"properties", "required"}})
            merged["properties"].update(part.get("properties", {}))
            required.extend(_string_list(part.get("required", [])))
        merged["required"] = list(dict.fromkeys(required))
    if schema.get("oneOf"):
        return _example_schema(schema["oneOf"][0])
    if schema.get("anyOf"):
        return _example_schema(schema["anyOf"][0])
    merged["required"] = _string_list(merged.get("required", []))
    return merged


def generate_example(schema: dict):
    """Generate a JSON example from a Swagger schema."""
    schema = _example_schema(schema)
    if "example" in schema:
        return schema["example"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    example = {}
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    for field, field_schema in properties.items():
        field_schema = _example_schema(field_schema)
        field_type = field_schema.get("type", "object" if "properties" in field_schema else "string")
        field_example = f"{field} example"  # Default example text

        # Define example values based on field type
        if field_type == "string":
            field_example = field_schema.get("example", f"{field} example")
        elif field_type == "integer":
            field_example = field_schema.get("example", 123)
        elif field_type == "boolean":
            field_example = field_schema.get("example", True)
        elif field_type == "array":
            item_schema = field_schema.get("items", {})
            field_example = [generate_example(item_schema)]
        elif field_type == "object":
            field_example = generate_example(field_schema)

        # Set required fields with defined example or default example
        if field in required_fields:
            example[field] = field_example
        elif "example" in field_schema:
            example[field] = field_example

    return example


def parse_request_body(request_body: dict) -> dict:
    """Parse Swagger requestBody schema to JSON example."""
    if 'content' not in request_body:
        raise ValueError("Invalid request body schema")

    # Extract JSON schema from the request body content
    json_schema = request_body['content'].get('application/json', {}).get('schema', {})
    return generate_example(json_schema)


def schema_to_text(schema, indent=0, max_indent=10000):
    """
    Converts a JSON schema to a readable text format.

    Parameters:
        schema (dict): The JSON schema to be converted.
        indent (int): The indentation level for nested objects.
        max_indent (int): The maximum indentation level for nested objects.

    Returns:
        str: Text representation of the schema.
    """
    schema = _example_schema(schema)
    text = ""
    indent_str = "  " * indent

    if indent > max_indent:
        indent_str = indent_str + " "*2
        text += f"{indent_str}(The content is too long, omit it.)\n"
        return text

    if "$ref" in schema:
        text += f"{indent_str}Reference: {schema['$ref']}\n"
    if schema.get("description"):
        text += f"{indent_str}Description: {schema['description']}\n"
    if "example" in schema:
        text += f"{indent_str}Example: {schema['example']}\n"
    if "enum" in schema:
        text += f"{indent_str}Allowed values: {schema['enum']}\n"
    if schema.get("oneOf") or schema.get("anyOf"):
        key = "oneOf" if schema.get("oneOf") else "anyOf"
        text += f"{indent_str}{key} alternatives:\n"
        for index, branch in enumerate(schema.get(key, []), 1):
            text += f"{indent_str}- Alternative {index}:\n"
            text += schema_to_text(branch, indent + 1, max_indent)
    if "type" in schema or "properties" in schema:
        schema_type = schema["type"]
        if schema_type == "object":
            text += f"{indent_str}Object with properties:\n"
            if schema.get("required"):
                text += f"{indent_str}Required fields: {schema['required']}\n"
            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "object" if "properties" in prop_schema else "Unknown type")
                prop_desc = prop_schema.get("description", "")
                prop_enum = prop_schema.get("enum", [])
                if prop_enum:
                    text += f"{indent_str}- {prop} ({prop_type}): {prop_desc}. Enum for this prop:{prop_enum}\n"
                else:
                    text += f"{indent_str}- {prop} ({prop_type}): {prop_desc}\n"
                # Recursively handle nested objects
                if prop_type == "object" or prop_type == "array":
                    text += schema_to_text(prop_schema, indent + 1, max_indent)
        elif schema_type == "array":
            items = schema.get("items", {})
            text += f"{indent_str}Array of:\n"
            text += schema_to_text(items, indent + 1, max_indent)
        else:
            text += f"{indent_str}{schema_type.capitalize()}\n"
    elif "properties" in schema:
        properties = schema.get("properties", {})
        for prop, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "Unknown type")
            prop_desc = prop_schema.get("description", "")
            prop_enum = prop_schema.get("enum", [])
            if prop_enum:
                text += f"{indent_str}- {prop} ({prop_type}): {prop_desc}. Enum for this prop:{prop_enum}\n"
            else:
                text += f"{indent_str}- {prop} ({prop_type}): {prop_desc}\n"
            # Recursively handle nested objects
            if prop_type == "object" or prop_type == "array":
                text += schema_to_text(prop_schema, indent + 1, max_indent)

    return text


def swagger_to_text(endpoint):
    """
    Converts a single Swagger endpoint specification into an English textual description.

    Parameters:
        endpoint (dict): A dictionary representing a single Swagger endpoint specification.

    Returns:
        str: A textual description of the endpoint.
    """
    # Extract essential parts of the endpoint
    tags = ", ".join(endpoint.get("tags", []))
    summary = endpoint.get("summary", "")
    operation_id = endpoint.get("operationId", "")
    request_body = endpoint.get("requestBody", {})
    responses = endpoint.get("responses", {})

    # Construct the basic description with summary and tags
    description = f"OperationId: {operation_id}: {summary}\n"

    # Handle request parameters if they exist
    parameters = endpoint.get("parameters", [])
    if parameters:
        description += "Parameters:\n"
        for param in parameters:
            param_desc = param.get("description", "No description provided.")
            param_name = param.get("name", "Unnamed")
            param_required = "Required" if param.get("required", False) else "Optional"
            param_type = param.get("schema", {}).get("type", "Unknown type")
            description += f"- {param_name} ({param_type}, {param_required}): {param_desc}\n"

    # Handle request body if it exists
    if request_body:
        description += f"Request Body:\n"
        rb_description = request_body.get("description", "")
        if rb_description:
            description += f"Description: {rb_description}.\n"

        request_body_content = request_body.get("content", {})
        if len(request_body_content) > 1:
            # default request body type
            request_body_type = "application/json"
        elif len(request_body_content) == 0:
            request_body_type = "No request body in document\n"
        else:
            request_body_type = list(request_body_content.keys())[0]
        if request_body_type == "application/json":
            description += f"Content Type: {request_body_type}\n"
        else:
            description += f"**Content Type**: **{request_body_type}**\n"

        body_schema = request_body_content.get(request_body_type).get("schema", {})
        if body_schema:
            description += "Body schema:\n" + schema_to_text(body_schema)


    # Handle responses
    if responses:
        description += "Responses:\n"
        for status, response in responses.items():
            response_desc = response.get("description", "No description provided.")
            resp_body_schema = response.get("content", {}).get("application/json", {}).get("schema", {})
            description += f"- Status {status}: {response_desc}. "
            if resp_body_schema and type(resp_body_schema) == dict:
                description += "Response Body:\n"
                valida_description = schema_to_text(resp_body_schema, indent=2)
                if len(valida_description) > 10000:
                    valida_description = schema_to_text(resp_body_schema, indent=2, max_indent=3)
                description += valida_description
            else:
                description += "\n"

    return description
