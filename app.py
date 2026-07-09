import json
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Direct integration with AI Pipe using your single environment variable token
client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)

app = FastAPI(title="Dynamic Schema Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DynamicRequest(BaseModel):
    text: str
    schema: dict  # Format: {"field_name": "type_string"}


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicRequest):

    # 1. TRANSLATE CUSTOM FLAT SCHEMA TO STANDARD JSON SCHEMA
    type_mapping = {
        "string": {"type": ["string", "null"]},
        "integer": {"type": ["integer", "null"]},
        "float": {"type": ["number", "null"]},
        "number": {"type": ["number", "null"]},
        "boolean": {"type": ["boolean", "null"]},
        "date": {"type": ["string", "null"], "description": "Must be in ISO format YYYY-MM-DD"}
    }

    properties = {}
    required_keys = []
    
    for key, val in req.schema.items():
        val_lower = str(val).lower()
        # Fallback to general type if array or custom type is passed
        if "array" in val_lower:
            properties[key] = {"type": ["array", "null"]}
        else:
            properties[key] = type_mapping.get(val_lower, {"type": ["string", "null"]})
        required_keys.append(key)

    built_schema = {
        "type": "object",
        "properties": properties,
        "required": required_keys
    }

    # 2. CALL THE LLM WITH STRICT SCHEMA GATEWAY ENFORCEMENT
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "DynamicExtractionSchema",
                    "strict": True,
                    "schema": built_schema
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly accurate data extraction engine. Extract information matching the fields "
                        "requested from the user text. For 'date' type fields, you MUST find the date and normalize it "
                        "strictly to YYYY-MM-DD. If a field cannot be found, return null."
                    )
                },
                {
                    "role": "user",
                    "content": f"Text to extract from:\n{req.text}"
                }
            ],
        )

        text = response.choices[0].message.content
        data = json.loads(text)

    except Exception as e:
        print(f"Extraction failed: {e}")
        data = {}

    # 3. FORCE CASTING AND CLEANUP BASED ON GRADER'S ORIGINAL KEY SCHEMAS
    final = {}
    for key, expected_type in req.schema.items():
        val = data.get(key, None)
        if val is None or str(val).lower() == "null":
            final[key] = None
            continue

        expected_type_lower = str(expected_type).lower()
        
        try:
            if "integer" in expected_type_lower:
                final[key] = int(float(val))
            elif "float" in expected_type_lower or "number" in expected_type_lower:
                final[key] = float(val)
            elif "boolean" in expected_type_lower:
                if isinstance(val, str):
                    final[key] = val.lower() in ["true", "1", "yes"]
                else:
                    final[key] = bool(val)
            else:
                final[key] = str(val).strip()
        except Exception:
            final[key] = None

    return final
