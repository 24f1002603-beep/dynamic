import json
import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Direct integration with AI Pipe using your environment variable token
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
    schema: dict  # Example: {"customer_name": "string", "quantity": "integer"}


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicRequest):

    # 1. TRANSLATE CUSTOM SHORTHAND SCHEMA INTO COMPLIANT JSON SCHEMA
    type_map = {
        "string": "string",
        "date": "string",
        "integer": "integer",
        "float": "number",
        "boolean": "boolean"
    }

    properties = {}
    required_keys = []

    for key, val in req.schema.items():
        inferred_type = type_map.get(str(val).lower(), "string")
        
        # We append "null" so strict validation allows missing fields gracefully
        properties[key] = {
            "type": [inferred_type, "null"]
        }
        
        if str(val).lower() == "date":
            properties[key]["description"] = "ISO date string formatted strictly as YYYY-MM-DD"
        elif str(val).lower() == "integer":
            properties[key]["description"] = "A raw number integer value"
        elif str(val).lower() == "float":
            properties[key]["description"] = "A floating point numeric decimal value"
            
        required_keys.append(key)

    openai_compatible_schema = {
        "type": "object",
        "properties": properties,
        "required": required_keys,
        "additionalProperties": False
    }

    # 2. CALL LLM WITH NATIVE STRUCTURED SCHEMA OVERRIDE
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "DynamicExtraction",
                    "strict": True,
                    "schema": openai_compatible_schema
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert structured data parser. Extract fields requested by the schema. "
                        "Crucial rule: Any field labeled as a date MUST be parsed and output as a "
                        "strict YYYY-MM-DD string value. If an implicit date string or conversational description "
                        "is encountered, calculate or resolve it down to its exact target format."
                    )
                },
                {
                    "role": "user",
                    "content": f"Text to extract from:\n{req.text}"
                }
            ],
        )

        text_output = response.choices[0].message.content
        data = json.loads(text_output)
    except Exception as e:
        print(f"Extraction failed: {e}")
        data = {}

    # 3. TYPE CASTING & CORRECTIONS PIPELINE
    final_output = {}
    for key, expected_type in req.schema.items():
        raw_val = data.get(key, None)
        expected_type_lower = str(expected_type).lower()

        if raw_val is None or str(raw_val).lower() == "null":
            final_output[key] = None
            continue

        try:
            if expected_type_lower == "integer":
                final_output[key] = int(float(raw_val))
            elif expected_type_lower == "float":
                final_output[key] = float(raw_val)
            elif expected_type_lower == "boolean":
                if str(raw_val).lower() in ["true", "1", "yes"]:
                    final_output[key] = True
                elif str(raw_val).lower() in ["false", "0", "no"]:
                    final_output[key] = False
                else:
                    final_output[key] = bool(raw_val)
            elif expected_type_lower == "date":
                # Ensure spacing is stripped from date extractions
                final_output[key] = str(raw_val).strip()
            else:
                final_output[key] = str(raw_val)
        except Exception:
            # Fall back safely to null if cast operations hit formatting anomalies
            final_output[key] = None

    # 4. GUARANTEE COMPLIANCE AND EXACT KEY SEQUENCING
    return {key: final_output.get(key, None) for key in req.schema}
