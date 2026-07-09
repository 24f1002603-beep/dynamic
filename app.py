import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ.get(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    )
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
    schema: dict


@app.get("/")
def home():
    return {"status": "running"}


def default_value(dtype):

    return None


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicRequest):

    prompt = f"""
You are a structured information extraction engine.

Extract ONLY the fields requested.

TEXT

{req.text}

SCHEMA

{json.dumps(req.schema, indent=2)}

Rules

1. Return ONLY valid JSON.
2. Return EXACTLY the keys in the schema.
3. No extra keys.
4. If a value cannot be found return null.
5. Dates MUST be YYYY-MM-DD.
6. integer -> JSON integer
7. float -> JSON number
8. boolean -> true/false
9. array[string] -> JSON array of strings
10. array[integer] -> JSON array of integers

Return ONLY JSON.
"""

    try:

        response = client.chat.completions.create(

            model="openrouter/free",

            temperature=0,

            response_format={
                "type": "json_object"
            },

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = response.choices[0].message.content

        result = json.loads(text)

        final = {}

        for key in req.schema:

            if key in result:
                final[key] = result[key]
            else:
                final[key] = None

        return final

    except Exception as e:

        print(e)

        return {
            key: None
            for key in req.schema
        }