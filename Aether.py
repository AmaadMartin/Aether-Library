# Aether.py
import requests
import json
from openai import OpenAI
import jsonschema
from pydantic import BaseModel
import asyncio
import aiohttp
import threading
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("AETHER_BASE_URL")

class EvaluationInput(BaseModel):
    task: str
    input: dict
    output_schema: dict
    output: dict
    version: str
    function_key: str
    api_key: str

class AetherClient:
    def __init__(self, api_key, openai_api_key, base_url=BASE_URL):
        self.api_key = api_key
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.openai = OpenAI(api_key = self.openai_api_key)

    def __call__(self, function_key, input_json, version=None, for_eval=False):
        return self.call_function(function_key, input_json, version, for_eval)

    def call_function(self, function_key, input_json, version=None, for_eval=False):
        # Get function parameters
        headers = {'X-API-Key': self.api_key}
        if version is None:
            version = "None"
        response = requests.get(f"{self.base_url}/function_call/{function_key}/{version}", headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error retrieving function parameters: {response.text}")
        function_params = response.json()

        # Validate input JSON against input schema
        input_schema = function_params['input_schema']
        try:
            jsonschema.validate(instance=input_json, schema=input_schema)
        except jsonschema.exceptions.ValidationError as e:
            raise Exception(f"Input validation error: {str(e)}")

        # Prepare OpenAI API call
        prompt = function_params['prompt']
        model = function_params['model']
        temperature = function_params['temperature']
        old_schema = function_params['output_schema']

        input = f"{json.dumps(input_json)}"

        # convert output schema format
        output_schema = self.convert_output_schema_to_openai_function_definition(old_schema)
        # print(output_schema)

        # Make OpenAI API call
        response = self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input}
            ],
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": output_schema
            }
        )

        output = json.loads(response.choices[0].message.content)
        # print("input", input_json)
        # print("output", output)

        if not for_eval:
            input_data = {
                "task": function_params['task'],
                "input": input_json,
                "output_schema": old_schema,
                "output": output,
                "version": function_params['version'],
                "function_key": function_key,
                "api_key": self.api_key
            }
            # print("input params", input_data)
            thread = threading.Thread(target=self._run_async_evaluation, args=(input_data, headers))
            thread.start()

        return output
        
        # send task, input, output schema, and output to api for evaluation asynchronously
        asyncio.create_task(self.evaluate_output_async(input_data, headers))
        return output
    
    def _run_async_evaluation(self, input_data, headers):
        asyncio.run(self.evaluate_output_async(input_data, headers))

    async def evaluate_output_async(self, input_data, headers):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/evaluate", json=input_data, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Error evaluating output: {await response.text()}")
        
    
    def convert_output_schema_to_openai_function_definition(self, output_schema):
    # Remove 'metrics' fields and 'title'
        cleaned_schema = {}
        def clean_schema(schema):
            if isinstance(schema, dict):
                schema = schema.copy()
                schema.pop('metrics', None)
                schema.pop('title', None)
                schema.pop('description', None)
                schema.pop('desiredProperties', None)
                if 'properties' in schema:
                    schema['properties'] = {k: clean_schema(v) for k, v in schema['properties'].items()}
                if 'items' in schema:
                    schema['items'] = clean_schema(schema['items'])
                if 'required' not in schema and 'properties' in schema:
                    schema['required'] = list(schema['properties'].keys())
                schema['additionalProperties'] = False
            return schema

        cleaned_schema['schema'] = clean_schema(output_schema)
        cleaned_schema['name'] = 'output'
        cleaned_schema['strict'] = True
        return cleaned_schema