# library/_Aether/AetherFunction.py
from .AetherCall import AetherCall
from openai import OpenAI
import threading
import json


class AetherFunction:
    def __init__(self, function_key, api, version=None, openai_key=None):
        # Initialize the function
        self.function_key = function_key
        self.api = api
        self.version = version
        self.openai_key = openai_key
        if openai_key is not None:
            self.openai = OpenAI(api_key=openai_key)
        self.current = version == None
        self.init()

    def init(self):
        function_data = self.api.getFunctionData(self.function_key)
        self.version = (
            function_data["current_version"] if self.version is None else self.version
        )
        self.name = function_data["name"]
        self.task = function_data["task"]
        self.type = function_data["type"]
        self.parameters = self.api.getParameters(self)
        self.input_schema = function_data["input_schema"]
        self.output_schema = function_data["output_schema"]
        self.metrics = function_data["metrics"]
        self.test_set = function_data["test_set"]
        self.version_map = function_data["version_map"]
        self.version_tree = function_data["version_tree"]

    def __call__(self, input_json):
        if self.openai_key is None:
            raise Exception("OpenAI key not set")

        call = self.init_call()
        for key in input_json:
            call.input(key, input_json[key])
        params = self.get_parameters()
        prompt = params["prompt"]
        model = params["model"]
        temperature = params["temperature"]
        old_schema = self.output_schema
        output_schema = self.convert_output_schema_to_openai_function_definition(
            old_schema
        )
        input = f"{json.dumps(input_json)}"
        # print("input_json", type(input_json), input_json)

        call.status("running")
        # Make OpenAI API call
        response = self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input},
            ],
            temperature=temperature,
            response_format={"type": "json_schema", "json_schema": output_schema},
        )

        output = json.loads(response.choices[0].message.content)
        for key in output:
            call.output(key, output[key])

        call.eval()

        call.status("complete")
        return output

    def get_version(self):
        return self.version

    def init_call(self):
        if self.current:
            self.api.getCurrentVersion(self)

        call = AetherCall(self, self.version, self.api)
        call.init()
        return call

    def get_parameters(self):
        # print("current", self.current)
        if self.current:
            # print("Getting current version data")
            current_version = self.api.getCurrentVersion(self)
            if self.version != current_version:
                self.version = current_version
                self.parameters = self.api.getParameters(self)
        return self.parameters

    def __getitem__(self, item):
        # print("current", self.current)
        if self.current:
            # print("Getting current version data")
            current_version = self.api.getCurrentVersion(self)
            if self.version != current_version:
                self.version = current_version
                self.parameters = self.api.getParameters(self)
        return self.parameters[item]

    def __setitem__(self, item, value):
        # print("current", self.current)
        if self.current:
            # print("Getting current version data")
            current_version = self.api.getCurrentVersion(self)
            if self.version != current_version:
                self.version = current_version
                self.parameters = self.api.getParameters(self)
        self.parameters[item] = value
        self.api.setParameter(self, item, value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def convert_output_schema_to_openai_function_definition(self, output_schema):
        # Remove 'metrics' fields and 'title'
        cleaned_schema = {}

        def clean_schema(schema):
            if isinstance(schema, dict):
                schema = schema.copy()
                schema.pop("metrics", None)
                schema.pop("title", None)
                schema.pop("description", None)
                if "properties" in schema:
                    schema["properties"] = {
                        k: clean_schema(v) for k, v in schema["properties"].items()
                    }
                    schema["additionalProperties"] = False
                if "items" in schema:
                    schema["items"] = clean_schema(schema["items"])
                if "required" not in schema and "properties" in schema:
                    schema["required"] = list(schema["properties"].keys())
            return schema

        cleaned_schema["schema"] = clean_schema(output_schema)
        cleaned_schema["name"] = "output"
        cleaned_schema["strict"] = True
        return cleaned_schema
