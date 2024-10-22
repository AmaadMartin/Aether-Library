# library/Aether.py
from aetherllm._Aether.AetherAPI import AetherAPI
from aetherllm._Aether.AetherFunction import AetherFunction


class Aether:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api = AetherAPI(api_key)

    def __call__(self, function_key, version=None, openai_key=None):
        function = AetherFunction(function_key, self.api, version, openai_key)
        function.init()
        return function
