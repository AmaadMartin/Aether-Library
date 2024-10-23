import os
import requests
import aiohttp

BASE_URL = "https://aether-ty31.onrender.com"


class AetherAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def getParameters(self, function):
        headers = {"X-API-Key": self.api_key}
        response = requests.get(
            f"{BASE_URL}/function_params/{function.function_key}/{function.version}",
            headers=headers,
        )
        if response.status_code != 200:
            raise Exception(f"Error retrieving parameters: {response.text}")
        parameters = response.json()["parameters"]
        return parameters

    def getParameter(self, function, parameter):
        headers = {"X-API-Key": self.api_key}
        response = requests.get(
            f"{BASE_URL}/function_param/{function.function_key}/{parameter}/{function.version}",
            headers=headers,
        )
        if response.status_code != 200:
            raise Exception(f"Error retrieving parameter: {response.text}")
        parameter = response.json()["parameter"]
        return parameter

    def setParameter(self, function, parameter, value):
        headers = {"X-API-Key": self.api_key}
        response = requests.post(
            f"{BASE_URL}/function_param/{function.function_key}/{parameter}/{function.version}",
            headers=headers,
            json={"value": value},
        )
        if response.status_code != 200:
            raise Exception(f"Error setting parameter: {response.text}")
        response_json = response.json()
        if response_json["new_parameter"] == True:
            params = response_json["params"]
            self.updateVersionTree(function, function.version, params)
        return response.json()

    def getCurrentVersion(self, function):
        headers = {"X-API-Key": self.api_key}
        response = requests.get(
            f"{BASE_URL}/function_data/{function.function_key}", headers=headers
        )
        if response.status_code != 200:
            raise Exception(f"Error retrieving function data: {response.text}")
        function_data = response.json()
        return function_data["current_version"]

    def createCall(self, function, version, inputs, outputs, logs):
        headers = {"X-API-Key": self.api_key}
        payload = {"inputs": inputs, "outputs": outputs, "logs": logs or []}
        response = requests.post(
            f"{BASE_URL}/create_call/{function.function_key}/{version}",
            headers=headers,
            json=payload,
        )
        if response.status_code != 200:
            raise Exception(f"Error creating call: {response.text}")
        call_key = response.json()["call_key"]
        return call_key

    async def updateCall(self, call):
        headers = {"X-API-Key": self.api_key}
        payload = {
            "inputs": call.inputs,
            "outputs": call.outputs,
            "logs": call.logs,
            "status": call._status,
            "evaluation": call.evaluation,
            "function_key": call.function_key,
            "version": call.version,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/update_call/{call.call_key}", headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Error updating call: {response.text}")
                return await response.json()

    def sync_updateCall(self, call):
        headers = {"X-API-Key": self.api_key}
        payload = {
            "inputs": call.inputs,
            "outputs": call.outputs,
            "logs": call.logs,
            "status": call._status,
            "evaluation": call.evaluation,
        }
        response = requests.post(
            f"{BASE_URL}/update_call/{call.call_key}", headers=headers, json=payload
        )
        if response.status_code != 200:
            raise Exception(f"Error updating call: {response.text}")
        return response.json()

    async def evaluateCall(self, call):
        headers = {"X-API-Key": self.api_key}
        payload = {
            "inputs": call.inputs,
            "outputs": call.outputs,
            "logs": call.logs,
            "status": call._status,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/evaluate_call/{call.function_key}/{call.version}/{call.call_key}",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"Error evaluating call: {response.text}")
                evaluation = response.json()["evaluation"]
                return evaluation

    def getFunctionData(self, function_key):
        headers = {"X-API-Key": self.api_key}
        response = requests.get(
            f"{BASE_URL}/function_data/{function_key}", headers=headers
        )
        if response.status_code != 200:
            raise Exception(f"Error retrieving function data: {response.text}")
        function_data = response.json()
        return function_data

    async def updateVersionTree(self, function, parent_version, params):
        headers = {"X-API-Key": self.api_key}
        response = requests.post(
            f"{BASE_URL}/update_version_tree/{function.function_key}/{parent_version}",
            headers=headers,
            json={"params": params},
        )
        if response.status_code != 200:
            raise Exception(f"Error updating version tree: {response.text}")
        return response.json()
