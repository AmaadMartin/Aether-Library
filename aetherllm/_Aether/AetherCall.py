# library/_Aether/AetherCall.py
import datetime
import asyncio
import threading


class AetherCall:
    def __init__(self, function, version, api):
        # call api to retrieve call data
        self.api = api
        self.function = function
        self.function_key = function.function_key
        self.version = version
        self.inputs = {}
        self.outputs = {}
        self.logs = []
        self.evaluation = {}
        self._status = "pending"

    def init(self):
        self.call_key = self.api.createCall(
            self.function,
            self.version,
            self.inputs,
            self.outputs,
            self.logs,
        )

    def status(self, _status):
        self._status = _status
        self.logs.append(
            {
                "status": _status,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        # call api to update call data
        self.update()

    def log(self, log):
        if type(log) == str:
            log = {
                "message": log,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        if "timestamp" not in log:
            log["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(log)
        # call api to update call data
        self.update()

    def eval(self):
        self._status = "evaluating"
        thread = threading.Thread(target=self._run_async_eval)
        thread.start()
        return

    def input(self, input_name, input_data):
        self.inputs[input_name] = input_data
        # call api to update call data
        self.update()

    def output(self, output_name, output_data):
        self.outputs[output_name] = output_data
        # call api to update call data
        self.update()

    def update(self):
        thread = threading.Thread(target=self._run_async_update)
        thread.start()
        return

    def _run_async_update(self):
        try:
            asyncio.run(self.api.updateCall(self))
        except Exception as e:
            return

    def _run_async_eval(self):
        try:
            asyncio.run(self.api.evaluateCall(self))
        except Exception as e:
            return

    def __str__(self):
        return self.call_key

    def __repr__(self):
        return self.call_key
