<!-- add images/aether_logo_trans.png next to title-->
![Banner](./aetherllm/images/The_Aether_Black.png)
# Aether Python Library

## Installation
```bash
pip install aetherllm
```


## Example Usage
Get your API key from the bottom right of the Aether dashboard. Use the function key attached to the function you want to call.

### Flow
```python
from aetherllm import Aether

# Initialize the Aether client
aether = Aether(AETHER_API_KEY)

# Initialize the flow
flow = aether(FLOW_KEY)


def custom_function(input_json):
    # Initialize the call
    call = flow.init_call()

    call.log("Running custom function")
    call.input("input", input_json)
    
    prompt = flow['prompt']
    output = "this is the prompt: " + prompt

    call.output("output", output)
    call.status("completed")

    call.eval()
    return output

custom_function("test_input")
```

### Function
```python
from aetherllm import Aether

# Initialize the Aether client
aether = Aether(AETHER_API_KEY)

# Initialize the function
function = aether(FUNCTION_KEY, openai_key=OPENAI_API_KEY)

# Call the function
output = function({
    "Article": "This is an article...",
})
```