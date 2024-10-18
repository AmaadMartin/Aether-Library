<!-- add images/aether_logo_trans.png next to title-->
![Banner](./images/The_Aether_Black.png)
# Aether Python Library

## Installation
To use the Aether library, clone this repo, pip install the requirements, then move Aether.py into your project's root directory.

```bash
pip install -r requirements.txt 
```
to install the required packages. 


## Example Usage

```python
from Aether import AetherClient

# Initialize the Aether client
client = AetherClient(
    api_key=AETHER_API_KEY, 
    openai_api_key=OPENAI_API_KEY
)

# Function ID and input data
function_key = FUNCTION_KEY
input_data = {
    "Article": "This is an article...",
}

# Call the function
output = client(function_key, input_data)
```

Get the API key from the Aether dashboard. Use the function key attached to the function you want to call.