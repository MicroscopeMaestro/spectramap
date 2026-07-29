import pandas as pd
import requests
import json
import io
import re

try:
    from google import genai
except ImportError:
    class _GenAIPlaceholder:
        Client = None
    genai = _GenAIPlaceholder()

def parse_with_gemini(api_key, file_content, file_name, model="gemini-2.5-flash"):
    lines = file_content.decode('utf-8', errors='ignore').split('\n')
    sample = '\n'.join(lines[:100])
    
    prompt = f"""
You are an expert Python data scientist helper.
Your job is to output a JSON object containing a short Python script to parse a spectroscopy data file into a Pandas DataFrame named `df`.

The file name is: '{file_name}'.
The first few lines of the file content are:
---
{sample}
---

Your Python script must:
1. Parse the variable `file_bytes` (which is already provided in the execution context as type `bytes`) using `pd.read_csv` and `io.BytesIO(file_bytes)`.
2. Do NOT define, re-create, or initialize `file_bytes` in your code! Use the existing `file_bytes` variable directly.
3. Automatically detect the correct separator (e.g., '|', ',', '\\t') and skip any metadata headers at the top by setting the correct `skiprows` integer.
4. Name the final DataFrame `df`.
5. The final `df` must have spectral intensities in the columns (numeric float column names) and optionally 'label', 'x', 'y', 'z'.
6. Rename columns like 'sample_id' to 'label', 'x_coord' to 'x', etc. if needed.

Your response must be a single JSON object matching the format below.
Do NOT output any markdown, conversational text, explanations, or any other keys.
Format:
{{
  "code": "import pandas as pd\\nimport io\\ndf = pd.read_csv(io.BytesIO(file_bytes), sep='|', skiprows=6)\\ndf.rename(columns={{'sample_id': 'label', 'x_coord': 'x', 'y_coord': 'y'}}, inplace=True)"
}}
"""
    if genai is None or getattr(genai, 'Client', None) is None:
        raise ImportError("google-genai library is not installed or available.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    result_text = getattr(response, 'text', str(response))
    
    json_str = result_text.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
        
    result = json.loads(json_str)
    code = result['code']
    
    local_vars = {'file_bytes': file_content}
    exec(code, {}, local_vars)
    df = local_vars.get('df')
    
    if df is None or not isinstance(df, pd.DataFrame):
        raise Exception("The generated code did not produce a pandas DataFrame named 'df'.")
        
    return df, code


def parse_with_ollama(file_content, file_name, model="gemma4:e2b"):
    """
    Uses Ollama to determine how to parse a custom data file into a Pandas DataFrame.
    It asks for a python code snippet that takes `file_content` (bytes) and returns a clean dataframe `df`.
    The `df` must have:
    - `label` column (optional)
    - `x`, `y`, `z` columns (optional)
    - the rest of the columns must be the spectral intensities (numeric wavenumbers as column names).
    """
    lines = file_content.decode('utf-8', errors='ignore').split('\n')
    sample = '\n'.join(lines[:100])
    
    prompt = f"""
You are an expert Python data scientist helper.
Your job is to output a JSON object containing a short Python script to parse a spectroscopy data file into a Pandas DataFrame named `df`.

The file name is: '{file_name}'.
The first few lines of the file content are:
---
{sample}
---

Your Python script must:
1. Parse the variable `file_bytes` (which is already provided in the execution context as type `bytes`) using `pd.read_csv` and `io.BytesIO(file_bytes)`.
2. Do NOT define, re-create, or initialize `file_bytes` in your code! Use the existing `file_bytes` variable directly.
3. Automatically detect the correct separator (e.g., '|', ',', '\\t') and skip any metadata headers at the top by setting the correct `skiprows` integer.
4. Name the final DataFrame `df`.
5. The final `df` must have spectral intensities in the columns (numeric float column names) and optionally 'label', 'x', 'y', 'z'.
6. Rename columns like 'sample_id' to 'label', 'x_coord' to 'x', etc. if needed.

Your response must be a single JSON object matching the format below.
Do NOT output any markdown, conversational text, explanations, or any other keys.
Format:
{{
  "code": "import pandas as pd\\nimport io\\ndf = pd.read_csv(io.BytesIO(file_bytes), sep='|', skiprows=6)\\ndf.rename(columns={{'sample_id': 'label', 'x_coord': 'x', 'y_coord': 'y'}}, inplace=True)"
}}
"""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=10
        )
        response.raise_for_status()
        result_text = response.json().get('response', '')
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
        raise requests.exceptions.ConnectionError(f"Ollama service is offline or unreachable at http://localhost:11434: {conn_err}")
    except Exception as e:
        raise Exception(f"Failed to connect to Ollama or receive response. Error: {e}")
    
    try:
        json_str = result_text.strip()
        # Clean up markdown if model still included it despite "format": "json"
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
            
        result = json.loads(json_str)
        code = result['code']
        
        # Execute the code safely
        local_vars = {'file_bytes': file_content}
        exec(code, {}, local_vars)
        df = local_vars.get('df')
        
        if df is None or not isinstance(df, pd.DataFrame):
            raise Exception("The generated code did not produce a pandas DataFrame named 'df'.")
            
        return df, code
    except Exception as e:
        raise Exception(f"Failed to parse with AI.\\nAI Response: {result_text}\\nError: {e}")
