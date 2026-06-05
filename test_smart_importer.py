import pandas as pd
import io
import smart_importer
from spectramap import spmap as sp
import numpy as np

# Mocking genai
class MockResponse:
    text = '''{
      "code": "import pandas as pd\\nimport io\\ndf = pd.read_csv(io.BytesIO(file_bytes), sep='|', skiprows=1, header=None)\\ndf.columns = ['label', 'x', 'y'] + [400.0, 500.0, 600.0]"
    }'''

class MockModels:
    def generate_content(self, model, contents):
        return MockResponse()

class MockClient:
    def __init__(self, api_key):
        self.models = MockModels()

smart_importer.genai.Client = MockClient

# Create a weird data file content
weird_data = b"""WEIRD DATA HEADER DO NOT READ
Class_A|1|2|100.5|200.5|300.5
Class_B|1|3|110.5|210.5|310.5
"""

df, code = smart_importer.parse_with_gemini("dummy_key", weird_data, "weird.txt")
print("Dataframe created successfully with shape:", df.shape)
print(df)

# Now test spectramap integration logic from app.py
obj = sp.hyper_object('weird', data_type='multi_spectra')
obj.data = df.drop(columns=['label', 'x', 'y', 'z'], errors='ignore')
if 'label' in df.columns:
    obj.label = pd.Series(df['label'])
else:
    obj.label = pd.Series([1]*len(df))

if 'x' in df.columns and 'y' in df.columns:
    if 'z' in df.columns:
        obj.position = df[['x', 'y', 'z']]
    else:
        obj.position = df[['x', 'y']]
else:
    obj.position = pd.DataFrame({'x': np.arange(len(df)), 'y': np.zeros(len(df))})

obj.m = int(pd.to_numeric(obj.position['x']).max() + 1) if 'x' in obj.position else len(df)
obj.n = int(pd.to_numeric(obj.position['y']).max() + 1) if 'y' in obj.position else 1
obj.resolution = 1
obj.sublabel = pd.Series(np.zeros(len(obj.data)), name="sublabel")

print("SpectraMap object created successfully!")
print("m:", obj.m, "n:", obj.n, "data shape:", obj.data.shape, "position shape:", obj.position.shape)
