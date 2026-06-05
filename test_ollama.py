import smart_importer

with open('data/messy_dataset.txt', 'rb') as f:
    content = f.read()

print("Calling Ollama...")
try:
    df, code = smart_importer.parse_with_ollama(content, 'messy_dataset.txt')
    print("Success!")
    print("Generated Code:\\n", code)
    print("Parsed Dataframe Shape:", df.shape)
    print(df.head())
except Exception as e:
    print("Failed!")
    print(str(e))
