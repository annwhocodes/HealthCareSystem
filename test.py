import google.generativeai as genai

# Initialize Gemini
genai.configure(api_key="AIzaSyCTgksUs2hzuuk8wyKv_xHBtaVGHDpo6II")

# List available models
models = genai.list_models()
for model in models:
    print(model.name)