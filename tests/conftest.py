import os

os.environ.pop("OPENAI_API_KEY", None)
os.environ["EMBEDDING_BACKEND"] = "hashed"
os.environ["LLM_PROVIDER"] = "none"
