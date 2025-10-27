import sys
sys.path.append("venv/lib/python3.11/site-packages")
import tiktoken
print(tiktoken.__version__)
enc=tiktoken.encoding_for_model("gpt-4o")
text="My name is utpal"
token=enc.encode(text)
print("Token",token)
decoded=enc.decode([5444, 1308, 382, 4518, 18413])
print("Decoded ",decoded)