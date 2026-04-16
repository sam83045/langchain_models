import langchain
import torch

print("LangChain version:", langchain.__version__)
print(torch.cuda.get_device_name(torch.cuda.current_device()))