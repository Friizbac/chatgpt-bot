# Import necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Set device (e.g. CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the GPT model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the maximum input length and prompt for the chatbot
MAX_LENGTH = 1024
prompt = "What is better php or python?"

# Pre-process the input and generate a response
encoded_prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)
attention_mask = torch.ones(encoded_prompt.shape, dtype=torch.long, device=device)
response = model.generate(encoded_prompt, attention_mask=attention_mask, max_length=MAX_LENGTH, top_p=0.9, top_k=0)
response = response.tolist()[0]
decoded_response = tokenizer.decode(response, skip_special_tokens=True)

# Return the response to the user
print(decoded_response)