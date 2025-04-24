from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pickle

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate story
def generate_story(prompt, max_length=1000):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

with open('model.pkl','wb') as file:
  pickle.dump(model,file)

with open('tokenizer.pkl','wb') as file:
  pickle.dump(tokenizer,file)