import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, render_template
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize the Flask application
app = Flask(__name__)

# Load the model and tokenizer using pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Function to generate the story
def generate_story(prompt, max_length=1000):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

# Route for the HTML form and story generation
@app.route('/', methods=['GET', 'POST'])
def index():
    story = None
    error_message = None

    if request.method == 'POST':
        # Get the prompt from the form
        prompt = request.form.get('prompt')

        if not prompt:
            error_message = "Prompt is required!"
        else:
            # Generate the story
            story = generate_story(prompt)

    return render_template('index.html', story=story, error_message=error_message)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
