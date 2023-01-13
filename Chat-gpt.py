from transformers import AutoModelWithLMHead, AutoTokenizer

# Load the model and tokenizer
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

def generate_response(prompt):
  # Encode the prompt and generate a response
  input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
  response = model.generate(input_ids, max_length=1000, temperature=0.7)
  response_text = tokenizer.decode(response[0], skip_special_tokens=True)

  # Remove the prompt from the response
  prompt_len = len(tokenizer.encode(prompt, return_tensors='pt')[0])
  response_text = response_text[prompt_len:]

  return response_text

# Test the chatbot
while True:
    question = input("You: ")
    if question == "exit":
        break
    answer = generate_response(question)
    print(f"Bot: {answer}")
