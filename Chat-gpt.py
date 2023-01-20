from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

# Önceden eğitilmiş model ve tokenizer'i yükle
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")

def generate_response(prompt):
    # Modeline girdi olarak prompt'u kodla
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    # Cevap üret
    response = model.generate(input_ids, max_length=1024)
    # Cevabı kodlamadan döndür
    return tokenizer.decode(response[0], skip_special_tokens=True)

# Örnek kullanım
prompt = "Merhaba, bugün size nasıl yardımcı olabilirim?"
response = generate_response(prompt)
print(response)
