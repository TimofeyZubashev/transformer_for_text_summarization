from transformers import AutoTokenizer

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
print(f"Tokeizer vocab size = {tokenizer.vocab_size} tokens")