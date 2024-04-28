from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_summarization_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")  # Replace with your model
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")  # Replace with your model
    return tokenizer, model
