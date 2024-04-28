from django.shortcuts import render
from .models import load_summarization_model  # Import the model loader function
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM #, max_new_tokens

from django import forms

class SummarizeTextForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={'rows': 10, 'class': 'form-control'}), label="Enter text to summarize:")


def summarize_text(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        tokenizer, model = load_summarization_model()  # Load the model here
        # Implement summarization logic using the model and tokenizer
        summary = summarize_with_model(text, tokenizer, model)  # Replace with your summarization function
        return render(request, 'summarizer/summary.html', {'summary': summary})
    return render(request, 'summarizer/summarizer_form.html', context = {'form': SummarizeTextForm()})

# Function to summarize text using the loaded model (replace with your implementation)
def summarize_with_model(text, tokenizer, model):
    """
    Preprocesses text, generates a summary using the model, and postprocesses the output.

    Args:
        text (str): The text to be summarized.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        model (transformers.PreTrainedModel): The summarization model.

    Returns:
        str: The generated summary of the text.
    """

    # Preprocess text (e.g., tokenization, handling special characters)
    inputs = tokenizer(text, return_tensors="pt")  # Convert text to model-compatible tensors

    # Generate summary using the model
    # outputs = model.generate(**inputs)  # Generate summary based on model's internal logic
    outputs = model.generate(**inputs, max_new_tokens=20000)
    decoded_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  # Decode generated tokens back to text

    # Postprocess summary (e.g., removing extra whitespace, length trimming)
    summary = decoded_text.strip()  # Remove leading/trailing whitespace

    return summary
