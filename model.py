from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)


def clean_caption(caption: str) -> str:
    """Remove overly repetitive words or meaningless patterns."""
    words = caption.split()
    # Remove consecutive duplicates
    cleaned_words = []
    for word in words:
        if len(cleaned_words) == 0 or cleaned_words[-1].lower() != word.lower():
            cleaned_words.append(word)
    caption = " ".join(cleaned_words)

    # If caption is too repetitive (e.g., <50% unique words)
    unique_ratio = len(set(cleaned_words)) / (len(cleaned_words) + 1e-6)
    if unique_ratio < 0.5:
        return "Unable to generate a meaningful caption."
    return caption.capitalize()


def generate_captions(image_path: str, num_captions: int = 3):
    """
    Generate multiple high-quality captions with repetition control.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    # Generate captions with beam search and repetition constraints
    out = model.generate(
        **inputs,
        num_beams=6,  # better beam search
        num_return_sequences=num_captions,
        max_length=40,
        repetition_penalty=2.5,  # strong repetition control
        no_repeat_ngram_size=3,  # prevent repeating 3-word sequences
        length_penalty=1.2  # prefer longer, more descriptive captions
    )

    # Clean and return captions
    return [clean_caption(processor.decode(o, skip_special_tokens=True)) for o in out]
