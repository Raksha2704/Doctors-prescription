from preprocess import preprocess_image
from ocr import extract_text

from rapidfuzz import process

# Medicine database
medicine_list = [
    "aceta",
    "alatrol",
    "esonix",
    "trilock"
]

# Image path
image_path = "test.jpg"

# Preprocess
processed = preprocess_image(image_path)

# OCR
text = extract_text(processed)

print("Raw OCR Output:")
print(text)

# Fuzzy matching
match = process.extractOne(
    text,
    medicine_list
)

print("\nMatched Medicine:")
print(match[0])