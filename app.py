from preprocess import preprocess_image
from ocr import extract_text
import cv2

# Put an image in your folder named test.jpg
image_path = "test.jpg"

processed = preprocess_image(image_path)

text = extract_text(processed)

print("Extracted Text:")
print(text)

# Optional: show image
cv2.imshow("Processed", processed)
cv2.waitKey(0)
cv2.destroyAllWindows()