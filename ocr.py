import easyocr

# Create reader
reader = easyocr.Reader(['en'])

def extract_text(image):

    result = reader.readtext(image)

    extracted_text = ""

    for item in result:
        extracted_text += item[1] + " "

    return extracted_text