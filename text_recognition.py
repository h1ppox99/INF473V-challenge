from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

def preprocess_image(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert the image to grayscale
        img = img.convert('L')
        # Enhance the contrast of the image
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        # Apply some additional filters if necessary
        img = img.filter(ImageFilter.SHARPEN)
        
        return img

def extract_text_from_image(preprocessed_image):
    # Use pytesseract to do OCR on the preprocessed image
    text = pytesseract.image_to_string(preprocessed_image)
    return text

# Example usage
image_path = 'path_to_your_cheese_label_image.jpg'
preprocessed_image = preprocess_image(image_path)
text = extract_text_from_image(preprocessed_image)
print("Extracted Text:", text)
