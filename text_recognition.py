import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
from fuzzywuzzy import fuzz
import os


os.environ['TESSDATA_PREFIX'] = '/users/eleves-b/2022/edouard.rabasse/tessdata'

def preprocess_image(image_path, save_path):
    with Image.open(image_path) as img:
        img = img.convert('L')
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = img.point(lambda p: 255 if p > 128 else 0)
        img = img.resize([int(dim * 2) for dim in img.size], Image.LANCZOS)
        img = img.filter(ImageFilter.MedianFilter())
        img.save(save_path, format='JPEG')
        return img

def clean_text(text):
    """Remove words with less than 3 letters from the text."""
    words = re.findall(r'\b[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]{3,}\b', text)
    cleaned_text = ' '.join(words)
    return cleaned_text

def extract_text_from_image(preprocessed_image):
    custom_config = r'--oem 3 --psm 6 -l fra'
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    return text if text.strip() else "No text recognised"

def compute_similarity_scores(text, cheese_names):
    """Compute normalized similarity scores for each cheese name given the cleaned text."""
    scores = {}
    for cheese in cheese_names:
        score = fuzz.partial_ratio(text.lower(), cheese.lower()) / 100  # Normalize score between 0 and 1
        scores[cheese] = score

    # Find the cheese with the highest score
    best_cheese = max(scores, key=scores.get)
    best_score = scores[best_cheese]

    # Return the best match only if the score is above 0.8
    if best_score > 0.8:
        return best_cheese, best_score
    else:
        return "No cheese matched", 0

cheese_names = [
    # On enlève le fromage de fromage frais
    "BRIE DE MELUN", "CAMEMBERT", "EPOISSES", "FOURME D’AMBERT", "RACLETTE",
    "MORBIER", "SAINT-NECTAIRE", "POULIGNY SAINT- PIERRE", "ROQUEFORT", "COMTÉ",
    "CHÈVRE", "PECORINO", "NEUFCHATEL", "CHEDDAR", "BÛCHETTE DE CHÈVRE",
    "PARMESAN", "SAINT- FÉLICIEN", "MONT D’OR", "STILTON", "SCARMOZA",
    "CABECOU", "BEAUFORT", "MUNSTER", "CHABICHOU", "TOMME DE VACHE",
    "REBLOCHON", "EMMENTAL", "FETA", "OSSAU- IRATY", "MIMOLETTE",
    "MAROILLES", "GRUYÈRE", "MOTHAIS", "VACHERIN", "MOZZARELLA",
    "TÊTE DE MOINES", "FRAIS"
]

image_path = '6Iz93eMOQvT6oy1 copie.jpg'
save_path = 'processedimage.jpg'
preprocessed_image = preprocess_image(image_path, save_path)
text = extract_text_from_image(preprocessed_image)
cleaned_text = clean_text(text)
print("Extracted Text:", cleaned_text)

best_cheese, best_score = compute_similarity_scores(cleaned_text, cheese_names)

# Print the result
print(f"Most probable cheese: {best_cheese} (Score: {best_score:.2f})")
