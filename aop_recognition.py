import os
import cv2
import numpy as np

def preprocess_image(image):
    """Convert to grayscale and apply histogram equalization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def resize_template(image, template, scale_factor):
    """Resize the template based on the scale factor."""
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    resized_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_template

def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def detect_aop_logo(image_path, template, threshold=0.8, scale_factors=[0.1, 0.15], angles=[0]):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return False

    gray_image = preprocess_image(image)

    for scale in scale_factors:
        resized_template = resize_template(gray_image, template, scale)
        for angle in angles:
            rotated_template = rotate_image(resized_template, angle)
            w, h = rotated_template.shape[::-1]

            result = cv2.matchTemplate(gray_image, rotated_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold)
            if len(loc[0]) > 0:
                return True
    return False

# Charger le modèle de logo AOP
current_dir = os.path.dirname(__file__)
template_path = os.path.join(current_dir, 'AOP.jpg')  # Assurez-vous que 'AOP.jpg' est dans le même répertoire que ce script
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise ValueError("Template image not found. Check the path.")

val_dir = "/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/dataset/val/"
logo_counts = {}

# Paramètres de détection
threshold = 0.85
scale_factors = [0.05, 0.1, 0.2]
angles = [0, 15, -15, 30, -30]

# Parcourir le dataset et effectuer la détection
for cheese_type in os.listdir(val_dir):
    cheese_dir = os.path.join(val_dir, cheese_type)
    logo_count = 0
    for image_name in os.listdir(cheese_dir):
        image_path = os.path.join(cheese_dir, image_name)
        detected = detect_aop_logo(image_path, template, threshold, scale_factors, angles)

        if detected:
            logo_count += 1

    logo_counts[cheese_type] = logo_count

# Afficher le nombre de logos AOP trouvés par type de fromage
for cheese_type, count in logo_counts.items():
    print(f"{cheese_type}: {count} logo(s) AOP trouvé(s)")
