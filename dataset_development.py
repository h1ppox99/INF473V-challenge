import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

# Liste des fromages
fromages = [
    "BRIE DE MELUN", "CAMEMBERT", "EPOISSES", "FOURME D’AMBERT", "RACLETTE", 
    "MORBIER", "SAINT-NECTAIRE", "POULIGNY SAINT- PIERRE", "ROQUEFORT", "COMTÉ", 
    "CHÈVRE", "PECORINO", "NEUFCHATEL", "CHEDDAR", "BÛCHETTE DE CHÈVRE", 
    "PARMESAN", "SAINT- FÉLICIEN", "MONT D’OR", "STILTON", "SCARMOZA", "CABECOU", 
    "BEAUFORT", "MUNSTER", "CHABICHOU", "TOMME DE VACHE", "REBLOCHON", "EMMENTAL", 
    "FETA", "OSSAU- IRATY", "MIMOLETTE", "MAROILLES", "GRUYÈRE", "MOTHAIS", 
    "VACHERIN", "MOZZARELLA", "TÊTE DE MOINES", "FROMAGE FRAIS"
]

# Fonction pour scraper les images
def scrape_images(fromage, num_images=100):
    query = f"Packaged OR wrapped {fromage}"
    url = f"https://www.google.com/search?q={query}&source=lnms&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    image_urls = []
    for img in soup.find_all("img", {"src": re.compile("gstatic.com")}):
        if len(image_urls) < num_images:
            image_urls.append(img['src'])
        else:
            break

    save_images(fromage, image_urls)

# Fonction pour sauvegarder les images
def save_images(fromage, image_urls):
    directory = f"dataset/val2/{fromage.replace(' ', '_')}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, url in enumerate(image_urls):
        response = requests.get(url)
        with open(f"{directory}/{fromage.replace(' ', '_')}_{i + 1}.jpg", 'wb') as f:
            f.write(response.content)
    print(f"Downloaded {len(image_urls)} images of {fromage}")

# Scraper les images pour chaque fromage
for fromage in fromages:
    scrape_images(fromage)
