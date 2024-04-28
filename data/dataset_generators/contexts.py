from pathlib import Path
from tqdm import tqdm


class ContextsPromptsDatasetGenerator:
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
    ):
        """
        Args:
            generator: image generator object
            batch_size: Number of images to generate per batch. Make sure to max out your GPU VRAM for maximum efficiency
            output_dir: Directory where the generated images will be saved
        """
        self.generator = generator
        self.batch_size = batch_size
        self.output_dir = output_dir

    def generate(self, labels_names):
        labels_prompts = self.create_prompts(labels_names)
        for label, label_prompts in labels_prompts.items():
            image_id_0 = 0
            for prompt_metadata in label_prompts:
                num_images_per_prompt = prompt_metadata["num_images"]
                prompt = [prompt_metadata["prompt"]] * num_images_per_prompt
                pbar = tqdm(range(0, num_images_per_prompt, self.batch_size))
                pbar.set_description(
                    f"Generating images for prompt: {prompt_metadata['prompt']}"
                )
                for i in range(0, num_images_per_prompt, self.batch_size):
                    batch = prompt[i : i + self.batch_size]
                    images = self.generator.generate(batch)
                    self.save_images(images, label, image_id_0)
                    image_id_0 += len(images)
                    pbar.update(1)
                pbar.close()

    def create_prompts(self, labels_names):
        """
        Prompts should be a dictionary with the following structure:
        {
            label_0: [
                {
                    "prompt": "Prompt for label_0",
                    "num_images": 100
                },
                {
                    "prompt": "Another prompt for label_0",
                    "num_images": 200
                }
            ],
            label_1: [
                {
                    "prompt": "Prompt for label_1",
                    "num_images": 100
                }
            ]
        }
        """
        adjectifs = {"crémeux": "creamy", "frais": "fresh", "affiné": "aged", "doux": "mild", "fort": "strong", 
        }

        contextes = {
            "sur une planche en bois": "on a wooden board",
            "avec des olives": "with olives",
            "avec du vin": "with a glass of wine",
            "sur une table rustique": "on a rustic table",
            "avec du pain frais": "with fresh bread",
            "dans une boîte en bois avec étiquette": "in a wooden box with a label",
            "sur fond uni": "on a plain background",
            "dans un emballage en papier": "in a paper wrapping",
            "dans un emballage en plastique": "in a plastic wrapping",
        }

        lumieres = {
            "avec une lumière naturelle": "under a natural light",
            "avec une lumière artificielle": "under a artificial light",
            "avec un éclairage modulé": "in modulated lighting",
            "avec un éclairage scénique": "in stage lighting",
            "avec une lumière contrôlée": "under a controlled light",
        }

        fonds = {
            "sur fond blanc": "on a white background",
            "sur fond noir": "on a black background",
            "dans un paysage naturel": "in a natural landscape",
            "avec un horizon dégagé": "with a clear horizon",
            "dans une pièce éclairée": "in a well-lit room",
            "contre un mur de briques": "against a brick wall"
        }

        styles_visuels = {
            "Minimaliste": "minimalist",
            "Vintage": "vintage",
            "Rustique": "rustic",
            "Promotionnel": "promotional",
            "Commercial": "commercial",
            "Décontracté": "casual"
        }



        cheese_prompts = {}
        
        for fromage in labels_names:
            cheese_prompts[fromage] = []
            for _, context in contextes.items(): 
                for _,adjectif in adjectifs.items():
                    for _,style in styles_visuels.items():
                        for _,fond in fonds.items():
                            for _,lumiere in lumieres.items():
                                prompt = f"An image of {adjectif} {fromage} cheese {context} {fond} in a {style} style {lumiere}"
                                cheese_prompts[fromage].append({"prompt": prompt, "num_images": 1})


        return cheese_prompts

    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
