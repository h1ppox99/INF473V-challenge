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

        fromages = [
            "BRIE DE MELUN", "CAMEMBERT", "EPOISSES", "FOURME D’AMBERT", "RACLETTE",
            "MORBIER", "SAINT-NECTAIRE", "POULIGNY SAINT-PIERRE", "ROQUEFORT", "COMTÉ",
            "CHÈVRE", "PECORINO", "NEUFCHATEL", "CHEDDAR", "BÛCHETTE DE CHÈVRE",
            "PARMESAN", "SAINT-FÉLICIEN", "MONT D’OR", "STILTON", "SCARMOZA", "CABECOU",
            "BEAUFORT", "MUNSTER", "CHABICHOU", "TOMME DE VACHE", "REBLOCHON", "EMMENTAL",
            "FETA", "OSSAU-IRATY", "MIMOLETTE", "MAROILLES", "GRUYÈRE", "MOTHAIS",
            "VACHERIN", "MOZZARELLA", "TÊTE DE MOINES", "FROMAGE FRAIS"
        ]

        cheese_prompts = {}
        for adjectif in adjectifs:
            for fromage in fromages:
                cheese_prompts[fromage] = []
                for context_fr, context_en in contextes.items():
                    prompt = f"An image of {adjectif} {fromage} cheese {context_en}"
                    cheese_prompts[fromage].append({"prompt": prompt, "num_images": 1})



        return cheese_prompts

    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
