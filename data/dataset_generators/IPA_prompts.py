from pathlib import Path
from tqdm import tqdm
import yaml


class IPAPromptsDatasetGenerator:
    def __init__(
        self,
        generator,
        cheese_description,
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
        self.cheese_description = cheese_description

    def generate(self, labels_names):
        labels_prompts = self.create_prompts(labels_names)
        for label, label_prompts in labels_prompts.items():
            #mettre image_id_0 à la plus grande valeur déjà existante dans le dossier
            image_id_0 = 200
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

        contextes = {
            "sur une assiette blanche usée, avec un couteau et des miettes de pain, au centre d'une table de cuisine rustique, avec de la charcuterie": "on a white plate, with a knife, on a rustic kitchen table, with charcuterie",
            "découpé sur une planche en bois, dans un style promotionnel": "cut on a wooden board, in a promotional style",
            "dehors, sur un étalage de marché en bois, entouré d'autres fromages, dans un style rustique": "outside, on a wooden market stall, surrounded by other cheeses, in a rustic style",
            "dans une assiette en céramique, avec des fruits et des noix, dans un style minimaliste": "on a ceramic plate, with fruits and nuts, in a minimalist style",
            "dans une assiette émaillée sur du pain frais": "on an enamel plate with fresh bread",
            "dans une boîte en bois avec étiquette claire": "in a wooden box with a label",
            "dans un emballage en papier avec une étiquette foncée dans un style commercial": "in a paper wrapping with a label in a commercial style",
            "dans un emballage en plastique, avec une partie sortie": "in a plastic wrapping, with a part exposed",
        }

        fonds = {
            "sur fond blanc": "on a white background",
            "sur fond noir": "on a black background",
            "dans un paysage naturel de campagne, avec un ciel avec quelques nuages": "in a natural countryside landscape",
            "dans une cuisine rustique, avec des ustensiles en bois et des pots en terre cuite": "in a rustic kitchen, with wooden utensils and terracotta pots",
            "dans un marché en plein air, avec des étals en bois et des parasols colorés": "in an outdoor market, with wooden stalls",
            "dans une pièce éclairée, avec un sol carrelé légèrement usé": "in a well-lit room, with a slightly worn tiled floor",
            "contre un vieux mur de briques avec des traces d'usure": "against a brick wall, with signs of wear",
        }


        cadrages = {
            "en gros plan": "close-up",
            "avec un cadrage large": "wide",
        } 
        cheese_prompts = {}

        # On génère les prompts pour chaque fromage : on a  ?? prompts générés par fromage
        

        for _, context in contextes.items(): 
            for _,fond in fonds.items():
                for _,cadrage in cadrages.items():
                    prompt = f"A {cadrage} realistic image with very precise details and natural colours, {context}, {fond}."
                    cheese_prompts.append({"prompt": prompt})

        return cheese_prompts

    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
