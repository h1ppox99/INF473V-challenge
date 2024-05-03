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


        fromages = {
            "BRIE DE MELUN": "Brie de Melun, very flat form, with a textured shaded beige rind, with a soft creamy ivory interior,",
            # "CAMEMBERT": "Camembert, round and flat, with a soft, creamy interior beneath a bloomy, velvety white rind, slightly dimpled with age,",
            # "ÉPOISSES": "Époisses, round, slightly bulging center, with a vibrant and sticky red-orange wash rind and a lusciously soft, golden interior,",
            # "FOURME D’AMBERT": "Fourme d'Ambert, tall cylindrical shape, with a smooth, pale gray rind and a creamy, blue-marbled interior,",
            # "RACLETTE": "Raclette, large wheel, semi-hard with a thin, smooth golden-brown rind and a pale yellow, elastic interior,",
            # "MORBIER": "Morbier, round with a dark ash layer through its creamy, pale yellow middle, encased in a bloomy rind,",
            # "SAINT-NECTAIRE": "Saint-Nectaire, a round and slightly flattened cheese, with a dusky, velvety rind and a supple, creamy, light beige interior,",
            # "POULIGNY SAINT-PIERRE": "Pouligny Saint-Pierre, pyramidal cheese with a wrinkled, pale rind and a firm, creamy-white core,",
            # "ROQUEFORT": "Roquefort, rectangular block, with a natural, craggy rind and a rich, blue-veined, moist interior,",
            # "COMTÉ": "Comté, large wheel, displaying a smooth, golden rind with a dense, pale yellow interior marked by occasional crystal formations,",
            # "CHÈVRE": "Chèvre, various shapes including logs, discs, and pyramids, with a thin, natural rind enveloping a soft, bright white, creamy center,",
            # "PECORINO": "Pecorino, round or drum-shaped, featuring a hard, pale rind and a dense, granular, off-white interior,",
            # "NEUFCHATEL": "Neufchâtel, heart-shaped with a soft, velvety rind shaping a creamy, slightly grainy heart,",
            # "CHEDDAR": "Cheddar, block form, with a firm, natural rind and a deep orange, slightly crumbly interior,",
            # "BÛCHETTE DE CHÈVRE": "Bûchette de Chèvre, small log shape with a delicate, bloomy rind and a soft, creamy, bright white center,",
            # "PARMESAN": "Parmesan, large wheel, known for its thick, hard, golden rind and rich, granular, pale yellow interior,",
            # "SAINT-FÉLICIEN": "Saint-Félicien, round and flat, featuring a soft, bloomy rind encasing a lush, buttery, pale cream center,",
            # "MONT D’OR": "Mont d'Or, round, encased in spruce bark with a soft, rippling, golden rind and a rich, creamy interior,",
            # "STILTON": "Stilton, cylindrical, with a rough, natural rind and a richly veined, semi-soft, blue-speckled interior,",
            # "SCARMOZA": "Scamorza, pear-shaped, sporting a smooth, thin rind with a dense, chewy, pale interior,",
            # "CABÉCOU": "Cabécou, small and round cheese with a thin, wrinkled rind and a creamy, soft interior,",
            # "BEAUFORT": "Beaufort, a grand, round cheese, with sturdy, golden-brown rind and a firm, dense rind, vibrant pale yellow interior with a uniform, creamy texture, showcasing subtle tiny holes,",
            # "MUNSTER": "Munster, round, with a glossy, orange rind and a soft, creamy, pale yellow interior,",
            # "CHABICHOU": "Chabichou, cylindrical with a smooth, thin rind and a dense, creamy-white interior,",
            # "TOMME DE VACHE": "Tomme de Vache, round or slightly oval, with a thick, rough, gray rind and a firm, buttery, light beige interior,",
            # "REBLOCHON": "Reblochon, round, featuring a velvety, orange rind with a soft, creamy, pale yellow core,",
            # "EMMENTAL": "Emmental cheese, large wheel, noted for its smooth, hard, pale yellow rind and a pale yellow interior punctuated by large holes,",
            # "FETA": "Feta, block or crumbled, creamy white interior, often without a rind,",
            # "OSSAU-IRATY": "Ossau-Iraty cheese, round or slightly oval, with a hard, dry, orange rind and a dense, creamy, ivory interior,",
            # "MIMOLETTE": "Mimolette, spherical, distinguished by its bright orange, hard form with a dense, crumbly interior,",
            # "MAROILLES": "Maroilles cheese, square, with a moist, orange rind and a soft, sticky, deep yellow interior,",
            # "GRUYÈRE": "Gruyère, large wheel, with a smooth, brown rind and a dense, pale yellow interior with small holes,",
            # "MOTHAIS": "Mothais, round, wrapped in chestnut leaves, with a thin rind and a creamy, soft, white interior,",
            # "VACHERIN": "Vacherin, round, encased in spruce bark with a supple, moist, rind and a rich, spoonable center,",
            # "MOZZARELLA": "Mozzarella, round balls or blocks, with a smooth, thin rind and a soft, milky white, elastic interior,",
            # "TÊTE DE MOINE": "Tête de Moine, cylindrical, with a smooth, brown rind and a dense, creamy, pale interior, traditionally shaved into rosettes,",
            # "FROMAGE FRAIS": "Fromage Frais, served in tubs or scoops, rindless with a smooth, soft, creamy white texture, similar to thick yogurt,"
        }

        cheese_prompts = {}

        # On génère les prompts pour chaque fromage : on a  ?? prompts générés par fromage
        
        for fromage_maj, fromage in fromages.items():
            cheese_prompts[fromage_maj] = []
            for _, context in contextes.items(): 
                for _,fond in fonds.items():
                    for _,cadrage in cadrages.items():
                        prompt = f"A {cadrage} realistic image with very precise details and natural colours of {fromage}, {context}, {fond}."
                        cheese_prompts[fromage_maj].append({"prompt": prompt, "num_images": 1})

        return cheese_prompts

    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
