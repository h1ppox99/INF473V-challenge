#########################
## MODULES NÉCESSAIRES ##
#########################


from pathlib import Path
from tqdm import tqdm


#########################
## FONCTION PRINCIPALE ##
#########################


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
            "dehors, sur un étalage de marché en bois, entouré d'autres fromages, dans un style rustique": "outside, on a wooden market stall, with a dozen of other cheeses, in a rustic style",
            "dans une assiette en céramique, avec des fruits et des noix": "on a ceramic plate, with fruits and nuts, in a minimalist style",
            "sur du pain frais sur une assiette émaillée": "on an enamel plate with fresh bread",
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
            "contre un vieux mur de briques avec des traces d'usure": "against a brick wall with signs of wear",
        }


        cadrages = {
            "en gros plan": "close-up",
            "avec un cadrage large": "wide",
        } 


        ############################################################################################################################
        ## DESCRIPTIONS DE FROMAGES OBTENUES GRACE A guidedufromage.com (voir slides présentation + soutenance pour explications) ##
        ############################################################################################################################


        fromages = {
            "EMMENTAL": "Emmental, a cheese with a yellow rind and a flexible yet firm ivory to yellow paste, with well-distributed, regular, and clear holes,",
            "FETA": "Feta, block or crumbled, with a creamy white interior, often without a rind,",
            "OSSAU-IRATY": "Ossau-Iraty, a sheep's milk cheese with a natural, hard crust ranging from yellow-orange to gray, featuring an ivory, white to amber, smooth cream paste, with small openings,",
            "MIMOLETTE": "Mimolette, a cheese featuring a gray-brown, crater-marked crust with a striking bright orange paste,",
            "MAROILLES": "Maroilles, a soft cheese with a washed, red-orange and variably moist crust, and a white to cream paste, speckled with small openings, and a supple texture with a firmer heart,",
            "GRUYÈRE": "Gruyère, a cheese with a firm, ivory to pale yellow paste featuring very small holes, and a rubbed, solid, and granular crust brown to orange,",
            "MOTHAIS": "Mothais, a soft goat's milk cheese in a flat cylinder shape, featuring a minimally developed, vermiculated white to ivory rind where the leaf rests, with potential natural spots of gray-blue mold,",
            "VACHERIN": "Vacherin, a round cheese, encased in spruce bark, with a white-orange rind and a supple, moist, rind and a rich, spoonable center,",
            "MOZZARELLA": "Mozzarella, round balls or blocks, with a smooth, thin rind and a soft, milky white, elastic interior,",
            "TÊTE DE MOINE": "Tête de Moine, a Swiss cheese with a cylindrical form, featuring a firm, grainy, reddish-brown crust and a homogeneous creamy paste ranging from ivory to pale yellow, shaved into rosettes using a girolle,",
            "FROMAGE FRAIS": "Fromage Frais, served in tubs or scoops, rindless with a smooth, soft, creamy white texture, similar to thick yogurt,",
            "BRIE DE MELUN": "Brie de Melun, very flat form, with a textured shaded beige rind, with a soft creamy ivory interior,",
            "CAMEMBERT": "Camembert, round and flat, with a soft, creamy interior beneath a bloomy, velvety white rind, slightly dimpled with age,",
            "EPOISSES": "Epoisse, a cylindrical cheese with flat faces and straight sides, featuring a shiny, smooth, slightly wrinkled crust, ivory-orange to brick-red, and a soft, creamy beige-to-white center,",
            "FOURME D'AMBERT": "Fourme d'Ambert, a tall, cylindrical cheese with a fine, dry, light gray rind with few white molds with blue hues, and an ivory-white paste dotted with blue-gray spots, soft, creamy,",
            "RACLETTE": "Raclette, a semi-hard cheese with a thin, smooth golden-brown rind and a pale yellow, elastic interior, ideal for melting,",
            "MORBIER": "Morbier, a cheese with a beige to orange crust with a smooth, ivory to pale yellow paste, distinguished by a continuous black line of vegetable ash through its center,",
            "SAINT-NECTAIRE": "Saint-Nectaire, a cylindrical cheese, featuring a bloomy and washed crust with white-brown mold, revealing a cream to orange base, and with a supple, creamy ivory paste with small, evenly distributed openings,",
            "POULIGNY-SAINT-PIERRE": "Pouligny-Saint-Pierre, a soft goat's milk cheese, featuring an elongated pyramid trunk shape with a marbled, vermiculated white-ivory crust, and a smooth, homogeneous white-ivory paste,",
            "ROQUEFORT": "Roquefort, a cheese featuring a fine, regularly veined crust and moist, ivory paste threaded with emerald-green mold, creating a creamy, melting texture,",
            "COMTÉ": "Comté, a large cheese with a rugged golden to brown rind and a dense, smooth interior ranging from ivory to yellow, speckled with white tyrosine clusters,",
            "CHÈVRE": "Chèvre, various shapes including logs, discs, and pyramids, with a thin, natural rind enveloping a soft, bright white, creamy center,",
            "PECORINO": "Pecorino, round or drum-shaped, featuring a hard, pale rind and a dense, granular, off-white interior,",
            "NEUFCHATEL": "Neufchâtel, a heart-shaped soft cow's milk cheese with a bloomy, white down-covered rind, featuring a smooth, firm yet supple creamy and tender paste,",
            "CHEDDAR": "Cheddar, block form, with a firm, natural rind and a deep orange, slightly crumbly interior,",
            "BÛCHETTE DE CHÈVRE": "Bûchette de Chèvre, small log shape with a delicate, bloomy rind and a soft, creamy, bright white center,",
            "PARMESAN": "Parmesan, a large cheese known for its thick, hard, golden rind and rich, granular, pale yellow interior,",
            "SAINT-FÉLICIEN": "Saint-Félicien, a soft cow's milk cheese shaped like a flat cylinder, featuring a slightly pleated, flexible ivory crust with occasional light blue down and a white, lactic paste, often sold in a ceramic dish,",
            "MONT D'OR": "Mont d'Or, a cheese encircled with spruce and housed in a wooden box, featuring a yellow to light brown, fragile rind with a slight white down, and a glossy, soft, creamy yellow-white paste,",
            "STILTON": "Stilton, cylindrical, with a rough, natural rind and a richly veined, semi-soft, blue-speckled interior,",
            "SCAMORZA": "Scamorza, an Italian cheese with a spun, firm paste and minimal aging, featuring a distinctive ovoid shape, cinched by a cord, and a smooth crust, either in white or a caramel-colored exterior,",
            "CABÉCOU": "Cabécou, a small, round, soft goat's milk cheese with a fine, pale yellow crust, lightly dusted with white geotrichum down, and creamy white paste,",
            "BEAUFORT": "Beaufort, a grand, round cheese, with sturdy, golden-brown rind and a firm, dense rind, vibrant pale yellow interior with a uniform, creamy texture, showcasing subtle tiny holes,",
            "MUNSTER": "Munster, a soft cow's milk cheese, showcasing a fine crust ranging from ivory-orange to orange-red, and a supple, creamy paste from ivory to light beige with a distinct core,",
            "CHABICHOU": "Chabichou, a bung-shaped, soft goat's milk cheese featuring a thin, vermiculated ivory rind covered in light white down and occasional colorful molds,",
            "TOMME DE VACHE": "Tomme de Vache, a cow's milk cheese, featuring a thick, smooth to slightly rugged gray crust with light to dark brown secondary molds, and a both supple and firm paste, white to yellow, with small openings,",
            "REBLOCHON": "Reblochon, a cheese from Savoie featuring a fine saffron-yellow crust with a discreet white bloom, and a creamy, smooth paste, ranging from white to ivory with small openings,"
        }


        cheese_prompts = {}

        # On génère les prompts pour chaque fromage : on a 112 (À MODIFIER) prompts générés par fromage
        
        for fromage_maj, fromage in fromages.items():
            cheese_prompts[fromage_maj] = []
            for _, context in contextes.items(): 
                for _,fond in fonds.items():
                    for _,cadrage in cadrages.items():
                        prompt = f"A {cadrage} realistic image with very precise details of {fromage}, {context}, {fond}."
                        cheese_prompts[fromage_maj].append({"prompt": prompt, "num_images": 1})

        return cheese_prompts

    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
