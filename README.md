
1) voir le fichier contexts dans data/dataset_generators/
2) voir le folder ocr
3) voir create_submition (surtout la fonction auxiliaire)
4) voir éventuellement les fichiers : 
- merge_submissions
- merge_model


###########################################################################
###########################################################################

# Explication des scripts

## curate.py
Ce script permet de supprimer les images générées qui sont peu semblables à celles du val set. On utilise pour cela clip.

On peut configurer le dossier à curer dans configs/curate.





## generator_IPA.py
Ce script génère des images à partir d'images réelles (val) et de descriptions.

Voici les instructions d'installation de IPAdapter (que nous avons réalisée dans un dossier IPA/): 

    # install latest diffusers
    pip install diffusers==0.22.1

    # install ip-adapter
    pip install git+https://github.com/tencent-ailab/IP-Adapter.git

    # download the models
    cd IP-Adapter
    git lfs install
    git clone https://huggingface.co/h94/IP-Adapter
    mv IP-Adapter/models models
    mv IP-Adapter/sdxl_models sdxl_models

Nous avons ensuite modifié le fichier de config (dans configs/IPAdapter) pour faire correspondre les chemins des modèles, encoders...

Il a aussi fallu choisir les images du validation set (que l'on a copiées dans dataset/IPA2)

## generator_IPA_fusion.py
Ce script fusionne 2 images (ou plus si besoin) en prenant la moyenne de leur embedding clip et génère une nouvelle image à l'aide d'un prompt textuel.

Il faut ajouter la classe suivante au package ip_adapter : 

```python
class IPAdapterPlusv2(IPAdapter):
    """IP-Adapter with fine-grained features and fusion of images"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
    
    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        
        if pil_image is not None:
            all_clip_image_embeds = []
            for image in pil_image:
                clip_image = self.clip_image_processor(images=[image], return_tensors="pt").pixel_values
                clip_image = clip_image.to(self.device, dtype=torch.float16)
                clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
                all_clip_image_embeds.append(clip_image_embeds)
            
            # Compute the average of all embeddings
            averaged_clip_image_embeds = torch.mean(torch.stack(all_clip_image_embeds), dim=0)
            image_prompt_embeds = self.image_proj_model(averaged_clip_image_embeds)

            # For unconditional embeddings
            uncond_clip_image = torch.zeros_like(clip_image)
            uncond_clip_image_embeds = self.image_encoder(uncond_clip_image, output_hidden_states=True).hidden_states[-2]
            uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
            
            return image_prompt_embeds, uncond_image_prompt_embeds
        elif clip_image_embeds is not None:
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            uncond_clip_image_embeds = self.image_encoder(
                torch.zeros_like(clip_image_embeds), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
            return image_prompt_embeds, uncond_image_prompt_embeds
        else:
            raise ValueError("Either pil_image or clip_image_embeds must be provided")'
```

   


#################################
#################################


# Cheese Classification challenge
This codebase allows you to jumpstart the INF473V challenge.
The goal of this channel is to create a cheese classifier without any real training data.
You will need to create your own training data from tools such as Stable Diffusion, SD-XL, etc...

## Installation

Cloning the repo:
```
git clone git@github.com:nicolas-dufour/cheese_classification_challenge.git
cd cheese_classification_challenge
```
Install dependencies:
```
conda create -n cheese_challenge python=3.10
conda activate cheese_challenge
```
### Install requirements:
##### Install Pytorch:
If CUDA>=12.0:
```
pip install torch torchvision
```
If CUDA == 11.8
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
```
Then install the rest of the requirements
```
pip install -r requirements.txt
```

Download the data from kaggle and copy them in the dataset folder
The data should be organized as follow: ```dataset/val```, ```dataset/test```. then the generated train sets will go to ```dataset/train/your_new_train_set```

## Using this codebase
This codebase is centered around 2 components: generating your training data and training your model.
Both rely on a config management library called hydra. It allow you to have modular code where you can easily swap methods, hparams, etc

### Training

To train your model you can run 

```
python train.py
```

This will save a checkpoint in checkpoints with the name of the experiment you have. Careful, if you use the same exp name it will get overwritten

to change experiment name, you can do

```
python train.py experiment_name=new_experiment_name
```

### Generating datasets
You can generate datasets with the following command

```
python generate.py
```

If you want to create a new dataset generator method, write a method that inherits from `data.dataset_generators.base.DatasetGenerator` and create a new config file in `configs/generate/dataset_generator`.
You can then run

```
python generate.py dataset_generator=your_new_generator
```

### VRAM issues
If you have vram issues either use smaller diffusion models (SD 1.5) or try CPU offloading (much slower). For example for sdxl lightning you can do

```
python generate.py image_generator.use_cpu_offload=true
```

## Create submition
To create a submition file, you can run 
```
python create_submition.py experiment_name="name_of_the_exp_you_want_to_score" model=config_of_the_exp
```

Make sure to specify the name of the checkpoint you want to score and to have the right model config
