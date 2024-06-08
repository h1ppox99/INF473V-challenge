Explication de cette branche


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

   


