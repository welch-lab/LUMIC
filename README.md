<!-- ABOUT THE PROJECT -->
## LUMIC

Latent diffUsion for Multiplexed Images of Cells is a diffusion model pipeline developed to generate the high-content fluorescent microscopy images of different cell type and chemical compound interactions. LUMIC combines diffusion models with DINO (self-Distillation with NO labels), a vision-transformer based, self-supervised method that can be trained on images to learn feature embeddings, and HGraph2Graph, a hierarchical graph encoder-decoder to represent chemicals.

<p align="center">
  <img src="https://github.com/welch-lab/LUMIC/blob/main/figs/model_arch.png" width="700" height="400">
</p>

Code based on lucidrains' [imagen](https://github.com/lucidrains/imagen-pytorch) and [ddpm](https://github.com/lucidrains/denoising-diffusion-pytorch)

### Installation

Clone the repo and create the environment using 
```
conda env create -f environment.yml
```

### File Descriptions
LUMIC Files
* `trainer.py`: contains the training functions to train LUMIC
* `dataset.py`: contains the dataloader to output the necessary images/embeddings
* `unet.py`: contains the Unet classes and building blocks for low-res and high-res diffusion models
* `unet_1d.py`: contains the Unet classes and building blocks for 1d diffusion model
* `gaussian_diffusion_superes.py`: contains necessary DDPM function for high-res diffusion models
* `gaussian_diffusion_1d.py`: contains necessary DDPM functions for 1d diffusion models
* `gaussian_diffusion.py`: contains necessary DDPM functions for low-res diffusion models

DINO Files
* `vision_transormer.py`: contains necessary functions for DINO
* `utils.py`: contains helper functions for DINO

Hgraph2Graph Files
* `config/hgraph2graph_config.yaml`: config used for Hgraph2Graph training and inference
* `hgraph/`: contains necessary functions for hgraph
* `hgraph2graph/zinc_lincs_sciplex_smiles.txt`: SMILES (306 JUMP, 61 style transfer, and ~250k ZINC) used for training HGraph2Graph (wrong file)
* `hgraph2graph/all_vocab_zinc.txt`: processing needed for HGraph2Graph (breaking down SMILES into Vocabulary)
* `hgraph2graph.py`: contains the functions necessary to use/sample from HGraph2Graph


<!-- GETTING STARTED -->
### Training



### Inference


### Model Checkpoints
Model checkpoints are available on huggingface [here](https://huggingface.co/azhung/StyleTransferData/tree/main).

### Dataset 
Datasets (both the JUMP and Style Transfer) are pre-processed and are available on hugging face [here](https://huggingface.co/azhung/StyleTransferData).

