# Embeddings 

This is a repository that includes scripts to create embeddings for a Sentinel-2 tile using a model trained on the [MMEarth](https://vishalned.github.io/mmearth) dataset. 

## Usage

- `helpers_embeddings.py` contains all scripts needed to generate embeddings. You can use the functions directly as shown in the `if __name__ == '__main__':` block at the bottom. 
- We use the Multi-modal MMEarth pre-trained weights that can be downloaded from [here](https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_1M_64_uncertainty_56-8/checkpoint-199.pth). This checkpoint corresponds to the **Multi-Modal** model, trained on **MMEarth64** and using the **12 band Sentinel-2** as input.


## Installation

To install the dependencies needed for the model, follow the README in the `MMEarth_train` directory. 

