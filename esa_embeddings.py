'''
A script to get the np array of one sentinel-2 tile, patchify it, run the MMEarth model on it, and save the embeddings.
'''


import numpy as np
import os
import argparse
import json
import torch

import MMEarth_train.models.convnextv2 as convnextv2
from MMEarth_train.helpers import load_custom_checkpoint
import torch.nn as nn


from MMEarth_train.MODALITIES import *
import time

import matplotlib.pyplot as plt




def visualize_s2_tile(store_path):
    '''
    Visualize the sentinel-2 tile.
    '''

    path = './sample_s2_tile.npy'
    tile = np.load(path)

    # we only take rgb bands, hence thats b4,b3,b2. 
    tile = tile[[3, 2, 1], :, :]

    print('tile shape:', tile.shape)

    # for s2 processing we divide by 10000
    tile = tile / 10000 
    tile = tile.transpose(1, 2, 0)

    clip_val = 0.5
    tile = np.clip(tile, 0, clip_val)
    tile = tile / clip_val

    plt.imshow(tile)
    plt.savefig(f'{store_path}/s2_tile.png')


def visualize_all_embeddings(store_path):
    '''
    Visualize all the embedding dimensions.
    We visualize each dimension of the embeddings in a grid along with the original tile resized to the same size as the embeddings.
    '''
    with open(f'{store_path}/embeddings.npy', 'rb') as f:
        embeddings = np.load(f)

    print('embeddings shape:', embeddings.shape)
    embeddings = embeddings.squeeze()
    h, w = embeddings.shape[1], embeddings.shape[2]


    # load the s2 tile to visualize the embeddings and the tile together
    with open('./sample_s2_tile.npy', 'rb') as f:
        tile = np.load(f)
    tile = tile[[3, 2, 1], :, :] # we only take rgb bands, hence thats b4,b3,b2.
    tile = tile / 10000
    tile = tile.transpose(1, 2, 0)
    clip_val = 0.5
    tile = np.clip(tile, 0, clip_val)
    tile = tile / clip_val

    
    save_path = f'{store_path}/all_embeddings_vis'
    os.makedirs(save_path, exist_ok=True)



    for dim in range(embeddings.shape[0]):
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(tile)
        plt.axis('off')
        plt.title('S2 Tile')


        one_dim = embeddings[dim, :, :]

        plt.subplot(1, 2, 2)
        plt.imshow(one_dim, cmap='viridis')
        plt.axis('off')

        # remove the white space around the plot
        plt.tight_layout()
        # remove the white space between the title and the plot
        plt.title(f'Dimension {dim + 1}')
        plt.savefig(os.path.join(save_path, f'dim_{dim}.png'), bbox_inches='tight')
        plt.close()


        break # only for testing purposes. Remove this line to visualize all dimensions.


def preprocess_sentinel2_tile(tile, band_stats):
    '''
    Parameters:
    -----------

    tile: np array
        The sentinel-2 tile to preprocess. The tile is of shape 12 x 256 x 256.

    band_stats: dict
        The band statistics used to normalize the tile. The band stats are of the form:
        {
            'mean': list of floats,
            'std': list of floats
        }

    Returns:
    --------

    tile: np array
        The preprocessed sentinel-2 tile. The tile is normalized using the mean and std of the training dataset.
    '''

    # the band stats has a NaN value for the B10 band. We need to remove it.
    mean = band_stats['mean']
    std = band_stats['std']

    # remove the NaN value from the mean and std (this is only for the B10 band. For MMEarth, we computed the stats for both l1c and l2a. Hence to keep the stats consistent in shape, B10 was included.
    mean = [m for m in mean if not np.isnan(m)]
    std = [s for s in std if not np.isnan(s)]
    mean = np.array(mean)
    std = np.array(std)

    # normalize the tile
    tile = (tile - mean[:, None, None]) / std[:, None, None]

    return tile


def load_MMearth_model():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # the following arguments are required by the model and have to be set.
    args.model = 'convnextv2_atto'
    args.return_embeddings = True # we need the embeddings to save them to disk.
    args.finetune = '/lustre/scratch/WUR/AIN/nedun001/MMEarth-train-LSAI/ckpts/checkpoint-199.pth' # the directory where the model checkpoints are saved. We use the multi-modal ckpt.
    args.in_chans = 12 # the number of input bands (sentinel-2 has 12 bands and we use all of them)



    # the below arguments are set to the values used in MMEarth model training. They are irrelevant for this task of computing embeddings.
    args.patch_size = 8 # patch size used when pretraining. Either 16 or 8 (for img sizes 112 and 56 respectively)
    args.use_orig_stem = False # if True the model uses the original stem as in ConvNeXtV2, else it uses the modified MP-MAE stem.
    args.drop_path = 0.1 # the drop path rate used in the model. Either 0.1 or 0.2 . This is irrelevant for this task.
    args.head_init_scale = 0.001 # irelevant for this task since we are not training the model.
    args.input_size = 56 # this is just the image size used in MMEarth model training.
    args.linear_probe = False # irrelevant for this task but the argument is required by the model.
    args.model_prefix = ''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model = convnextv2.__dict__['convnextv2_atto'](
        num_classes=1, # irrelevant for this task 
        drop_path_rate=args.drop_path,
        head_init_scale=args.head_init_scale,
        args=args,
        patch_size=args.patch_size,
        img_size=args.input_size,
        use_orig_stem=args.use_orig_stem,
        in_chans=args.in_chans, # the number of input bands
    )

    model, _ = load_custom_checkpoint(model, args)
    model = model.to(device)

    model.eval()

    return model



def run_MMEarth_model_on_patches(model, patches):
    '''
    Parameters:
    -----------

    patches: list of np arrays
        A list of patches to run the MMEarth model on. 

    Returns:
    --------

    embeddings: list of np arrays 
        A list of embeddings for each patch. The embeddings are of shape 320 x 6 x 6.


    Notes:
    ------

    - For now the script generates the embedding map which downsamples the input image size by 8. We then compute the average pool to reduce it to 6x6. 
    - You can change the kernel size in the AvgPool2d to get a different size.

    '''
    # create the parser required for the model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = []
    start_time = time.time()
    for patch in patches:
        patch = torch.tensor(patch).to(device).float()
        patch = patch.unsqueeze(0) # add a batch dimension

        with torch.no_grad():
            embedding = model(patch)
            # print('embedding shape:', embedding.shape)
            # The embedding is of shape 320x133x133. We compute average pool over the spatial dim to get 320x6x6
            avg_pool = nn.AvgPool2d(kernel_size=22, stride=22)
            embedding = avg_pool(embedding)

            # print('embedding shape:', embedding.shape)
            # exit()
        embeddings.append(embedding.cpu().numpy())

    end_time = time.time()
    print(f'Time taken to run the model on one image: {end_time - start_time} seconds')

    return embeddings # a list of embeddings for each patch




def save_embeddings(embeddings, store_path):
    '''
    Parameters:
    -----------

    embeddings: list of np arrays
        A list of embeddings to save to disk. The embeddings are of shape 320 x 6 x 6.

    store_path: str
        The directory where the embeddings are saved to disk.

    '''
    print('final embeddings shape:', embeddings[0].shape)
    with open(f'{store_path}/embeddings.npy', 'wb') as f:
        np.save(f, embeddings)
    pass


def create_embeddings_from_MMEarth(model, tile, store_path):
    '''
    Parameters:
    -----------

    tile: np array
        The sentinel-2 tile to preprocess and run the MMEarth model on. The tile is of shape 12 x 256 x 256.

    store_path: str
        The directory where the embeddings are saved to disk.

    '''

    # Preprocess the sentinel-2 tile similar to the MMEarth model
    # the only thing we do is normalize using the mean and std of the training dataset. 
    # we use similar normalization statistics as in MMEarth model training.
    band_stats = json.load(open('./MMEarth_train/band_stats.json')) # pre computed band stats for the MMEarth training data.
    band_stats = band_stats['sentinel2_l2a'] 
    tile = preprocess_sentinel2_tile(tile, band_stats)

    patches = [tile] # created a list incase there are plans to patchify the tile. For now we just run the model on the full tile.
    print(f'Shape of the tile: {tile.shape}')

    # Run the MMEarth model on the patches
    embeddings = run_MMEarth_model_on_patches(model, patches)


    print('number of embeddings:', len(embeddings))
    print('embedding shape', embeddings[0].shape) # we run the model on the full tile, so we should only have one embedding

    # Save the embeddings to disk
    save_embeddings(embeddings, store_path)







if __name__ == '__main__':
    store_path = './'
    os.makedirs(store_path, exist_ok=True)
    path = './sample_s2_tile.npy'
    # bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] # 12 bands in sentinel-2 L2A
    tile = np.load(path) # load one tile

    # load the MMEarth model
    model = load_MMearth_model()
    
    # creating embeddings
    create_embeddings_from_MMEarth(model, tile, store_path)


    # visualizations
    visualize_s2_tile(store_path)
    # visualize_embeddings()
    visualize_all_embeddings(store_path)
