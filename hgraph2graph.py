import argparse
from hgraph import *
import rdkit
import yaml 
from types import SimpleNamespace
import torch

class VAE():
    def __init__(self, config_file = "/nfs/turbo/umms-welchjd/azhung/dalle2/LUMIC/config/hgraph2graph_config.yaml"):
        with open(config_file, 'r') as file:
            self.args = yaml.safe_load(file)
        self.args = SimpleNamespace(**self.args)
        self.args.atom_vocab = common_atom_vocab

        vocab = [x.strip("\r\n ").split() for x in open(self.args.vocab)]
        self.args.vocab = PairVocab(vocab)
       
        self.vae = HierVAE(self.args)
        self.vae.load_state_dict(torch.load(self.args.model, map_location=torch.device('cuda'))[0])
        self.vae.eval()
        self.vae.cuda()
        
    # Sample a latent vector for conditioning given smile embedding 
    def get_latent(self, smile):
        root_vecs, root_kl = self.vae.rsample(smile, self.vae.R_mean, self.vae.R_var, perturb=True)
        return root_vecs.detach()
    
    # Convert latent back to a smile 
    def decode(self, latent):
        decoded_smiles = self.vae.decoder.decode((latent, latent, latent), greedy=True, max_decode_step=200)
        return decoded_smiles