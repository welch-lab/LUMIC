import pandas as pd
from unet import *
from gaussian_diffusion import *
from unet_1d import *
from gaussian_diffusion_1d import *
from gaussian_diffusion_superes import *
from dataset import *
from torch.optim import Adam
from ema_pytorch import EMA
import vision_transformer as vits
from vision_transformer import DINOHead
import utils
from tqdm.auto import tqdm
from hgraph2graph import *
import os 

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
class Trainer(object):
    def __init__(self, diffusion_model, stage, *, image_size = 64, batch_size = 64, split_batches = True):
        super().__init__()
        self.model = diffusion_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = batch_size
        image_size = image_size
        adam_betas = (0.9, 0.99)
        self.stage = stage
        
        # held out smiles for test set 
        # JUMP pilot seen gene smiles 
        seen_gene_smile = ["C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO",
"COCCOc1ccc(Nc2ncc(F)c(Nc3cccc(NC(=O)C=C)c3)n2)cc1",
"O=C(CCNNC(=O)c1ccncc1)NCc1ccccc1",
"CN(c1ccc2c(C)n(C)nc2c1)c1ccnc(Nc2ccc(C)c(c2)S(N)(=O)=O)n1",
"OCCN1C[C@@H](O)[C@@H](O)C(O)C1CO",
"Cc1cn2cc(CC(=O)N3CCC4(CN(C4)[C@@H]4CCc5cc(ccc45)-c4cc(C)ncn4)CC3)nc2s1",
"NS(=O)(=O)c1cccc2c1c(cc1[nH]c(=O)c(=O)[nH]c21)[N+]([O-])=O",
"Cc1nc[nH]c1CCN",
"Oc1ccc(cc1)-c1nc(c([nH]1)-c1ccncc1)-c1ccc(F)cc1",
"CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)[C@H](CSc1cccs1)C(=O)NO",
"O=C1CCc2cc(OCCCCc3nnnn3C3CCCCC3)ccc2N1",
"CCCCCCCCCCCCCCCCOc1nc2ccc(C)cc2c(=O)o1",
"COC(=O)C[C@](O)(CCCC(C)(C)O)C(=O)O[C@H]1[C@H]2c3cc4OCOc4cc3CCN3CCC[C@]23C=C1OC",
"CO[C@H]1\C=C\O[C@@]2(C)Oc3c(C2=O)c2c(O)cc(NC(=O)\C(C)=C/C=C/[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)c(O)c2c(O)c3C"]
    
        # JUMP pilot unseen gene smiles 
        unseen_gene_smile = ["Nc1nc(F)nc2n(cnc12)[C@@H]1O[C@H](CO)[C@@H](O)[C@@H]1O",
"CCOc1ccc(cc1OCC)-c1nc(no1)-c1cccc2C(CCc12)NCCO",
"CCCC(CCC)C(=O)NCc1ccc2n(ncc2c1)-c1ccccc1OC",
"CCCCC[C@H](O)\C=C\[C@H]1[C@H](O)C[C@@H]2O\C(C[C@H]12)=C/CCCC(O)=O",
"Nc1cc(F)ccc1NC(=O)\C=C\c1cnn(C\C=C\c2ccccc2)c1",
"CN(C1CCN(CC1)c1nnc(-c2ccnn2C)c2ccccc12)C(=O)c1ccc(F)cc1C(F)(F)F",
"CC(=O)c1ccc(cc1)S(=O)(=O)NC(=O)NC1CCCCC1",
"CCOC(=O)c1c(\C(=C/N)C#N)c2ccc(Cl)c(Cl)c2n1C",
"CCCCC1C(=O)N(N(C1=O)c1ccccc1)c1ccccc1",
"[O-][N+](=O)c1ccc2nc(COCc3nc4ccc(cc4[nH]3)[N+]([O-])=O)[nH]c2c1",
"CCN1\C(Sc2ccc(OC)cc12)=C\C(C)=O",
"C\C=C1/NC(=O)[C@H]2CSSCC\C=C\[C@H](CC(=O)N[C@H](C(C)C)C(=O)N2)OC(=O)[C@@H](NC1=O)C(C)C",
"NC(CO)(CO)CO",
"CC(C)Oc1ccc(cc1C#N)-c1nc(no1)-c1cccc2[C@H](CCc12)NCCO",
"OC[C@H]1O[C@H](C[C@@H]1O)n1cnc2[C@H](O)CNC=Nc12",
"OC(=O)[C@@H]1CCC(=O)N1"]

        # Style transfer smiles
        style_transfer_sm = ["CC\\1=C(C2=C(/C1=C\C3=CC=C(C=C3)S(=O)C)C=CC(=C2)F)CC(=O)O",
"C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)O",
"CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",
"C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl",
"CCCC1=CC(=O)NC(=S)N1",
"C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@]2(C#C)O)CCC4=CC(=O)CC[C@H]34",
"CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N",
"C=CCSCC=C",
"CS(=O)(=O)OCCCCOS(=O)(=O)C",
"C1=CN=C(C=N1)C(=O)N"]
        
        df = pd.read_csv("~/jump_style_transfer_combined_mapping.csv", index_col = 0)
        
        # test set consists of: all cell lines with style transfer smiles, unseen gene smile, and seen gene smiles and  all hela cells,
        self.test_df = df[df["smile"].isin(style_transfer_sm) | (df["CellLine"] == "HeLa") | (df["smile"].isin(unseen_gene_smile))| (df["smile"].isin(seen_gene_smile))]
        # training set includes everything not in the test set, but include Hela control images
        temp_df = self.test_df[~((self.test_df["CellLine"] == "HeLa") & (self.test_df["smile"] == "CS(=O)C"))]
        self.train_df = df.drop(temp_df.index)
        print("Training df: ", self.test_df.shape)
        print("Testing df: ", self.train_df.shape)
        
        # Weighting the probability of sampling a cell line + chemical the same across all combinations
        label_df_train = self.train_df["CellLineSmile"].value_counts()
        weights_train = ((1.0 / 758) / label_df_train[self.train_df["CellLineSmile"]]).to_numpy()
        samples_weight_train = torch.from_numpy(weights_train)
        self.sampler_train = torch.utils.data.WeightedRandomSampler(samples_weight_train, len(samples_weight_train), replacement = True)
        self.train_data =None
        self.test_data = None
       
        if stage == "1d":
            self.optimizer = Adam(self.model.parameters(), lr = 5e-4, betas = adam_betas)
            smile_emb = pd.read_csv("/nfs/turbo/umms-welchjd/azhung/hgraph2graph/jump_style_transfer_encoded_smiles.csv", index_col = 0)
            self.train_data = DiffusionDataset1D(self.train_df, smile_emb)
            self.test_data = DiffusionDataset1D(self.test_df, smile_emb)
        elif stage == "lowres":
            self.optimizer = Adam(self.model.parameters(), lr = 5e-5, betas = adam_betas)
            self.train_data = DiffusionDataset(self.train_df)
            self.test_data = DiffusionDataset(self.test_df)
        elif stage == "superes":
            self.optimizer = Adam(self.model.parameters(), lr = 5e-5, betas = adam_betas)
            self.train_data = DiffusionDatasetSuperRes(self.train_df)
            self.test_data = DiffusionDatasetSuperRes(self.test_df)
            
        self.train_dataset = DataLoader(self.train_data, batch_size = batch_size,  num_workers = 8, drop_last = False, pin_memory = True, sampler = self.sampler_train, persistent_workers = True)
        self.test_dataset = DataLoader(self.test_data, batch_size = batch_size, num_workers = 8, drop_last = False, pin_memory = True, shuffle = True, persistent_workers = True)
        
        self.train_dl = cycle(self.train_dataset)
        self.val_dl = cycle(self.test_dataset)
        self.ema = EMA(diffusion_model, beta = 0.995, update_every = 10)
        self.ema.to(device)
        self.dino = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
        self.dino.cuda()
        utils.load_pretrained_weights(self.dino, "/nfs/turbo/umms-welchjd/azhung/dalle2/LUMIC/dino_checkpoint.pth", "teacher", "vit_small", 16)
        self.dino.eval()
        self.transform = torchvision.transforms.Resize(64)
        
    def save(self, iter, path = "/nfs/turbo/umms-welchjd/azhung/dalle2/checkpoints/"):
        data = {'model': self.model.state_dict(),
                'ema': self.ema.state_dict()
            }
        if self.stage == "1d":
            torch.save(data, path + "1d_diff_" + str(iter) + ".pt")
        elif self.stage == "lowres":
            torch.save(data, path + "lowres_imagen_"+ str(iter) + ".pt")
        elif self.stage == "superes":
            torch.save(data, path + "superes_imagen_"+ str(iter) + ".pt")
        
    def load(self, iter, path = "/nfs/turbo/umms-welchjd/azhung/dalle2/checkpoints/"):
        if self.stage == "1d":
            data = torch.load(path + "1d_diff_" + str(iter) + ".pt")
        elif self.stage == "lowres":
            data = torch.load(path + "lowres_imagen_" + str(iter) + ".pt")
        elif self.stage == "superes":
            data = torch.load(path + "superes_imagen_" + str(iter) + ".pt")
        self.model.load_state_dict(data['model'])
        self.ema.load_state_dict(data["ema"])
    def gen_1d_samples(self, embedding_fname):
        df = pd.read_csv(embedding_fname, index_col = 0)
        cell_lines = ["U2OS"]
        self.ema.ema_model.eval()
        vae = VAE()
        counter = -1 
        ds = DiffusionDataset1DSample(df)
        dl = DataLoader(ds, batch_size = 32, num_workers = 4, shuffle = True, drop_last = True, pin_memory = True, persistent_workers = True)
        emb_df = []
        actual_df = []
        if not os.path.exists("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/U2OS_unconditional_actual/"):
            os.makedirs("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/U2OS_unconditional_actual/")
        if not os.path.exists("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/U2OS_unconditional_actual_64/"):
            os.makedirs("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/U2OS_unconditional_actual_64/")
        counter = 0
        for step, batch in enumerate(dl):
            img, chem_emb, control_img, unpadded_img, small_img = batch[0].cuda(), batch[1].cuda().squeeze(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda()
            for a in range(img.shape[0]):
                torchvision.utils.save_image(unpadded_img[a, :, :, :], "/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/U2OS_unconditional_actual/" + str(step) + "_" + str(a) + ".png", nrow = 1)
                torchvision.utils.save_image(small_img[a, :, :, :], "/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/U2OS_unconditional_actual_64/" + str(step) + "_" + str(a) + ".png", nrow = 1)
            img_emb = self.dino(img)
            control_emb = self.dino(control_img)
            chem_emb = vae.get_latent(chem_emb)
            sample = self.ema.ema_model.sample(gene_emb = chem_emb, dmso_emb = control_emb, cond_scale = 5)
            emb_df.append(sample.detach().cpu().numpy().squeeze())
            actual_df.append(img_emb.detach().cpu().numpy().squeeze())
            if counter >= 32:
                break
            counter += 1
            
        emb_df = np.concatenate(emb_df, axis=0)
        actual_df = np.concatenate(actual_df, axis=0)
        emb_df = pd.DataFrame(emb_df)
        emb_df.to_csv("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/CSV/u2os_unconditional_gen_emb.csv")
        actual_df = pd.DataFrame(actual_df)
        actual_df.to_csv("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/CSV/u2os_unconditional_actual_emb.csv")
            
       
    def gen_lowres_samples(self, embedding_fname):
        self.ema.ema_model.eval()
        gen_emb = pd.read_csv(embedding_fname, index_col = 0)
        gen_emb = torch.tensor(gen_emb.values).cuda()
        if not os.path.exists("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/u2os_unconditional_64_gen_emb/"):
            os.makedirs("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/u2os_unconditional_64_gen_emb/")
        for i in range(16):
            sample_gen = self.ema.ema_model.sample(cond_emb = gen_emb[64 * i: 64 * i + 64, :], cond_scale = 5)
            for k in range(sample_gen.shape[0]):
                torchvision.utils.save_image(sample_gen[k, :,:, :].squeeze(), "/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/u2os_unconditional_64_gen_emb/" + str(i) + "_" + str(k) + ".png", nrow = 1)
            
    def gen_super_res_sample(self):
        tran = torchvision.transforms.Compose([torchvision.transforms.Resize(256),torchvision.transforms.ToTensor()])
        self.ema.ema_model.eval()
        with torch.no_grad():
            for cell_line in ["HEK293T"]:
                for i in range(0, 10):
                    gen_emb = pd.read_csv("/home/azhung/dalle2/" + cell_line + "_gen_emb_JUMP_style_transfer_no_channel_drop_smile_" + str(i) + ".csv", index_col = 0)
                    gen_emb = torch.tensor(gen_emb.values).cuda()
                    for l in range(16):
                        images = []
                        for a in range(64):
                            image = Image.open("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/" + cell_line + "_gen_emb_updated_no_channel_drop_smile_" + str(i) + "/" + str(l) + "_" + str(a) + ".png")
                            img = tran(image).cuda()
                            images.append(img)
                        images = torch.stack(images, dim = 0)
                        print(images.shape)
                        sample_gen = self.ema.ema_model.sample(img = images, cond_emb = gen_emb[64 * l: 64 * l + 64, :], cond_scale = 5)
                        if not os.path.exists("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/" + cell_line + "_gen_emb_updated_no_channel_drop_smile_256_" + str(i) + "/"):
                            os.makedirs("/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/" + cell_line + "_gen_emb_updated_no_channel_drop_smile_256_" + str(i) + "/")
                        for k in range(sample_gen.shape[0]):
                            torchvision.utils.save_image(sample_gen[k, :,:, :].squeeze(), "/nfs/turbo/umms-welchjd/azhung/Pilot_Results/Images/" + cell_line + "_gen_emb_updated_no_channel_drop_smile_256_" + str(i) + "/" + str(l) + "_" + str(k) + ".png", nrow = 1)
    def calculate_kid(self, folder_1, folder_2):
        kid = KernelInceptionDistance(subset_size = 500, normalize = True, reset_real_features = True)
        metrics = torch_fidelity.calculate_metrics(input1 = folder_1, input2 = folder_2, cuda = True, kid = True)
        print(metrics)
        
    def train_lowres(self, iter = 0, iters = 150000, path = "/nfs/turbo/umms-welchjd/azhung/dalle2/checkpoints/"):
        print("Starting lowres training: ")
        with tqdm(position = iter, total = iters) as pbar:
            while iter < iters:
                total_loss = 0
                img, dino_img = next(self.train_dl) 
                
                img = img.cuda()
                dino_img = dino_img.cuda()
 
                perturb_emb = self.dino(dino_img)

                loss = self.model(img, perturb_emb) 
                total_loss += loss.item()
                
                pbar.set_description(f'loss: {loss.item():.4f}')
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                iter += 1
                self.ema.update()
                if iter % 1000 == 0:
                    self.save(iter)
                    if iter != 1000:
                        os.remove(path + "lowres_imagen_" + str(iter - 1000) + ".pt")
                    f = open("lowres_imagen.txt", "a")
                    f.write("iter: " + str(iter) + " train loss: " + str(loss.item()) + "\n")
                    f.close()
                if iter % 5000 == 0:
                    self.ema.ema_model.eval()
                    with torch.no_grad():
                        avg_val_loss = 0
                        val_count = 0
                        
                        for val_step, val_batch in enumerate(self.test_dataset):
                            val_images, val_dino_images = val_batch[0].cuda(), val_batch[1].cuda()
                            val_perturb_emb = self.dino(val_dino_images)
                            val_perturb_emb = val_perturb_emb.cuda().squeeze()
                            val_loss = self.model(val_images, val_perturb_emb)
                            avg_val_loss += val_loss.item()
                            val_count += 1
                            if val_count >= 100:
                                break
                        f = open("lowres_imagen.txt", "a")
                        f.write("iter: " + str(iter) + " val loss: " + str(avg_val_loss/val_count) + " train loss: " + str(loss.item()) + "\n")
                        f.close()
                        test_images, test_dino_images = next(self.val_dl)
                        test_images = test_images.cuda()
                        test_dino_images = test_dino_images.cuda()
                        test_perturb_emb = self.dino(test_dino_images)
                        sample = self.ema.ema_model.sample(cond_emb = test_perturb_emb, cond_scale = 5)
                        all_images = torch.cat([test_images, sample], dim = 0)
                        torchvision.utils.save_image(all_images, "/home/azhung/dalle2/lowres_imagen_" + str(iter) + ".png", nrow = 16)
                        self.ema.ema_model.train()
                pbar.update(1)
    def train_1d(self, iter = 0, iters = 75000, path = "/nfs/turbo/umms-welchjd/azhung/dalle2/checkpoints/"):
        mse = nn.MSELoss()
        vae = VAE()
        print("Starting 1D training: ")
        with tqdm(position = iter, total = iters) as pbar:
            while iter < iters:
                total_loss = 0
                img, chem_emb, control_img = next(self.train_dl)
                img = img.cuda()
                control_img = control_img.cuda()
                img_emb = self.dino(img)
                control_emb = self.dino(control_img)
                chem_emb = vae.get_latent(chem_emb.cuda().squeeze())
                img_emb = img_emb[:, None, :]
                loss = self.model(img_emb, chem_emb, control_emb)
                total_loss += loss.item()
                
                pbar.set_description(f'loss: {loss.item():.4f}')
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                iter += 1
                self.ema.update()
                if iter % 1000 == 0:
                    self.save(iter)
                    if iter != 1000:
                        os.remove(path + "1d_diff_" + str(iter - 1000) + ".pt")
                    f = open("1d_diff.txt", "a")
                    f.write("iter: " + str(iter) + " train loss: " + str(loss.item()) + "\n")
                    f.close()
                if iter % 5000 == 0:
                    self.ema.ema_model.eval()
                    with torch.no_grad():
                        avg_val_loss = 0
                        val_count = 0
                        for val_step, val_batch in enumerate(self.test_dataset):
                            val_img, val_chem_emb, val_control_img = val_batch[0].cuda(), val_batch[1].cuda().squeeze(), val_batch[2].cuda()
                            val_img_emb = self.dino(val_img)
                            val_control_emb = self.dino(val_control_img)
                            val_chem_emb = vae.get_latent(val_chem_emb)
                            val_img_emb = val_img_emb[:, None, :]
                            val_loss = self.model(val_img_emb, val_chem_emb, val_control_emb)
                            avg_val_loss += val_loss.item()
                            val_count += 1
                            if val_count >= 100:
                                break
                        test_img, test_chem_emb, test_control_img = next(self.val_dl)
                        test_chem_emb = vae.get_latent(test_chem_emb.cuda().squeeze())
                        test_control_emb = self.dino(test_control_img.cuda())
                        test_img_emb = self.dino(test_img.cuda())
                        sample = self.ema.ema_model.sample(gene_emb = test_chem_emb, dmso_emb = test_control_emb, cond_scale = 5)
                        f = open("1d_diff.txt", "a")
                        f.write("test: " + str(mse(test_img_emb, sample.squeeze())))
                        f.close()
                        self.ema.ema_model.train()
                pbar.update(1)
    def train_superes(self, iter = 0, iters = 1000000, path = "/nfs/turbo/umms-welchjd/azhung/dalle2/checkpoints/"):
        print("Starting super resolution training: ")
        with tqdm(position = iter, total = iters) as pbar:
            while iter < iters:
                total_loss = 0
                img, dino_img = next(self.train_dl)
                
                img = img.cuda()
                perturb_emb = self.dino(torchvision.transforms.Pad(50)(img))
    
                loss = self.model(img, perturb_emb).sum()
 
                total_loss += loss.item()
                
                pbar.set_description(f'loss: {loss.item():.4f}')
                
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                
                iter += 1
                self.ema.update()
           
                if iter % 1000 == 0:
                    self.save(iter)
                    if iter != 1000:
                        os.remove("superes_imagen_" + str(iter - 1000) + ".pt")
                    f = open("superes_imagen.txt", "a")
                    f.write("iter: " + str(iter) + " train loss: " + str(loss.item()) + "\n")
                    f.close()
                if iter % 10000 == 0:
                    self.ema.ema_model.eval()
                    with torch.no_grad():
                        avg_val_loss = 0
                        val_count = 0
                        
                        for val_step, val_batch in enumerate(self.test_dataset):
                            val_images, val_dino_images = val_batch[0].cuda(), val_batch[1].cuda()
                            val_perturb_emb = self.dino(torchvision.transforms.Pad(50)(val_images))
                            val_perturb_emb = val_perturb_emb.cuda().squeeze()
                            val_loss = self.model(val_images, val_perturb_emb)
                            avg_val_loss += val_loss.item()
                            val_count += 1
                            if val_count >= 100:
                                break
                        f = open("superes_imagen.txt", "a")
                        f.write("iter: " + str(iter) + " val loss: " + str(avg_val_loss/val_count) + " train loss: " + str(loss.item()) + "\n")
                        f.close()
                        test_images, test_dino_images = next(self.val_dl)
                        test_images = test_images.cuda()
                        test_dino_images = test_dino_images.cuda()
                        test_perturb_emb = self.dino(torchvision.transforms.Pad(50)(test_dino_images))

                        sample = self.ema.ema_model.sample(img = test_images, cond_emb = test_perturb_emb, cond_scale = 5)
                        all_images = torch.cat([test_images, sample], dim = 0)
                        torchvision.utils.save_image(all_images, "/home/azhung/dalle2/checkpoints/superes_imagen_" + str(iter) + ".png", nrow = 4)
                        torch.cuda.empty_cache()
                        self.ema.ema_model.train()

                pbar.update(1)
if __name__ == '__main__':
    # model = Unet(dim = 64, channels = 3, dim_mults = (1, 2, 4, 8), cond_drop_prob = 0.2)
    # diffusion = GaussianDiffusion(model, batch_size = 64, image_size = 64, timesteps = 1000)
    # trainer = Trainer(diffusion, batch_size = 64, image_size = 64, stage = "lowres")
    # trainer.train_lowres()
    
    # model = Unet1D(dim = 64, channels = 1, dim_mults = (1, 2, 4, 8), cond_drop_prob = 0.2)
    # diffusion = GaussianDiffusion1D(model, batch_size = 64, seq_length = 384, timesteps = 1000)
    # trainer = Trainer(diffusion, batch_size = 64, image_size = 64, stage = "1d")
    # trainer.train_1d()
    
    # model = UnetSuperRes(dim = 128, channels = 3, dim_mults = (1, 2, 4, 8), cond_drop_prob = 0.2)
    # model = nn.DataParallel(model)
    # diffusion = SuperRes(unet = model)
    # trainer = Trainer(diffusion, batch_size = 11, stage = "superes")
    # trainer.train_superes()
    
    # Load a previous checkpoint
    #trainer.load(150000)
    
    # Generate samples 
    # trainer.gen_lowres_samples("~/dalle2/updated_style_transfer_mapping.csv")
    
    # Calculate KID between 2 folders
    # trainer.calculate_kid("/nfs/turbo/umms-welchjd/azhung/IMPA/multimodal_samples_gen/", "/nfs/turbo/umms-welchjd/azhung/IMPA/multimodal_samples_actual/")
