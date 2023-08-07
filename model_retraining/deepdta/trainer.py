# data processing and training of the DeepDTA paper in pytorch code with your own data
# Author: @ksun63
# Date: 2023-04-14

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt


class Dataset(Dataset):
    """
    Here, the input dataset should be a pandas dataframe with the following columns:
    protein, ligands, affinity, where proteins are the protein seqeunces, ligands are the 
    isomeric SMILES representation of the ligand, and affinity is the binding affinity
    """
    def __init__(self, df, seqlen=2000, smilen=200):
        """
        df: pandas dataframe with the columns proteins, ligands, affinity
        seqlen: max length of the protein sequence
        smilen: max length of the ligand SMILES representation
        """
        self.proteins = df['proteins'].values
        self.ligands = df['ligands'].values
        self.affinity = df['affinity'].values
        self.smilelen = smilen
        self.seqlen = seqlen
        self.protein_vocab = set()
        self.ligand_vocab = set()
        for lig in self.ligands:
            for i in lig:
                self.ligand_vocab.update(i)
        for pr in self.proteins:
            for i in pr:
                self.protein_vocab.update(i)

        # having a dummy token to pad the sequences to the max length
        self.protein_vocab.update(['dummy'])
        self.ligand_vocab.update(['dummy'])
        self.protein_dict = {x: i for i, x in enumerate(self.protein_vocab)}
        self.ligand_dict = {x: i for i, x in enumerate(self.ligand_vocab)}
        

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.proteins)

    def __getitem__(self, idx):
        """
        Get the protein, ligand, and affinity of the idx-th sample

        param idx: index of the sample
        """
        pr = self.proteins[idx]
        lig = self.ligands[idx]
        target = self.affinity[idx]
        protein = [self.protein_dict[x] for x in pr] + [self.protein_dict['dummy']] * (self.seqlen - len(pr))
        ligand = [self.ligand_dict[x] for x in lig] + [self.ligand_dict['dummy']] * (self.smilelen - len(lig))

        return torch.tensor(protein), torch.tensor(ligand), torch.tensor(target, dtype=torch.float)

def collate_fn(batch):
    """
    Collate function for the DataLoader
    """
    proteins, ligands, targets = zip(*batch)
    proteins = torch.stack(proteins, dim=0)
    ligands = torch.stack(ligands, dim=0)
    targets = torch.stack(targets, dim=0)
    return proteins, ligands, targets


class Trainer:
    """
    Trainer class of the DeepDTA model
    """
    def __init__(self, model, channel, protein_kernel, ligand_kernel, df, train_idx, val_idx, test_idx,
                 log_file, smilen=200, seqlen=2000):
        """
        model: DeepDTA model defined in model.py
        df: pandas dataframe with the columns protein, ligands, affinity
        train_idx: indices of the training set
        val_idx: indices of the validation set
        smilen: max length of the ligand SMILES representation
        seqlen: max length of the protein sequence
        log_file: file to save the training logs
        """
        self.dataset = Dataset(df, smilen=smilen, seqlen=seqlen)
        self.protein_vocab = len(self.dataset.protein_vocab) + 1
        self.ligand_vocab = len(self.dataset.ligand_vocab) + 1
        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        self.test_dataset = Subset(self.dataset, test_idx)
        self.protein_kernel = protein_kernel
        self.ligand_kernel = ligand_kernel
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model(self.protein_vocab, self.ligand_vocab, channel, protein_kernel, ligand_kernel).to(self.device)
        self.log_file = log_file
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def train(self, lr, num_epochs, batch_size, save_path):
        """
        Train the model
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        writer = SummaryWriter()

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, drop_last = False, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, drop_last = False, collate_fn=collate_fn)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, drop_last = False, collate_fn=collate_fn)

        # save the encoding dictionaries into json files
        with open('protein_dict-prk{}-ldk{}.json'.format(self.protein_kernel, self.ligand_kernel), 'w') as f:
            json.dump(self.dataset.protein_dict, f)
        with open('ligand_dict-prk{}-ldk{}.json'.format(self.protein_kernel, self.ligand_kernel), 'w') as f:
            json.dump(self.dataset.ligand_dict, f)

        
        best_weights = self.model.state_dict()
        best_val_loss = np.inf
        best_epoch = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            with tqdm(total=len(train_loader)) as pbar:
                for protein, ligand, target in train_loader:
                    protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(protein, ligand)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    pbar.update(1)

            train_loss /= len(train_loader)
            self.logger.info('Epoch: {} - Training Loss: {:.6f}'.format(epoch+1, train_loss))
            writer.add_scalar('train_loss', train_loss, epoch)

            # switch to evaluation mode
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for protein, ligand, target in val_loader:
                    protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                    output = self.model(protein, ligand)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict()
                best_epoch = epoch
                self.logger.info('Best Model So Far in Epoch: {}'.format(epoch+1))
            self.logger.info('Epoch: {} - Validation Loss: {:.6f}'.format(epoch+1, val_loss))
            writer.add_scalar('val_loss', val_loss, epoch)
        
        self.model.load_state_dict(best_weights)
        test_result = []
        with torch.no_grad():
            for protein, ligand, target in test_loader:
                protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                output = self.model(protein, ligand)
                test_result.append(output.cpu().numpy())
        test_result = np.concatenate(test_result)
        np.savetxt('test-result-prk{}-ldk{}.txt'.format(self.protein_kernel, self.ligand_kernel), test_result)
        
        self.logger.info('Best Model Loaded from Epoch: {}'.format(best_epoch+1))
        torch.save(self.model.state_dict(), save_path)
        self.logger.handlers[0].close()
        self.logger.removeHandler(self.logger.handlers[0])
        writer.close()
    
