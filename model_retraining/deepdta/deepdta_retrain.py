from model import DeepDTA
from trainer import Trainer
import pandas as pd

# this CSV file has 4 columns, protein, ligands, affinity, split.

fp = 'path_to_your_data.csv'

df = pd.read_csv(fp)
train_idx = df[df['split'] == 'train'].index.values
val_idx = df[df['split'] == 'val'].index.values
test_idx = df[df['split'] == 'test'].index.values

model = DeepDTA
channel = 32
protein_kernel = [8, 12]
ligand_kernel = [4, 8]

for prk in protein_kernel:
    for ldk in ligand_kernel:
        # epoch 50 is enough for convergence in this case, but may need more for other datasets
        trainer = Trainer(model, channel, prk, ldk, df, train_idx, val_idx, test_idx, "training-prk{}-ldk{}.log".format(prk, ldk))
        trainer.train(num_epochs=30, batch_size=256, lr=0.001, save_path='deepdta_retrain-prk{}-ldk{}.pt'.format(prk, ldk))
