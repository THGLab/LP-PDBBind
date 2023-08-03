# UCBSplit: Developing More Generalizable Scoring Functions with Better Utilization of the PDBBind Dataset
This repository contains all the code for creating UCBSplit of PDBBind dataset, building the BDB2020+ dataset, prepared dataset files, and scripts for retraining AutoDock vina, IGN, RFScore and DeepDTA models

## Authors
* Jie Li `jerry-li1996@berkeley.edu`
* Xingyi Guan `nancy_guan@berkeley.edu`
* Oufan Zhang `oz57@berkeley.edu`
* Kunyang Sun `kysun@berkeley.edu`
* Yingze Wang `ericwangyz@berkeley.edu`
* Dorian Bagni `dorianbagni@berkeley.edu`
* Teresa Head-Gordon `thg@berkeley.edu`

## Compiled datasets
The UCBSplit of PDBBind 2020 is given in `dataset/UCBSplit.csv`. The `new_split` column corresponds to which category the data belongs to in UCBSplit. Additionally, `CL1`, `CL2`, `CL3` and `covalent` are boolean columns indicating whether the data is in the corresponding clean levels, and whether the data is covalent or not. Therefore, the following python code reads in the dataset and selects all data in the train set, and satisfy CL1 and non-covalent:

```python
import pandas as pd
df = pd.read_csv('dataset/UCBSplit.csv', index_col=0)
df_train = df[(df['new_split'] == 'train') & df.CL1 & ~df.covalent]
```