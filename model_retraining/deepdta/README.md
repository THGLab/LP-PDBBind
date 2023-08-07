# DeepDTA-Pytorch
Pytorch Implementation of the original DeepDTA paper (https://github.com/hkmztrk/DeepDTA/)

Requirements (most of them come with Anaconda except torch, pytorch-cuda, and tqdm)
```
python==3.8.16  
numpy==1.24.1  
pandas==1.5.2  
matplotlib==3.5.3  
scipy==1.8.1  
torch==2.1.0  
pytorch-cuda==11.7  
tqdm==4.65.0  
```

The data format should be in the form of a csv file with four columns: proteins, ligands, affinity, split, where proteins store all the sequence information, ligands store the isomeric smile strings of the molecular binders, and affinity was either the Kd/Ki value or the bidning affinity in kcal/mol (this needs to be consistent for all data). The final split column will have three possible values that indicate the train-val-test splitting: 'train', 'val', and 'test'.

To run the code, go to deepdta_retrain.py to do the appropriate modification of fp and then run 
```python deepdta_retrain.py```

For analysis, there's a separate jupyter notebook files for some preliminary scatter plots and using the trained model to analyze a held-out set of data. Make sure to change the name of ligand_dict and protein_dict and the model you want to use to your choices. The best model here has a protein kernel = 8 and ligand kernel = 8.


## Citation
```bibtex
@article{10.1093/bioinformatics/bty593,  
    author = {Öztürk, Hakime and Özgür, Arzucan and Ozkirimli, Elif},  
    title = "{DeepDTA: deep drug–target binding affinity prediction}",  
    journal = {Bioinformatics},  
    volume = {34},  
    number = {17},  
    pages = {i821-i829},  
    year = {2018},  
    month = {09},  
    issn = {1367-4803},  
    doi = {10.1093/bioinformatics/bty593},  
    url = {https://doi.org/10.1093/bioinformatics/bty593},  
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/34/17/i821/25702584/bty593.pdf},  
}
```
