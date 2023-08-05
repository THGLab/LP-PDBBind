#Vina PDBBind Retrain

To retrain and test vina on UCBSPlit, run python on **vina_reweight.py** :
```python vina_reweight.py```

## Installation Requirements

Ensure you have the following Python packages installed:

```bash
conda install -c conda-forge numpy pandas ase scipy
python -m pip install git+https://github.com/theochem/iodata.git
```

### Vina Binary
In order to run Vina, the following package/script need to be installed.

Download and install the Vina Binary execution file from: https://vina.scripps.edu/downloads/

Install Meeko for ligand preprocessing of Vina:
```
$ git clone https://github.com/forlilab/Meeko
$ cd Meeko
$ pip install .
```

Install ADFR suite for protein preprocessing: https://ccsb.scripps.edu/adfr/downloads/

Change the **VINA_BINARY**, **protein_prep_path** and **meeko_ligprep_path** variable in generric_vina_ad4.py to the path where you installed the packages.