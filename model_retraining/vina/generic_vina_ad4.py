#!/usr/bin/env python

# @Time: 4/27/23 
# @Author: Nancy Xingyi Guan
# @File: generic_vina_ad4.py.py

import numpy as np
import pandas as pd
import os
import sys
import shutil
import subprocess
import re
from pathlib import Path
import contextlib
from typing import List, Union, Optional, Tuple
import multiprocessing as mp
from functools import partial
import scipy
from scipy.optimize import minimize


# from vina import Vina
from ase import Atoms
from ase.io.sdf import read_sdf
from iodata import load_one
from iodata.utils import angstrom

Vina_weights_original_tuple = (-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 0.05846)
AD4_original_weights_tuple = (0.1662, 0.1209, 0.1406, 0.1322, 0.2983)
VINA_BINARY = "/global/scratch/users/nancy_guan/cgem/cgem_autodock/autodock/vina"
protein_prep_path = "/global/scratch/users/nancy_guan/cgem/cgem_autodock/ADFRsuite/bin/prepare_receptor"
meeko_ligprep_path = "/global/scratch/users/nancy_guan/cgem/cgem_autodock/autodock/Meeko/scripts/mk_prepare_ligand.py"
cwd = os.getcwd()

############
### Vina ###
############

def vina_preprocess(folder,lig_file,prot_file, lig_out_name = None, prot_out_name = None, add_h = True):
    processed_prot_file = prot_file[:-4]+'-processed.pdb'
    if lig_out_name == None:
        lig_out_name = lig_file.split('.')[0]+'.pdbqt'
    if prot_out_name == None:
        prot_out_name = prot_file.split('.')[0]+'.pdbqt'

    with set_directory(folder):
        if not os.path.isfile(processed_prot_file):
            with open(prot_file, "r") as f:
                protein_file = f.read().split("\n")
            new_file = [i for i in protein_file if not i.startswith('HETATM')]
            with open(processed_prot_file, "w") as f1:
                f1.write("\n".join(new_file))
        try:
            subprocess.run([meeko_ligprep_path,
                            "-i", lig_file, "-o", lig_out_name])
            if add_h:
                subprocess.run([protein_prep_path, '-r', processed_prot_file, '-o', prot_out_name,
                                '-A', 'checkhydrogens'])
            else:
                subprocess.run(
                    [protein_prep_path, "-r", processed_prot_file, "-o", prot_out_name])

        except subprocess.CalledProcessError as e:
            print(e.output)
            os.chdir(cwd)
            raise


def vina_scoring_binary(folder, lig_file, prot_file, weights=None,save_out_file = False):
    """
    lig_file, prot_file: .mol2, .sdf, .pdb file basename before preprocess
    """
    lig_name = lig_file.split('.')[0]
    prot_name = prot_file.split('.')[0]
    # name = folder.split('/')[-1]

    if not os.path.isfile(f"{folder}/{prot_name}.pdbqt") or not os.path.isfile(f"{folder}/{lig_name}.pdbqt"):
        vina_preprocess(folder,f"{lig_name}.sdf",f"{prot_name}.pdb")

    with set_directory(folder):
        ligand_COM = get_COM(lig_file)
        write_config_vina(f'{lig_name}.pdbqt', f'{prot_name}.pdbqt',ligand_COM,config_fp = "config.txt",weights=weights)
        cmd = f"{VINA_BINARY} --config config.txt --score_only"
        try:
            code, out, err = run_command(cmd, timeout=100)
        except CommandExecuteError as e:
            print(f"error in {folder}: {e}")
            print("out: ",out)
            print("err: ", err)
            os.chdir(cwd)
            raise

        # obtain docking score from the results
        strings = re.split('Estimated Free Energy of Binding   :', out)
        line = strings[1].split('\n')[0]
        energy = float(line.strip().split()[0])
        if save_out_file:
            with open("vina.out", 'w') as f:
                f.write(out)
    return energy

def write_config_vina(lig_pdbqt,prot_pdbqt,center, config_fp = "config.txt", weights=None, boxsize=25, exhaustiveness=32, num_modes=1, energy_range=30, **kwargs):
    '''
    Write the config file for AutoDock Vina docking
    :param exhaustiveness: int, the exhaustiveness of the docking
    :param num_modes: int, the number of modes (conformations) to be generated
    :param energy_range: int, the energy range of the docking
    '''

    lines = ["receptor = {}".format(prot_pdbqt),
             "ligand = {}".format(lig_pdbqt),
             "scoring = vina",
             "",
             "center_x = {}".format(center[0]),
             "center_y = {}".format(center[1]),
             "center_z = {}".format(center[2]),
             "",
             "size_x = {}".format(boxsize),
             "size_y = {}".format(boxsize),
             "size_z = {}".format(boxsize),
             "",
             # "exhaustiveness = {}".format(exhaustiveness),
             # "num_modes = {}".format(num_modes),
             # "energy_range = {}".format(energy_range),
             ]
    if weights is not None:
        assert len(weights) == 6, "Autodock vina needs 6 weights"
        # --weight_gauss1 1 --weight_gauss2 0 --weight_repulsion 0  --weight_hydrophobic 0 --weight_hydrogen 0 --weight_rot 0"
        lines.extend([
            f"weight_gauss1 = {weights[0]}",
            f"weight_gauss2 = {weights[1]}",
            f"weight_repulsion = {weights[2]}",
            f"weight_hydrophobic = {weights[3]}",
            f"weight_hydrogen = {weights[4]}",
            f"weight_rot = {weights[5]}",
        ])
    with open(config_fp, "w") as f:
        f.write("\n".join(lines))

############
### AD4  ###
############

def ad4_preprocess(folder,lig_file,prot_file, lig_out_name = None, prot_out_name = None, add_h=True, rerun = False):
    '''
    Convert a pdb file to a pdbqt file that can be used for AutoDock docking
    :param folder: str, path to the pdb file
    :param add_h: bool, whether or not add H to receptor
    :return: True if the run is successful
    '''
    # name = folder.split('/')[-1]
    processed_prot_file = prot_file[-4] + '-processed.pdb'
    if lig_out_name == None:
        lig_out_name = lig_file.split('.')[0] + '.pdbqt'
    if prot_out_name == None:
        prot_out_name = prot_file.split('.')[0] + '.pdbqt'
    do_preproc = True
    if not rerun:
        if os.path.isfile(f"{folder}/{lig_out_name}") and os.path.isfile(f"{folder}/{prot_out_name}"):
            do_preproc = False

    with set_directory(folder):
        if do_preproc:
            if not os.path.isfile(processed_prot_file):
                with open(prot_file, "r") as f:
                    protein_file = f.read().split("\n")
                new_file = [i for i in protein_file if not i.startswith('HETATM')]
                with open(processed_prot_file, "w") as f1:
                    f1.write("\n".join(new_file))
            try:
                subprocess.run([meeko_ligprep_path, '-i', lig_file, '-o', lig_out_name])
                if add_h:
                    subprocess.run([protein_prep_path, '-r', processed_prot_file, '-o', prot_out_name,
                                    '-A', 'checkhydrogens'])
                else:
                    subprocess.run(
                        [protein_prep_path, "-r", processed_prot_file, "-o", prot_out_name])
            except subprocess.CalledProcessError as e:
                print(e.output)
                os.chdir(cwd)
                raise
        path = "ad4files"
        if not os.path.isdir(path):
            os.mkdir(path)
        shutil.copy(lig_out_name, path)
        shutil.copy(prot_out_name, path)

    return True

def ad4_binary(folder, lig_pdbqt, prot_pdbqt, restart=False, save_out_file=False, score_only=True, weights=None):
    # https://autodock-vina.readthedocs.io/en/latest/docking_basic.html
    # assume successful prep of pdbqts
    prot_name = prot_pdbqt[:-6]
    path = f"{folder}/ad4files"
    if os.path.isdir(path) and restart:
        shutil.rmtree(path)
    if not os.path.isdir(path):
        os.mkdir(path)
    shutil.copy(f"{folder}/{lig_pdbqt}", path)
    shutil.copy(f"{folder}/{prot_pdbqt}", path)

    with set_directory(path, mkdir=False):
        # write gpf receptorH.gpf
        if weights is not None:
            write_parameter_file("parameters.dat", weights)
        write_config_ad4(lig_pdbqt,prot_name, config_fp="config.txt", score_only=score_only, weights=weights)
        try:
            if restart or (not os.path.isfile(f"{prot_name}.e.map")):
                subprocess.run(["python",
                                "/global/scratch/users/nancy_guan/cgem/cgem_autodock/DUD-E_all/write-gpf.py",
                                prot_pdbqt, "-l", lig_pdbqt])
                if weights is not None:
                    with open(f"{prot_name}.gpf", 'r') as f:
                        lines = f.readlines()
                    with open(f"{prot_name}.gpf", 'w') as f:
                        f.write("parameter_file parameters.dat\n")
                        f.writelines(lines)

                # run autogrid to Generating affinity maps for AutoDock FF
                # This command will generate the following files:
                # receptorH.maps.fld       # grid data file
                # receptorH.*.map          # affinity maps for A, C, HD, H, NA, N, OA atom types
                # receptorH.d.map          # desolvation map
                # receptorH.e.map          # electrostatic map

                subprocess.run(["/global/scratch/users/nancy_guan/cgem/cgem_autodock/autodock/autogrid4",
                                "-p", f"{prot_name}.gpf", "-l", f"{prot_name}.glg",
                                ])
        except subprocess.CalledProcessError as e:
            print(e.output)
            os.chdir(cwd)
            raise
        # run vina / ad4
        if score_only:
            cmd = f"{VINA_BINARY} --config config.txt --score_only"
        else:
            cmd = f"{VINA_BINARY} --config config.txt"

        try:
            code, out, err = run_command(cmd, timeout=100)
        except CommandExecuteError as e:
            print(f"error in {folder}: {e}")
            print("out: ", out)
            print("err: ", err)
            os.chdir(cwd)
            raise

        if save_out_file:
            with open("ad4.out", 'w') as f:
                f.write(out)

    # obtain docking score from the results
    strings = re.split('Estimated Free Energy of Binding   :', out)
    line = strings[1].split('\n')[0]
    energy = float(line.strip().split()[0])
    return energy

def write_config_ad4(lig_pdbqt,prot_name, config_fp="config.txt", weights=None, score_only=True, exhaustiveness=32, num_modes=1,
                     energy_range=30, **kwargs):
    '''
    Write the config file for AutoDock Vina docking
    :param exhaustiveness: int, the exhaustiveness of the docking
    :param num_modes: int, the number of modes (conformations) to be generated
    :param energy_range: int, the energy range of the docking
    '''

    lines = [
        "ligand = {}".format(lig_pdbqt),
        "maps = {}".format(prot_name),
        "scoring = ad4",

    ]
    if not score_only:
        lines.extend([
            "exhaustiveness = {}".format(exhaustiveness),
            "num_modes = {}".format(num_modes),
            "energy_range = {}".format(energy_range),
        ])

    if weights is not None:
        assert len(weights) == 5, "Autodock 4 needs 5 weights"
        # ad4 cgem: --weight_ad4_vdw 0.1496 --weight_ad4_hb 0.2046 --weight_ad4_elec 0.161 --weight_ad4_dsolv 0.1227 --weight_ad4_rot 0.3176
        lines.extend([
            f"weight_ad4_vdw = {weights[0]}",
            f"weight_ad4_hb = {weights[1]}",
            f"weight_ad4_elec = {weights[2]}",
            f"weight_ad4_dsolv = {weights[3]}",
            f"weight_ad4_rot = {weights[4]}",
        ])
    with open(config_fp, "w") as f:
        f.write("\n".join(lines))

def write_parameter_file(fpath, weights):
    assert len(weights) == 5, "Autodock 4 needs 5 weights"
    fstr = f"""# AutoDock 4 free energy coefficients with respect to original (AD2) energetic parameters
#  This model assumes that the bound and unbound conformations are the same.
#  See Table 3 in Huey,Morris,Olson&Goodsell (2007) J Comput Chem 28: 1145-1152.
#
#               Free Energy Coefficient
#               ------
FE_coeff_vdW    {weights[0]}
FE_coeff_hbond  {weights[1]}
FE_coeff_estat  {weights[2]}
FE_coeff_desolv {weights[3]}
FE_coeff_tors   {weights[4]}

#        Atom   Rii                             Rij_hb       rec_index
#        Type         epsii           solpar         epsij_hb    map_index
#                            vol                          hbond     bond_index
#        --     ----  -----  -------  --------  ---  ---  -  --  -- --
atom_par H      2.00  0.020   0.0000   0.00051  0.0  0.0  0  -1  -1  3  # Non H-bonding Hydrogen
atom_par HD     2.00  0.020   0.0000   0.00051  0.0  0.0  2  -1  -1  3  # Donor 1 H-bond Hydrogen
atom_par HS     2.00  0.020   0.0000   0.00051  0.0  0.0  1  -1  -1  3  # Donor S Spherical Hydrogen
atom_par C      4.00  0.150  33.5103  -0.00143  0.0  0.0  0  -1  -1  0  # Non H-bonding Aliphatic Carbon
atom_par A      4.00  0.150  33.5103  -0.00052  0.0  0.0  0  -1  -1  0  # Non H-bonding Aromatic Carbon
atom_par N      3.50  0.160  22.4493  -0.00162  0.0  0.0  0  -1  -1  1  # Non H-bonding Nitrogen
atom_par NA     3.50  0.160  22.4493  -0.00162  1.9  5.0  4  -1  -1  1  # Acceptor 1 H-bond Nitrogen
atom_par NS     3.50  0.160  22.4493  -0.00162  1.9  5.0  3  -1  -1  1  # Acceptor S Spherical Nitrogen
atom_par OA     3.20  0.200  17.1573  -0.00251  1.9  5.0  5  -1  -1  2  # Acceptor 2 H-bonds Oxygen
atom_par OS     3.20  0.200  17.1573  -0.00251  1.9  5.0  3  -1  -1  2  # Acceptor S Spherical Oxygen
atom_par F      3.09  0.080  15.4480  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Fluorine
atom_par Mg     1.30  0.875   1.5600  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Magnesium
atom_par MG     1.30  0.875   1.5600  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Magnesium
atom_par P      4.20  0.200  38.7924  -0.00110  0.0  0.0  0  -1  -1  5  # Non H-bonding Phosphorus
atom_par SA     4.00  0.200  33.5103  -0.00214  2.5  1.0  5  -1  -1  6  # Acceptor 2 H-bonds Sulphur
atom_par S      4.00  0.200  33.5103  -0.00214  0.0  0.0  0  -1  -1  6  # Non H-bonding Sulphur
atom_par Cl     4.09  0.276  35.8235  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Chlorine
atom_par CL     4.09  0.276  35.8235  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Chlorine
atom_par Ca     1.98  0.550   2.7700  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Calcium
atom_par CA     1.98  0.550   2.7700  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Calcium
atom_par Mn     1.30  0.875   2.1400  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Manganese
atom_par MN     1.30  0.875   2.1400  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Manganese
atom_par Fe     1.30  0.010   1.8400  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Iron
atom_par FE     1.30  0.010   1.8400  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Iron
atom_par Zn     1.48  0.550   1.7000  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Zinc
atom_par ZN     1.48  0.550   1.7000  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Zinc
atom_par Br     4.33  0.389  42.5661  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Bromine
atom_par BR     4.33  0.389  42.5661  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Bromine
atom_par I      4.72  0.550  55.0585  -0.00110  0.0  0.0  0  -1  -1  4  # Non H-bonding Iodine
atom_par Z      4.00  0.150  33.5103  -0.00143  0.0  0.0  0  -1  -1  0  # Non H-bonding covalent map
atom_par G      4.00  0.150  33.5103  -0.00143  0.0  0.0  0  -1  -1  0  # Ring closure Glue Aliphatic Carbon  # SF
atom_par GA     4.00  0.150  33.5103  -0.00052  0.0  0.0  0  -1  -1  0  # Ring closure Glue Aromatic Carbon   # SF
atom_par J      4.00  0.150  33.5103  -0.00143  0.0  0.0  0  -1  -1  0  # Ring closure Glue Aliphatic Carbon  # SF
atom_par Q      4.00  0.150  33.5103  -0.00143  0.0  0.0  0  -1  -1  0  # Ring closure Glue Aliphatic Carbon  # SF
atom_par OC     0.00  0.000   0.0000   0.00000  0.0  0.0  0  -1  -1  3  # Off-site Charge
    """
    with open(fpath, 'w') as f:
        f.write(fstr)

### Helper functions ###

def get_COM(file):
    if file.endswith('mol2') or file.endswith('xyz'):
        mol = load_one(file)
        ase_mol = Atoms(numbers=mol.atnums, positions=mol.atcoords / angstrom)
    elif file.endswith('sdf'):
        ase_mol = read_sdf(file)
    else:
        raise NotImplementedError(f"file extension not supported for {file}")

    return ase_mol.get_center_of_mass()

@contextlib.contextmanager
def set_directory(dirname: os.PathLike, mkdir: bool = False):
    """
    Set current workding directory within context

    Parameters
    ----------
    dirname : os.PathLike
        The directory path to change to
    mkdir: bool
        Whether make directory if `dirname` does not exist

    Yields
    ------
    path: Path
        The absolute path of the changed working directory

    Examples
    --------
    >>> with set_directory("some_path"):
    ...    do_something()
    """
    pwd = os.getcwd()
    path = Path(dirname).resolve()
    if mkdir:
        path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)
    yield path
    os.chdir(pwd)


def run_command(
        cmd: Union[List[str], str],
        raise_error: bool = True,
        input: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
) -> Tuple[int, str, str]:
    """
    Run shell command in subprocess
    Parameters
    ----------
    cmd: list of str, or str
        Command to execute
    raise_error: bool
        Wheter to raise an error if the command failed
    input: str, optional
        Input string for the command
    timeout: int, optional
        Timeout for the command
    **kwargs:
        Arguments in subprocess.Popen

    Raises
    ------
    AssertionError:
        Raises if the error failed to execute and `raise_error` set to `True`

    Return
    ------
    return_code: int
        The return code of the command
    out: str
        stdout content of the executed command
    err: str
        stderr content of the executed command
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    elif isinstance(cmd, list):
        cmd = [str(x) for x in cmd]

    sub = subprocess.Popen(
        args=cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs
    )
    if input is not None:
        sub.stdin.write(bytes(input, encoding=sys.stdin.encoding))
    try:
        out, err = sub.communicate(timeout=timeout)
        return_code = sub.poll()
    except subprocess.TimeoutExpired:
        sub.kill()
        print("Command %s timeout after %d seconds" % (cmd, timeout))
        return 999, "", ""  # 999 is a special return code for timeout
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    if raise_error and return_code != 0:
        raise CommandExecuteError("Command %s failed: \n%s" % (cmd, err))
    return return_code, out, err

class CommandExecuteError(Exception):
    """
    Exception for command line exec error
    """

    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg

    def __repr__(self):
        return self.msg
