"""
@Time: 03/2023
@Author: Nancy Xingyi Guan

This script is for reweighting vina using new splits of the PDBBind dataset
"""

from scipy.stats import spearmanr

from generic_vina_ad4 import *

dir_dict = {"core": "CASF-2016/coreset", "refined": "refined-set", "general":"v2020-other-PL"}
cwd = os.getcwd()
vina_weights_original = {'gauss1':-0.035579 ,'gauss2':-0.005156,'repulsion':0.840245,'hydrophobic':-0.035069,
                        'hydrogen':-0.587439,'rot':0.05846}

def get_list(cl_level=["CL1","CL2","CL2"],mpro_rm = False, casf_rm = False, casf_only = False):
    """
    get list of train, test and validation pdbs from the UCBsplit data as dictionaries
    """
    if mpro_rm or casf_rm:
        csv = "UCBSplit_training_data.csv"
    else:
        csv = "dataset/UCBSplit.csv"

    data = pd.read_csv(csv)
    data.rename(columns={"Unnamed: 0": 'pdb'}, inplace=True)
    if casf_only:
        casf_data = data[(data['category']=="core") & (~data.covalent)]
        casf_dic = {}
        for k in casf_data.columns:
            casf_dic[k] = np.array(casf_data[k])
        return casf_dic

    test_dic = {}
    test = data[(data['new_split'] == "test") & (~data.covalent)]
    if mpro_rm:
        test = test[test['remove_for_mpro'] == False]
    if casf_rm:
        test = test[test['remove_for_casf'] == False]
    test = test[test[cl_level[2]] == True]
    for k in test.columns:
        test_dic[k] = np.array(test[k])

    train_dic = {}
    train = data[(data['new_split'] == "train") & (~data.covalent)]
    if mpro_rm:
        train = train[train['remove_for_mpro'] == False]
    if casf_rm:
        train = train[train['remove_for_casf'] == False]
    train = train[train[cl_level[0]]==True]
    for k in train.columns:
        train_dic[k] = np.array(train[k])

    val_dic = {}
    val = data[(data['new_split'] == "val") & (~data.covalent)]
    if mpro_rm:
        val = val[val['remove_for_mpro'] == False]
    if casf_rm:
        val = val[val['remove_for_casf'] == False]
    val = val[val[cl_level[1]]==True]
    for k in val.columns:
        val_dic[k] = np.array(val[k])

    if mpro_rm:
        mpro_dict = {}
        mpro = data[(data['remove_for_mpro'] == True) & (~data.covalent)]
        for k in val.columns:
            mpro_dict[k] = np.array(mpro[k])
        return train_dic, test_dic, val_dic, mpro_dict
    if casf_rm:
        casf_dict = {}
        casf = data[(data['remove_for_casf'] == True) & (~data.covalent)]
        for k in val.columns:
            casf_dict[k] = np.array(casf[k])
        return train_dic, test_dic, val_dic, casf_dict
    return train_dic, test_dic, val_dic

def compute_delta_G(K, option = "pKa"):
    #\Delta G = R*T*ln{Kd/c0} = -RTln(Ka*c0)  (C0 = 1mol/L)
    R = 1.987204 * 1e-3 #kcal⋅K−1⋅mol−1
    if option == "pKa": # this is log10
        return -R * 298 * K * np.log(10)
    elif option == "Kd":
        return R * 298 * np.log(K)
    elif option == "Ki":
        raise NotImplementedError
    else:
        raise ValueError("Supported options are [pKa,Ki,Kd]")

def get_delta_G(name,category):
    """
    get deltaG from PDBBind dataset
    """
    if category == 'core':
        with open("CASF-2016/power_scoring/CoreSet.dat",'r') as f:
            for l in f.readlines():
                if l.startswith(name):
                    logKa = float(l.split()[3])
                    return compute_delta_G(logKa,option="pKa")
    elif category == 'refined':
        with open("refined-set/index/INDEX_refined_data.2020",'r') as f:
            for l in f.readlines():
                if l.startswith(name):
                    logKa = float(l.split()[3])
                    return compute_delta_G(logKa, option="pKa")
    elif category == 'general':
        with open("v2020-other-PL/index/INDEX_general_PL_data.2020",'r') as f:
            for l in f.readlines():
                if l.startswith(name):
                    logKa = float(l.split()[3])
                    return compute_delta_G(logKa, option="pKa")
    else:
        raise ValueError(f"Unknow category: {category}")

def vina_preprocess_pdbbind(folder):
    name = folder.split('/')[-1]
    vina_preprocess(folder,f"{name}_ligand.mol2",f"{name}_protein.pdb")


def vina_component_binary(folder,w = Vina_weights_original_tuple):
    """
    Run binary of vina to get score of each component
    """
    # Due to numerical instability of vina, cannot use 1 as weights, therefore first use original vina weight and do validation later
    name = folder.split('/')[-1]
    if not os.path.isfile(f"{folder}/{name}_protein.pdbqt") or not os.path.isfile(f"{folder}/{name}_ligand.pdbqt"):
        vina_preprocess_pdbbind(folder)
    with set_directory(folder):
        ligand_COM = get_COM(f"{name}_ligand.mol2")
        ligand_scores = []
        for i in range(6):
            weights = np.zeros(6)
            weights[i] = w[i]
            write_config_vina(f"{name}_ligand.pdbqt", f"{name}_protein.pdbqt",ligand_COM,config_fp = "config.txt",weights=weights)
            cmd = f"{VINA_BINARY} --config config.txt --score_only"
            try:
                code, out, err = run_command(cmd, timeout=100)
            except CommandExecuteError as e:
                print(f"error in {folder}: {e}")
                print("out: ",out)
                print("err: ", err)
                raise

            # obtain docking score from the results
            strings = re.split('Estimated Free Energy of Binding   :', out)
            line = strings[1].split('\n')[0]
            energy = float(line.strip().split()[0]) / w[i]
            ligand_scores.append(energy)

    return ligand_scores


def vina_binary_pdbbind(folder, weights=None, save_out_file = False):
    name = folder.split('/')[-1]
    if not os.path.isfile(f"{folder}/{name}_protein.pdbqt") or not os.path.isfile(f"{folder}/{name}_ligand.pdbqt"):
        vina_preprocess_pdbbind(folder)
    with set_directory(folder):
        ligand_COM = get_COM(f"{name}_ligand.mol2")
        write_config_vina(f"{name}_ligand.pdbqt", f"{name}_protein.pdbqt", ligand_COM,config_fp = "config.txt",weights=weights)
        cmd = f"{VINA_BINARY} --config config.txt --score_only"
        try:
            code, out, err = run_command(cmd, timeout=100)
        except CommandExecuteError as e:
            print(f"error in {folder}: {e}")
            print("out: ",out)
            print("err: ", err)
            raise

        # obtain docking score from the results
        strings = re.split('Estimated Free Energy of Binding   :', out)
        line = strings[1].split('\n')[0]
        energy = float(line.strip().split()[0])
        if save_out_file:
            with open("vina.out", 'w') as f:
                f.write(out)

    return energy


def get_vina_components(names,categorys,w=Vina_weights_original_tuple,outfile = "vina_components.csv"):
    with open(outfile,'w') as f:
        f.write("pdb,category,gauss1,gauss2,repulsion,hydrophobic,hydrogen,rot,deltaG\n")
        for i,name in enumerate(names):
            path = dir_dict[categorys[i]]
            print(f"{path}/{name}")
            try:
                scores = vina_component_binary(os.path.join(path, name), w=w)
            except Exception as e:
                print(f"Error for {categorys[i]} {name}: {e}")
                scores = [None,None,None,None,None,None]
                os.chdir(cwd)
            delta_G = get_delta_G(name,categorys[i])
            f.write(f"{name},{categorys[i]},")
            f.write(",".join([str(s) for s in scores]))
            f.write(f",{delta_G}\n")

def get_old_and_new_score(name,category, weights = None):
    path = dir_dict[category]
    try:
        old_score = vina_binary_pdbbind(os.path.join(path, name))
        new_score = vina_binary_pdbbind(os.path.join(path, name), weights)
    except Exception as e:
        print(f"Error for {category} {name}: {e}")
        os.chdir(cwd)
        return [name,category,None,None]
    return [name,category,old_score,new_score]

def test_new_vina(names,categorys,weights,outfile = "vina_scores.csv",std_output = False):
    with mp.Pool() as pool:  # use all available cores, otherwise specify the number you want as an argument
        results = pool.starmap(partial(get_old_and_new_score, weights=weights), zip(names,categorys))
    with open(outfile, 'w') as f:
        if std_output:
            f.write("PDBID,old_model_prediction,new_model_prediction,deltaG\n")
            for result in results:
                name, category, old_score, new_score = result
                delta_G = get_delta_G(name, category)
                f.write(f"{name},{old_score},{new_score},{delta_G}\n")
        else:
            f.write("pdb,category,old vina,new_vina,deltaG\n")
            for result in results:
                name, category, old_score, new_score = result
                delta_G = get_delta_G(name, category)
                f.write(f"{name},{category},{old_score},{new_score},{delta_G}\n")

def test_new_vina_from_comp(names,new_weights,f_comp,outfile = "vina_scores.csv", std_output = False):
    df = pd.read_csv(f_comp)
    df = df[~(df['gauss1'] == "None")]
    df = df[df['pdb'].isin(names)]
    terms_to_fit = ['gauss1','gauss2','repulsion','hydrophobic','hydrogen',]
    if 0.05846 in new_weights:
        new_weights = list(new_weights)
        new_weights.remove(0.05846)

    X = df[terms_to_fit].to_numpy(dtype=float)
    y = df['deltaG'].to_numpy(dtype=float)
    old_vina_pred = np.sum([X[:, i] * Vina_weights_original_tuple[i] for i in range(len(terms_to_fit))], axis=0)
    vina_pred = np.sum([X[:, i] * new_weights[i] for i in range(len(terms_to_fit))], axis=0)
    if len(new_weights) == 6:
        vina_pred += new_weights[-1]
    if std_output:
        with open(outfile, 'w') as f:
            f.write("PDBID,old_model_prediction,new_model_prediction,deltaG\n")
            for i in range(df.shape[0]):
                row = df.iloc[i]
                f.write(f"{row['pdb']},{old_vina_pred[i]},{vina_pred[i]},{y[i]}\n")
    else:
        with open(outfile, 'w') as f:
            f.write("pdb,category,old_vina,new_vina,deltaG\n")
            for i in range(df.shape[0]):
                row = df.iloc[i]
                f.write(f"{row['pdb']},{row['category']},{old_vina_pred[i]},{vina_pred[i]},{y[i]}\n")

def test_additive_score(score_csv,component_csv, new_weights):
    old_weights = [-0.035579, -0.005156, 0.840245, -0.035069, -0.587439] #, 0.05846]
    df_score = pd.read_csv(score_csv)
    df_comp = pd.read_csv(component_csv)
    df_score = df_score[~(df_score['old vina'] == "None")]
    df_score['old vina'] = df_score['old vina'].astype(float)
    df_score['new_vina'] = df_score['new_vina'].astype(float)
    df_score['deltaG'] = df_score['deltaG'].astype(float)
    for index, row in df_score.iterrows():
        name = row["pdb"]
        category = row['category']
        old_score = row['old vina']
        new_score = row['new_vina']
        row_comps = df_comp[df_comp['pdb'] == name]
        comps = []
        for k in ['gauss1','gauss2','repulsion','hydrophobic','hydrogen',]:
            comps.append(float(row_comps[k]))
        diff_old = old_score - np.sum([comps[i] * old_weights[i] for i in range(5)])
        diff_new = new_score - np.sum([comps[i] * new_weights[i] for i in range(5)])
        if np.abs(diff_old) > 0.05:
            print(name, category)
            print("Comps: ",comps)
            print(f"Old score: {old_score}, added sum: {np.sum([comps[i] * old_weights[i] for i in range(5)])}, diff: {diff_old}")
        if np.abs(diff_new) > 0.05:
            print(name, category)
            print("Comps: ",comps)
            print(f"New score: {new_score}, added sum: {np.sum([comps[i] * new_weights[i] for i in range(5)])}, diff: {diff_new}")
            path = dir_dict[category]
            test(os.path.join(path, name),w=new_weights)
            raise ValueError()




### weight refit ###

def get_weight_with_rep(rep=0,component_file = "vina_components.csv"):
    """
    From a file that store score of each component, optimize the weights of vina
    rep: contraint of repulsion weight, may need to force this term to be larger than 0
    """
    df = pd.read_csv(component_file)
    df_vina = df[~(df['gauss1']=="None")]

    terms_to_fit = ['gauss1','gauss2','repulsion','hydrophobic','hydrogen',]

    X = df_vina[terms_to_fit].to_numpy(dtype=float)
    y = df_vina['deltaG'].to_numpy(dtype=float)
    guess = [v for k,v in vina_weights_original.items() if k in terms_to_fit]

    def objective(guess):
        vina_pred = np.sum([X[:,i] * guess[i] for i in range(len(terms_to_fit))], axis = 0)
        return np.mean(np.abs(np.array(y-vina_pred)))

    print(rep)

    result = minimize(objective,
                      guess,
                      method='Nelder-Mead',
                      # bounds = [(None,None),(None,None),(rep,np.inf),(None,None),(None,None)]
                      bounds=[(None, None), (None, None), (rep, np.inf), (None, None), (None, None)]
                              )
    print(result)
    if result.success:
        print(result.x)
        vina_weights = {k:result.x[i] for i,k in enumerate(terms_to_fit)}
        print(vina_weights)
        print("MAE: ", objective(result.x))
    weights = list(result.x)
    weights.append(0.05846)
    return weights



def get_error_from_score_file(f_score):
    df = pd.read_csv(f_score)
    df = df[~(df['old_vina'] == "None")]
    df['old_vina'] = df['old_vina'].astype(float)
    df['new_vina'] = df['new_vina'].astype(float)
    df['deltaG'] = df['deltaG'].astype(float)
    r1 = spearmanr(df['old_vina'], df['deltaG'])[0]
    mae1 = np.mean(np.abs(df['old_vina'] - df['deltaG']))
    r2 = spearmanr(df['new_vina'], df['deltaG'])[0]
    mae2 = np.mean(np.abs(df['new_vina'] - df['deltaG']))
    return [r1,mae1,r2,mae2]



if __name__ =='__main__':
    ### Finding new weights of vina

    train_dic, test_dic, val_dic= get_list(cl_level=["CL1","CL2","CL2"])

    get_vina_components(train_dic['pdb'], train_dic['category'],
                        w=Vina_weights_original_tuple, outfile="vina_components.csv")
    new_weights = get_weight_with_rep(rep=0, component_file="vina_components.csv")
    print(new_weights)

    test_new_vina(test_dic['pdb'], test_dic['category'], new_weights,
                  outfile=f"vina_score.csv",std_output=True)



