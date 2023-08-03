# InteractionGraphNet (IGN) Retrain
  InteractionGraphNet: a Novel and Efficient Deep Graph Representation Learning Framework for Accurate Protein-Ligand Interaction Prediction and Large-scale Structure-based Virtual Screening. 
  Refer to https://github.com/zjujdj/InteractionGraphNet.git for detailed training & prediction scripts.

# Environment used
  conda create --prefix xxx --file env.yml

# Training settings
  python ./codes/ign_train.py --gpuid 0 --epochs 500 --batch_size 128 --graph_feat_size 128 --num_layers 2 --outdim_g3 128 --d_FC_layer 128 --repetitions 3 --lr 0.001 --l2 0.00001 --dropout 0.2

# Binding affinity prediction with retrained model
  python codes/prediction.py --input_path xxx  --graph_ls_path xxx --graph_dic_path xxx --model_path model_save/2023-06-23_00_36_38_49161.pth
