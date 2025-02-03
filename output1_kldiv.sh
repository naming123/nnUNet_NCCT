export nnUNet_preprocessed=/data/nnUNet_Dataset/nnUNet_preprocessed
export nnUNet_raw=/data/nnUNet_Dataset/nnUNet_raw
export nnUNet_results=/data/nnUNet_Dataset/nnUNet_results

export CUDA_VISIBLE_DEVICES=0

nnUNetv2_train 505 3d_fullres 0 -tr KDTrainer_KLdiv_osm_t2