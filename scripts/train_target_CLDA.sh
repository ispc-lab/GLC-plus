gpuid=${1:-0}
random_seed=${2:-2021}

# export CUDA_VISIBLE_DEVICES=$gpuid

"""
GLC_ORIG 
"""

lam_psd=0.30
echo "CLDA Adaptation ON Office"
python train_target.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"

lam_psd=1.50
echo "CLDA Adaptation ON Office-Home"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type CLDA --note "GLC_ORIG"


"""
GLC_PLUS
"""

lam_psd=0.30
echo "CLDA Adaptation ON Office"
python train_target.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"

lam_psd=1.50
echo "CLDA Adaptation ON Office-Home"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type CLDA --note "GLC_PLUS"