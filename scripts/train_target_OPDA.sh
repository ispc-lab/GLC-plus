gpuid=${1:-0}
random_seed=${2:-2021}

"""
GLC_ORIG 
"""

lam_psd=0.30
echo "OPDA Adaptation ON VisDA"
python train_target.py --backbone_arch resnet50 --lr 0.0001 --dataset VisDA --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA  --epochs 20 --note "GLC_ORIG"

echo "OPDA Adaptation ON Office"
python train_target.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"

lam_psd=1.50
echo "OPDA Adaptation ON Office-Home"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --note "GLC_ORIG"

echo "OPDA Adaptation ON DomainNet"
python train_target.py --dataset DomainNet --s_idx 0 --t_idx 1 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_ORIG"
python train_target.py --dataset DomainNet --s_idx 0 --t_idx 2 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_ORIG"
python train_target.py --dataset DomainNet --s_idx 1 --t_idx 0 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_ORIG"
python train_target.py --dataset DomainNet --s_idx 1 --t_idx 2 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_ORIG"
python train_target.py --dataset DomainNet --s_idx 2 --t_idx 0 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_ORIG"
python train_target.py --dataset DomainNet --s_idx 2 --t_idx 1 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_ORIG"

"""
GLC_PLUS
"""

lam_psd=0.30
echo "OPDA Adaptation ON VisDA"
python train_target.py --backbone_arch resnet50 --lr 0.0001 --dataset VisDA --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA  --epochs 20 --note "GLC_PLUS"

echo "OPDA Adaptation ON Office"
python train_target.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"

lam_psd=1.50
echo "OPDA Adaptation ON Office-Home"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --lan_reg 1.0 --target_label_type OPDA --note "GLC_PLUS"

echo "OPDA Adaptation ON DomainNet"
python train_target.py --dataset DomainNet --s_idx 0 --t_idx 1 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_PLUS"
python train_target.py --dataset DomainNet --s_idx 0 --t_idx 2 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_PLUS"
python train_target.py --dataset DomainNet --s_idx 1 --t_idx 0 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_PLUS"
python train_target.py --dataset DomainNet --s_idx 1 --t_idx 2 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_PLUS"
python train_target.py --dataset DomainNet --s_idx 2 --t_idx 0 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_PLUS"
python train_target.py --dataset DomainNet --s_idx 2 --t_idx 1 --lr 0.0001  --lam_psd $lam_psd --lan_reg 0.0 --target_label_type OPDA --epochs 10 --note "GLC_PLUS"
