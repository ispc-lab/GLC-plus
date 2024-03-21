import os
from tqdm import tqdm 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def train_transform(resize_size=256, crop_size=224,):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]) 
    
def test_transform(resize_size=256, crop_size=224,):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
    
    
'''
assume classes across domains are the same.
[0 1 ............................................................................ N - 1]
|---- common classes --||---- source private classes --||---- target private classes --|

|--------------------------------------------------------------|
|                       DATASET PARTITION                      |
|--------------------------------------------------------------|
|DATASET    |        class split(com/sou_pri/tar_pri)          |
|--------------------------------------------------------------|
|DATASET    |    CLDA   |    PDA    |    OSDA     | OPDA/UniDA |
|--------------------------------------------------------------|
|Office-31  |  31/0/0   |  10/21/0  |  10/0/11    |  10/10/11  |
|--------------------------------------------------------------|
|OfficeHome |  65/0/0   |  25/40/0  |  25/0/40    |  10/5/50   |
|--------------------------------------------------------------|
|VisDA-C    |     -     |   6/6/0   |   6/0/6     |   6/3/3    |
|--------------------------------------------------------------|  
|DomainNet  |     -     |     -     |     -       | 150/50/145 |
|--------------------------------------------------------------|
'''

class SFUniDADataset(Dataset):
    
    def __init__(self, args, data_dir, data_list, d_type, preload_flg=True) -> None:
        super(SFUniDADataset, self).__init__()
        
        self.d_type = d_type
        self.dataset = args.dataset
        self.preload_flg = preload_flg
        
        self.shared_class_num = args.shared_class_num
        self.source_private_class_num = args.source_private_class_num
        self.target_private_class_num = args.target_private_class_num 
        
        self.shared_classes = [i for i in range(args.shared_class_num)]
        self.source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        
        if args.dataset == "Office" and args.target_label_type == "OSDA":
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num + 10 for i in range(args.target_private_class_num)]
        else:
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
            
        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes
        
        self.data_dir = data_dir 
        self.data_list = [item.strip().split() for item in data_list]
        
        # Filtering the data_list
        if self.d_type == "source":
            # self.data_dir = args.source_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.source_classes]
        else:
            # self.data_dir = args.target_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.target_classes]
            
        self.pre_loading()
        
        self.train_transform = train_transform()
        self.test_transform = test_transform()
        
    def pre_loading(self):
        if "Office" in self.dataset and self.preload_flg:
            self.resize_trans = transforms.Resize((256, 256))
            print("Dataset Pre-Loading Started ....")
            self.img_list = [self.resize_trans(Image.open(os.path.join(self.data_dir, item[0])).convert("RGB")) for item in tqdm(self.data_list, ncols=60)]
            print("Dataset Pre-Loading Done!")
        else:
            pass
    
    def load_img(self, img_idx):
        img_f, img_label = self.data_list[img_idx]
        if "Office" in self.dataset and self.preload_flg:
            img = self.img_list[img_idx]
        else:
            img = Image.open(os.path.join(self.data_dir, img_f)).convert("RGB")        
        return img, img_label
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, img_idx):
        
        img, img_label = self.load_img(img_idx)
        
        if self.d_type == "source":
            img_label = int(img_label)
            pri_label = -1
        else:
            # pri_label = int(img_label) - self.shared_class_num - self.source_private_class_num
            # pri_label = pri_label if pri_label >= 0 else -1
            pri_label = self.target_private_classes.index(int(img_label)) if int(img_label) in self.target_private_classes else -1
            
            img_label = int(img_label) if int(img_label) in self.source_classes else len(self.source_classes)

        img_train = self.train_transform(img)
        img_test = self.test_transform(img)

        return img_train, img_test, img_label, pri_label, img_idx
    
if __name__ == "__main__":
    
    import torch
    import argparse
    import numpy as np
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="VisDA")
    parser.add_argument("--source_data_dir", type=str, default="./data/VisDA/train/")
    parser.add_argument("--target_data_dir", type=str, default="./data/VisDA/validation/")
    parser.add_argument("--shared_class_num", type=int, default=6)
    parser.add_argument("--source_private_class_num", type=int, default=0)
    parser.add_argument("--target_private_class_num", type=int, default=6)
    
    args = parser.parse_args()
    vis_source_imglist = open(os.path.join(args.source_data_dir, "image_unida_list.txt"), "r").readlines()
    vis_target_imglist = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    vis_source_dsize = len(vis_source_imglist)
    vis_source_tr_size = int(vis_source_dsize * 0.9)
    vis_source_va_size = vis_source_dsize - vis_source_tr_size
    vis_sr_tr_list, vis_sr_va_list = torch.utils.data.random_split(vis_source_imglist, [vis_source_tr_size, vis_source_va_size],
                                                                    generator=torch.Generator().manual_seed(2021))
    
    vis_source_train_dataset = SFUniDADataset(args, args.source_data_dir, vis_sr_tr_list, d_type="source")
    vis_source_test_dataset = SFUniDADataset(args, args.source_data_dir, vis_sr_va_list, d_type="source")
    vis_target_test_dataset = SFUniDADataset(args, args.target_data_dir, vis_target_imglist, d_type="target")
    
    print("source_train_size: ", len(vis_source_train_dataset))
    print("source_test_size : ", len(vis_source_test_dataset))
    print("target_test_size : ", len(vis_target_test_dataset))
    
    vis_source_train_loader = DataLoader(vis_source_train_dataset, batch_size=64, shuffle=True, drop_last=False)
    vis_target_test_loader = DataLoader(vis_target_test_dataset, batch_size=32, shuffle=True, drop_last=False)
    
    
    for img_train, img_test, img_label, img_idx in tqdm(vis_target_test_loader):

        print(img_train.shape)
        print(img_label.shape)
        print(img_label)
        break
        