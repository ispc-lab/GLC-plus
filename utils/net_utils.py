import os
import sys 
import logging 

import torch 
import faiss 
import random 

import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 

from tqdm import tqdm 
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def set_random_seed(seed=0):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def log_args(args):
    s = "\n==========================================\n"
    
    s += ("python" + " ".join(sys.argv) + "\n")
    
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    
    s += "==========================================\n"
    
    return s

def set_logger(args, log_name="train_log.txt"):
    
    log_format = "%(asctime)s [%(levelname)s] - %(message)s"
    # creating logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # file logger handler
    if args.test:
        # Append the test results on existing logging file.
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="a")
        file_format = logging.Formatter("%(message)s")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
    else:
        # Init the logging file.
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="w")
        
        file_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
    
    # terminal logger handler
    terminal_handler = logging.StreamHandler()
    terminal_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(terminal_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler)
    if not args.test:
        logger.debug(log_args(args))
    
    return logger

def compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flag=True, pred_unc_all=None, open_thresh=0.5):
    
    # class_list:
    #   :source [0, 1, ..., N_share - 1, ...,           N_share + N_src_private - 1]
    #   :target [0, 1, ..., N_share - 1, N_share + N_src_private + N_tar_private -1]
    # gt_label_all [N]
    # pred_cls_all [N, C]
    # open_flag    True/False
    # pred_unc_all [N], if exists. [0~1.0]
    
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)
    pred_label_all = torch.max(pred_cls_all, dim=1)[1] #[N]
    
    if open_flag:
        cls_num = pred_cls_all.shape[1]
        
        if pred_unc_all is None:
            # If there is not pred_unc_all tensor,
            # We normalize the Shannon entropy to [0, 1] to denote the uncertainty.
            pred_unc_all = Entropy(pred_cls_all)/np.log(cls_num)# [N]

        unc_idx = torch.where(pred_unc_all > open_thresh)[0]
        pred_label_all[unc_idx] = cls_num # set these pred results to unknown

    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_all == label)[0]
        correct_idx = torch.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))

    per_class_acc = per_class_correct / (per_class_num + 1e-5)

    if open_flag:
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    else:
        # if args.dataset == "VisDA":
        #     known_acc = per_class_acc.mean()
        # else:
        known_acc = per_class_correct.sum() / (per_class_num.sum() + 1e-5)
        unknown_acc = 0.0
        h_score = 0.0

    return h_score, known_acc, unknown_acc, per_class_acc

def compute_h_score_with_private_discovery(args, class_list, gt_label_all, pred_cls_all, gt_private_all, embed_feat_all, open_flag=True, pred_unc_all=None, open_thresh=0.5, discovery_flag=True):
    # class_list:
    #   :source [0, 1, ..., N_share - 1, ...,           N_share + N_src_private - 1]
    #   :target [0, 1, ..., N_share - 1, N_share + N_src_private + N_tar_private -1]
    # gt_label_all [N]
    # pred_cls_all [N, C]
    # open_flag    True/False
    # pred_unc_all [N], if exists. [0~1.0]
    
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)
    pred_label_all = torch.max(pred_cls_all, dim=1)[1] #[N]
    
    if open_flag:
        cls_num = pred_cls_all.shape[1]
        
        if pred_unc_all is None:
            # If there is not pred_unc_all tensor,
            # We normalize the Shannon entropy to [0, 1] to denote the uncertainty.
            pred_unc_all = Entropy(pred_cls_all)/np.log(cls_num)# [N]

        unc_idx = torch.where(pred_unc_all > open_thresh)[0]
        pred_label_all[unc_idx] = cls_num # set these pred results to unknown

    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_all == label)[0]
        correct_idx = torch.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))

    per_class_acc = per_class_correct / (per_class_num + 1e-5)

    if open_flag:
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    else:
        # if args.dataset == "VisDA":
        #     known_acc = per_class_acc.mean()
        # else:
        known_acc = per_class_correct.sum() / (per_class_num.sum() + 1e-5)
        unknown_acc = 0.0
        h_score = 0.0

    if open_flag and discovery_flag:
        # the following evaluation is based on novel categroies discovery setting.
        
        gt_private_idx = torch.where(gt_label_all == cls_num)[0]
        # pred_private_idx = torch.where(pred_label_all == cls_num)[0]
        gt_private_label_bank = gt_private_all[gt_private_idx].cpu().numpy()# may involve both known and unknown novel categories.
        
        gt_privatea_class_number = args.target_private_class_num
        
        # gt_private_label_bank = gt_private_all[gt_private_idx].cpu().numpy()
        # pred_private_label_bank = pred_label_all[pred_private_idx]
        pred_private_feat = embed_feat_all[gt_private_idx].cpu().numpy() # [N, D]
        kmeans = KMeans(n_clusters=gt_privatea_class_number, random_state=0).fit(pred_private_feat)
        pred_private_label_bank = label_matching(gt_private_label_bank, kmeans.labels_)
        
        private_discovery_acc = (gt_private_label_bank == pred_private_label_bank).sum() / len(gt_private_label_bank)
    else:
        private_discovery_acc = 0.0
    
    return h_score, known_acc, unknown_acc, private_discovery_acc
  
  
def label_matching(gt_labels, pred_labels):
    # Create a confusion matrix (contingency table)
    contingency_matrix = confusion_matrix(gt_labels, pred_labels)
    
    # The Hungarian algorithm finds the minimum cost assignment, so we need to
    # convert our problem into a minimization problem. We do this by subtracting
    # our contingency matrix from a matrix of all max values.
    cost_matrix = contingency_matrix.max() - contingency_matrix
    
    # Adjust the contingency matrix to be square
    num_ground_truth = len(np.unique(gt_labels))
    num_clusters = len(np.unique(pred_labels))

    if num_clusters > num_ground_truth:
        # Add dummy columns
        contingency_matrix = np.pad(contingency_matrix, ((0, 0), (0, num_clusters - num_ground_truth)), mode='constant')
    elif num_clusters < num_ground_truth:
        # Add dummy rows
        contingency_matrix = np.pad(contingency_matrix, ((0, num_ground_truth - num_clusters), (0, 0)), mode='constant')

    # Convert to a cost matrix
    cost_matrix = contingency_matrix.max() - contingency_matrix

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create a mapping from cluster labels to ground-truth labels, ignoring dummy assignments
    label_mapping = {}
    for cluster_label, ground_truth_label in zip(col_ind, row_ind):
        if cluster_label < num_clusters and ground_truth_label < num_ground_truth:
            label_mapping[cluster_label] = ground_truth_label

    # Apply the mapping to the cluster labels
    mapped_labels = [label_mapping.get(label, -1) for label in pred_labels]  # -1 for unmapped clusters
    
    return mapped_labels

def computer_pred_class_prior_weight(pred_cls_all, topk_ratio=0.3):
    
    cls_num = pred_cls_all.shape[1]
    data_num = pred_cls_all.shape[0]
    sample_num = int(data_num / cls_num * topk_ratio)
    topk_pred_sample, _ = torch.topk(pred_cls_all, k=sample_num, dim=0, largest=True)
    pred_class_prior = torch.mean(topk_pred_sample, dim=0)
    
    return pred_class_prior

class CrossEntropyLabelSmooth(nn.Module):
    
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """      

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets, applied_softmax=True):
        """
        Args:
            inputs: prediction matrix (after softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes).
        """
        if applied_softmax:
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)
        
        if inputs.shape != targets.shape:
            # this means that the target data shape is (B,)
            targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
         
        if self.reduction:
            return loss.mean()
        else:
            return loss