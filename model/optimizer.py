import sys
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F

from sklearn.cluster import KMeans

def compute_attribute_loss(lossfn, features, recon, outlier_wt):
    loss = lossfn(features, recon)
    loss = loss.sum(dim=1)

    outlier_wt = torch.log(1/outlier_wt)

    attr_loss = torch.sum(torch.mul(outlier_wt, loss))

    return attr_loss

def compute_structure_loss(adj, embed, outlier_wt):   # embed is the Z matrix, outlier_wt is the tensor to represent Osi
    # to compute F(x_i).F(x_j)
    embeddot = torch.mm(embed, torch.transpose(embed, 0, 1))    # Z·Z^T

    adj_tensor = adj.to_dense()

    # compute A_ij - F(x_i)*F(x_j)
    difference = adj_tensor - embeddot                          # A-Z·Z^T
    # square difference and sum
    # torch.mm只适用于二维矩阵，而torch.matmul可以适用于高维，作用都是矩阵乘法
    # torch.mul得到的结果是哈达玛乘积，可以用于矩阵与矩阵、矩阵与数字
    loss = torch.sum(torch.mul(difference, difference), dim=1)  # |A-Z·Z^T|^2, compress the row
    # 这里的sun要压缩列是因为公式里面的Ai和ZZi都表示第i行

    outlier_wt = torch.log(1/outlier_wt)                        # \sum 1/log(1/Osi)

    struct_loss = torch.sum(torch.mul(outlier_wt, loss))        # cal the final loss

    return struct_loss

def update_o1(adj, embed):
    # to compute F(x_i).F(x_j)
    embed = embed.data
    embeddot = torch.mm(embed, torch.transpose(embed, 0, 1))

    adj_tensor = adj.to_dense()

     # compute A_ij - F(x_i)*F(x_j)
    difference = adj_tensor - embeddot
    # square difference and sum
    error = torch.sum(torch.mul(difference, difference), dim=1)

    # compute the denominator
    normalization_factor = torch.sum(error)

    # normalize the errors
    o1 = error/normalization_factor

    return o1

def update_o2(features, recon):
    features = features.data
    recon = recon.data
    # error = x - F(G(x))
    error = features - recon
    # error now = (x - F(G(x)))^2, summed across dim 1
    error = torch.sum(torch.mul(error, error), dim=1)

    # compute the denominator
    normalization_factor = torch.sum(error)

    # normalize the errors
    o2 = error/normalization_factor

    return o2
