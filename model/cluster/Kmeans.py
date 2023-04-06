import sys
import torch
import torch.nn as nn

from sklearn.cluster import KMeans

class Cluster_Kmean(object):

    def __init__(self, K, n_init=10, max_iter=250) -> None:
        super().__init__()
        self.K = K
        self.n_init = n_init
        self.max_iter = max_iter

        self.u = None
        self.M = None

    def cluster(self, embed):
        embed_up = embed.detach().cpu().numpy()
        clustering = KMeans(n_cluster=self.K, n_init=self.n_init, max_iter=self.max_iter)

        self.M = clustering.labels_
        self.u = self._compute_centers(self.M, embed_up)

    def get_loss(self, embed):
        loss = torch.Tensor([0.])
        for i, clusteridx in enumerate(self.M):
            x = embed[i]
            c = self.u[clusteridx]
            difference = x - c
            err = torch.sum(torch.mul(difference, difference))
            loss += err
        
        return loss
    
    def get_membership(self):
        return self.M
    
    def _compute_centers(self, labels, embed):
        clusters = {}
        for i, lbl in enumerate(labels):
            if clusters.get(lbl) is None:
                clusters[lbl] = []
        clusters[lbl].append(torch.FloatTensor(embed[i]))

        centers = {}
        for k in clusters:
            all_embed = torch.stack(clusters[k])
            center = torch.mean(all_embed, 0)
            centers[k] = center
        
        return centers
