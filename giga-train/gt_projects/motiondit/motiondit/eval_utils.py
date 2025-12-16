from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance

import scipy
import numpy as np
class TM2TMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "matching, fid, and diversity scores"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []
        # Matching scores
        self.add_state("Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Matching_score",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
        for k in range(1, top_k + 1):
            self.add_state(
                f"R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"R_precision_top_{str(k)}")
        for k in range(1, top_k + 1):
            self.add_state(
                f"gt_R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")

        self.metrics.extend(self.Matching_metrics)

        # Fid
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("FID")

        # Diversity
        self.add_state("Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "gt_Diversity"])

        # chached batches
        self.add_state("text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_texts = torch.cat(self.text_embeddings,
                              axis=0).cpu()[shuffle_idx, :]# torch.Size([686, 512])
        all_genmotions = torch.cat(self.recmotion_embeddings,
                                   axis=0).cpu()[shuffle_idx, :]# torch.Size([686, 512])
        all_gtmotions = torch.cat(self.gtmotion_embeddings,
                                  axis=0).cpu()[shuffle_idx, :]# torch.Size([686, 512])

        # Compute r-precision
        assert count_seq > self.R_size, f"count_seq: {count_seq}, R_size: {self.R_size}"
        top_k_mat = torch.zeros((self.top_k, ))# torch.Size([3])
        for i in range(count_seq // self.R_size):
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]# torch.Size([32, 512])
            group_motions = all_genmotions[i * self.R_size:(i + 1) * self.R_size]# torch.Size([32, 512])
            # dist_mat = pairwise_euclidean_distance(group_texts, group_motions)
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()# torch.Size([32, 32])
            self.Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        R_count = count_seq // self.R_size * self.R_size
        metrics["Matching_score"] = self.Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # Compute r-precision with gt
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]# torch.Size([32, 512])
            group_motions = all_gtmotions[i * self.R_size:(i + 1) *
                                          self.R_size]# torch.Size([32, 512])
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()# torch.Size([32, 32])
            # match score
            self.gt_Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        metrics["gt_Matching_score"] = self.gt_Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"gt_R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        assert count_seq > self.diversity_times
        metrics["Diversity"] = calculate_diversity_np(all_genmotions,
                                                      self.diversity_times)
        metrics["gt_Diversity"] = calculate_diversity_np(
            all_gtmotions, self.diversity_times)

        return {**metrics}

    def update(
        self,
        text_embeddings: Tensor,
        recmotion_embeddings: Tensor,
        gtmotion_embeddings: Tensor,
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        # text_embeddings = torch.flatten(text_embeddings, start_dim=1).detach()
        # recmotion_embeddings = torch.flatten(recmotion_embeddings,
        #                                      start_dim=1).detach()
        # gtmotion_embeddings = torch.flatten(gtmotion_embeddings,
        #                                     start_dim=1).detach()

        # store all texts and motions
        self.text_embeddings.append(text_embeddings)
        self.recmotion_embeddings.append(recmotion_embeddings)
        self.gtmotion_embeddings.append(gtmotion_embeddings)


def euclidean_distance_matrix_np(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1,
                keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = (torch.unsqueeze(torch.arange(size),
                              1).to(mat.device).repeat_interleave(size, 1))
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = torch.cat(top_k_list, dim=1)
    return top_k_mat


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_activation_statistics_np(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity_np(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples,
                                     diversity_times,
                                     replace=False)
    second_indices = np.random.choice(num_samples,
                                      diversity_times,
                                      replace=False)
    dist = scipy.linalg.norm(activation[first_indices] -
                             activation[second_indices],
                             axis=1)
    return dist.mean()


class MMMetrics(Metric):
    full_state_update = True

    def __init__(self, mm_num_times=10, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "MultiModality scores"

        self.mm_num_times = mm_num_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = ["MultiModality"]
        self.add_state("MultiModality",
                       default=torch.tensor(0.),
                       dist_reduce_fx="sum")

        # chached batches
        self.add_state("mm_motion_embeddings", default=[], dist_reduce_fx=None)

    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # if in sanity check stage then jump
        if sanity_flag:
            return metrics

        # cat all embeddings
        all_mm_motions = torch.cat(self.mm_motion_embeddings,
                                   axis=0).cpu().numpy()
        metrics['MultiModality'] = calculate_multimodality_np(
            all_mm_motions, self.mm_num_times)

        return {**metrics}

    def update(
        self,
        mm_motion_embeddings: Tensor,
        lengths: List[int],
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # store all mm motion embeddings
        self.mm_motion_embeddings.append(mm_motion_embeddings)
        
        
# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * torch.mm(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = torch.sum(torch.square(matrix1), axis=1,
                   keepdims=True)  # shape (num_test, 1)
    d3 = torch.sum(torch.square(matrix2), axis=1)  # shape (num_train, )
    dists = torch.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (mu1.shape == mu2.shape
            ), "Training and test mean vectors have different lengths"
    assert (sigma1.shape == sigma2.shape
            ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
            print("Warning: Imaginary component {}".format(m))
            # print("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean
    
    
    
    
def calculate_multimodality_np(activation, multimodality_times):
    assert len(activation.shape) == 3
    #假设 activation 的形状为 [16, 10, 256]，则：
    # num_samples = 16
    # num_embeddings_per_sample = 10
    # embedding_dim = 256
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent,
                                   multimodality_times,
                                   replace=False)
    second_dices = np.random.choice(num_per_sent,
                                    multimodality_times,
                                    replace=False)
    dist = scipy.linalg.norm(activation[:, first_dices] -
                             activation[:, second_dices],
                             axis=2)
    return dist.mean()    