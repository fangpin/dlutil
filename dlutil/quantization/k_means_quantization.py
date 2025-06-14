import torch

from fast_pytorch_kmeans import KMeans
from collections import namedtuple

Codebook = namedtuple("Codebook", ["centroids", "labels"])


def k_means_quantize(fp32_tensor: torch.Tensor, bits=4, codebook=None):
    """
    quantize tensor using k-means clustering
    :param fp32_tensor:
    :param bitwidth: [int] quantization bit width, default=4
    :param codebook: [Codebook] (the cluster centroids, the cluster label tensor)
    :return:
        [Codebook = (centroids, labels)]
            centroids: [torch.(cuda.)FloatTensor] the cluster centroids
            labels: [torch.(cuda.)LongTensor] cluster label tensor
    """
    if codebook is None:
        n_clusters = 2**bits
        kmeans = KMeans(n_clusters=n_clusters, mode="euclidean", verbose=0)
        labelsN = kmeans.fit_predict(fp32_tensor.view(-1, 1))

        assert labelsN is not None
        assert kmeans.centroids is not None

        labels = labelsN.to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    quantized_tensor = codebook.centroids[codebook.labels]
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook


class KMeansQuantizer:
    def __init__(self, model: torch.nn.Module, bits=4):
        self.codebook = KMeansQuantizer.quantize(model, bits)

    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    KMeansQuantizer.update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = k_means_quantize(
                    param, codebook=self.codebook[name]
                )

    @torch.no_grad()
    @staticmethod
    def quantize(model: torch.nn.Module, bits: int | dict = 4):
        codebook = dict()
        if isinstance(bits, dict):
            for name, param in model.named_parameters():
                if name in bits:
                    codebook[name] = k_means_quantize(param, bits=bits[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bits)
        return codebook

    @staticmethod
    @torch.no_grad()
    def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
        n_clusters = codebook.centroids.numel()
        t = fp32_tensor.view(-1)
        for k in range(n_clusters):
            codebook.centroids[k] = t[codebook.labels == k].mean()
