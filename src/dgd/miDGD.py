from src.dgd.latent import GaussianMixture
import torch.nn as nn


class miDGD(nn.Module):
    def __init__(self, decoder, n_mix, rep_dim, gmm_spec={}):
        super(miDGD, self).__init__()
        self.decoder = decoder
        self.rep_dim = rep_dim      # Dimension of representation

        self.gmm = GaussianMixture(n_mix, rep_dim, **gmm_spec)
        self.train_rep_mrna = None
        self.train_rep_mirna = None
        self.val_rep_mrna = None
        self.val_rep_mirna = None
        self.test_rep_mrna = None
        self.test_rep_mirna = None

    def forward(self, z):
        return self.decoder(z)

    def loss(self, z, y, target, scale, gmm_loss=True, reduction="sum"):
        self.dec_loss = self.decoder.loss(
            y, target, scale, reduction=reduction)
        if gmm_loss:
            self.gmm_loss = self.gmm(z)
            if reduction == "mean":
                self.gmm_loss = self.gmm_loss.mean()
            elif reduction == "sum":
                self.gmm_loss = self.gmm_loss.sum()
            return self.dec_loss, self.gmm_loss
        else:
            return self.dec_loss, None

    def forward_and_loss(self, z, target, scale, gmm_loss=True, reduction="sum"):
        y = self.decoder(z)
        return self.loss(z, y, target, scale, gmm_loss, reduction)

    def get_representations(self, type="train", data_type="mrna"):
        if type == "train":
            return self.train_rep_mrna.z.detach().cpu().numpy() if data_type == "mrna" else self.train_rep_mirna.z.detach().cpu().numpy()
        elif type == "val":
            return self.val_rep_mrna.z.detach().cpu().numpy() if data_type == "mrna" else self.val_rep_mirna.z.detach().cpu().numpy()
        elif type == "test":
            return self.test_rep_mrna.z.detach().cpu().numpy() if data_type == "mrna" else self.test_rep_mirna.z.detach().cpu().numpy()

    def get_gmm_means(self):
        return self.gmm.mean.detach().cpu().numpy()

    def get_latent_space_values(self, rep_type="train", n_samples=1000, data_type="mrna"):
        # Adjusted to handle both mRNA and miRNA data
        rep = self.get_representations(rep_type, data_type)
        gmm_means = self.get_gmm_means()
        gmm_samples = self.gmm.sample(n_samples).detach().cpu().numpy()

        return rep, gmm_means, gmm_samples
