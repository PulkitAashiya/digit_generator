import torch
import torch.nn as nn

class CVAE(nn.Module):
    def _init_(self, latent_dim=20):
        super(CVAE, self)._init_()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(10, 10)

        self.encoder = nn.Sequential(
            nn.Linear(28*28 + 10, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + 10, 400)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        x = torch.flatten(x, start_dim=1)
        c = self.label_emb(labels)
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        c = self.label_emb(labels)
        z = torch.cat([z, c], dim=1)
        h = self.decoder_input(z)
        return self.decoder(h).view(-1, 1, 28, 28)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar