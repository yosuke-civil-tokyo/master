# VAE model to generate models' adjacency matrix
import json
import os
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DeepGenerativeModel(nn.Module):
    def __init__(self, z_dim=8):
        super().__init__()
        self.z_dim = z_dim
        # Adjust the network architecture to match z_dim
        self.dense_enc1 = nn.Linear(z_dim * z_dim, 32)  # Adjusted to accept (z_dim*z_dim)
        self.dense_encmean = nn.Linear(32, z_dim)
        self.dense_encvar = nn.Linear(32, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, z_dim * z_dim)  # Adjusted to output (z_dim*z_dim)

    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x.view(-1, self.z_dim*self.z_dim)))
        mean = self.dense_encmean(x)
        var = F.softplus(self.dense_encvar(x))
        return mean, var

    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape)
        return mean + torch.sqrt(var) * epsilon

    def _decoder(self, z):
        x = torch.sigmoid(self.dense_dec1(z))
        return x.view(-1, self.z_dim, self.z_dim)

    def forward(self, x):
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x_recon = self._decoder(z)
        return x_recon, z

    def loss(self, x):
        delta = 1e-7
        mean, var = self._encoder(x)
        KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var + delta) - mean**2 - var, dim=1))
        z = self._sample_z(mean, var)
        y = self._decoder(z)
        reconstruction = torch.mean(torch.sum(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta), dim=[1, 2]))
        lower_bound = [-KL, reconstruction]
        return -sum(lower_bound)
    
    def weighted_loss(self, x, bic_score):
        '''
        x: adjacency matrix (m, m)
        bic_score: bic score of each node (m, )
        '''
        delta = 1e-7
        mean, var = self._encoder(x)
        KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var + delta) - mean**2 - var, dim=1))

        z = self._sample_z(mean, var)
        y = self._decoder(z)

        weighted_reconstruction = torch.mean(torch.sum(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta), dim=[1, 2]) * bic_score)
        lower_bound = [-KL, weighted_reconstruction]
        return -sum(lower_bound)
    
    def train(self, data_loader, optimizer, epochs=1000):
        print("Training...")
        for epoch in range(epochs+1):
            total_loss = 0
            for batch_idx, data in enumerate(data_loader):
                optimizer.zero_grad()
                loss = self.loss(data)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            if epoch % 50 == 0:
                print('Epoch {}: Loss: {:.3f}'.format(epoch+1, total_loss / len(data_loader)))

    # generate data, and check cycle
    # generate till we get num_samples non-cyclic graphs
    def sample(self, num_samples):
        with torch.no_grad():
            samples = []
            while len(samples) < num_samples:
                z = torch.randn(1, self.z_dim)
                sample = self._decoder(z)
                sample = sample.squeeze(0)
                sample = sample.round()
                sample = sample.numpy()
                sample = sample.astype(int)
                sample = torch.tensor(sample)
                visited = torch.zeros(self.z_dim)
                finished = torch.zeros(self.z_dim)
                if not detectCycle(sample, 0, visited, finished):
                    samples.append(sample)
            samples = torch.stack(samples)
        return samples
    

class ConditionalVAE(nn.Module):
    def __init__(self, z_dim=8):
        super().__init__()
        self.z_dim = z_dim

        # Encoder: Adjusted to accept concatenated input of adjacency matrix and BIC scores
        self.dense_enc1 = nn.Linear(z_dim * z_dim + z_dim, 32)  
        self.dense_encmean = nn.Linear(32, z_dim)
        self.dense_encvar = nn.Linear(32, z_dim)

        # Decoder: Takes latent z and concatenated BIC scores
        self.dense_dec1 = nn.Linear(z_dim + z_dim, z_dim * z_dim)

    def _encoder(self, x, bic_score):
        # Reshape x and concatenate with bic_score
        combined_input = torch.cat((x.view(-1, self.z_dim * self.z_dim), bic_score), dim=1)
        h = F.relu(self.dense_enc1(combined_input))
        mean = self.dense_encmean(h)
        var = F.softplus(self.dense_encvar(h))
        return mean, var

    def _decoder(self, z, bic_score):
        # Concatenate z and bic_score
        combined_input = torch.cat((z, bic_score), dim=1)
        x = torch.sigmoid(self.dense_dec1(combined_input))
        return x.view(-1, self.z_dim, self.z_dim)

    def forward(self, x, bic_score):
        mean, var = self._encoder(x, bic_score)
        z = self._sample_z(mean, var)
        x_recon = self._decoder(z, bic_score)
        return x_recon, z

    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape)
        return mean + torch.sqrt(var) * epsilon
    
    def loss(self, x, bic_score):
        delta = 1e-7
        mean, var = self._encoder(x, bic_score)
        KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var + delta) - mean**2 - var, dim=1))
        z = self._sample_z(mean, var)
        y = self._decoder(z, bic_score)
        reconstruction = torch.mean(torch.sum(x * torch.log(y + delta) + (1 - x) * torch.log(1 - y + delta), dim=[1, 2]))
        lower_bound = [-KL, reconstruction]
        return -sum(lower_bound)
    
    def train(self, data_loader, optimizer, epochs=1000):
        print("Training...")
        for epoch in range(epochs+1):
            total_loss = 0
            for batch_idx, (data, bic_score) in enumerate(data_loader):
                optimizer.zero_grad()
                loss = self.loss(data, bic_score)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
            if epoch % 50 == 0:
                print('Epoch {}: Loss: {:.3f}'.format(epoch+1, total_loss / len(data_loader)))
    
    def sample_with_good_BIC(self, num_samples, bic_score):
        with torch.no_grad():
            samples = []
            while len(samples) < num_samples:
                z = torch.randn(1, self.z_dim)
                sample = self._decoder(z, bic_score)
                sample = sample.squeeze(0)
                sample = sample.round()
                sample = sample.numpy()
                sample = sample.astype(int)
                sample = torch.tensor(sample)
                visited = torch.zeros(self.z_dim)
                finished = torch.zeros(self.z_dim)
                if not detectCycle(sample, 0, visited, finished):
                    samples.append(sample)
            samples = torch.stack(samples)
        return samples
    
    def sample_with_random_bic(self, num_samples, bic_score_range):
        with torch.no_grad():
            samples = []
            while len(samples) < num_samples:
                z = torch.randn(1, self.z_dim)
                # Sample a random BIC score within a given range
                random_bic = torch.FloatTensor(1, self.z_dim).uniform_(*bic_score_range)
                sample = self._decoder(z, random_bic)
                # Process the sample as needed
                sample = sample.squeeze(0).round().numpy().astype(int)
                samples.append(sample)
            samples = torch.stack([torch.tensor(s) for s in samples])
        return samples

def createAdjacencyMatrix(config, truthConfig):
    variables = truthConfig["variables"]
    num_vars = len(variables)
    adjacency_matrix = torch.zeros((num_vars, num_vars))

    var_to_idx = {var: idx for idx, var in enumerate(variables.keys())}
    for var, details in variables.items():
        for parent in config["variables"][var]["parents"]:
            adjacency_matrix[var_to_idx[var], var_to_idx[parent]] = 1

    return adjacency_matrix

def createBICScore(config, truthConfig):
    variables = truthConfig["variables"]
    num_vars = len(variables)
    bic_scores = torch.zeros(num_vars)

    var_to_idx = {var: idx for idx, var in enumerate(variables.keys())}
    for var, details in variables.items():
        bic_scores[var_to_idx[var]] = config["variables"][var]["BIC"]

    return bic_scores

def createTensorsFromConfigs(folder, truthConfig, addBIC=False):
    print("Loading configs from: ", folder)
    configs = []
    for modelNum in os.listdir(folder):
        modelPath = os.path.join(folder, modelNum)
        try:
            with open(modelPath, "r") as f:
                config = json.load(f)
        except:
            print("error in loading not json: ", modelPath)
            continue
        configs.append(config)

    adjacency_matrices = [createAdjacencyMatrix(config, truthConfig) for config in configs]
    adjacency_matrices = torch.stack(adjacency_matrices)

    if addBIC:
        bic_scores = torch.stack([createBICScore(config, truthConfig) for config in configs])
        # standardize BIC scores
        bic_scores = (bic_scores - bic_scores.mean(dim=0)) / bic_scores.std(dim=0)
        bic_scores = bic_scores.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        return adjacency_matrices, bic_scores
    
    return adjacency_matrices

# function to check if the adjacency matrix has loops
def detectCycle(adjacency_matrix, node, visited, finished):
    visited[node] = True
    child_nodes = adjacency_matrix[:, node].nonzero().flatten()
    for child_node in child_nodes:
        if finished[child_node]:
            continue
        elif visited[child_node]:
            if not finished[child_node]:
                return True
        elif detectCycle(adjacency_matrix, child_node, visited, finished):
            return True
    finished[node] = True
    return False

"""
if __name__ == "__main__":
    folder = "data/modelData/model2/model2-objorder_optimization/"
    with open("data/modelData/model2/truth/truth.json", "r") as f:
        truthConfig = json.load(f)
    adjacency_matrices = createTensorsFromConfigs(folder, truthConfig=truthConfig)
    data_loader = DataLoader(adjacency_matrices, batch_size=8, shuffle=True)
    model = DeepGenerativeModel(z_dim=33)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train(data_loader=data_loader, optimizer=optimizer, epochs=2000)
    torch.save(model.state_dict(), os.path.join(folder, "model_state.pth"))
"""
if __name__ == "__main__":
    folder = "data/modelData/model2/model2-objorder_optimization/"
    with open("data/modelData/model2/truth/truth.json", "r") as f:
        truthConfig = json.load(f)
    adjacency_matrices, bic_scores = createTensorsFromConfigs(folder, truthConfig=truthConfig, addBIC=True)
    TensorData = TensorDataset(adjacency_matrices, bic_scores)
    data_loader = DataLoader(TensorData, batch_size=8, shuffle=True)
    model = ConditionalVAE(z_dim=33)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train(data_loader=data_loader, optimizer=optimizer, epochs=2000)
    torch.save(model.state_dict(), os.path.join(folder, "model_state.pth"))