# VAE model to generate models' adjacency matrix
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepGenerativeModel(nn.Module):
    def __init__(self, z_dim=8):
        super(DeepGenerativeModel, self).__init__()
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
    

def createAdjacencyMatrix(config, truthConfig):
    variables = truthConfig["variables"]
    num_vars = len(variables)
    adjacency_matrix = torch.zeros((num_vars, num_vars))

    var_to_idx = {var: idx for idx, var in enumerate(variables.keys())}
    for var, details in variables.items():
        for parent in details["parents"]:
            adjacency_matrix[var_to_idx[var], var_to_idx[parent]] = 1

    return adjacency_matrix

def createTensorsFromConfigs(folder, truthConfig):
    print("Loading configs from: ", folder)
    configs = []
    for modelNum in os.listdir(folder):
        modelPath = os.path.join(folder, modelNum)
        try:
            with open(modelPath, "r") as f:
                config = json.load(f)
        except:
            print("error in loading: ", modelPath)
            continue
        configs.append(config)

    adjacency_matrices = [createAdjacencyMatrix(config, truthConfig) for config in configs]
    adjacency_matrices = torch.stack(adjacency_matrices)
    return adjacency_matrices

if __name__ == "__main__":
    folder = "data/modelData/model2/model2-objstructure_optimization/"
    with open("data/modelData/model2/truth/truth.json", "r") as f:
        truthConfig = json.load(f)
    adjacency_matrices = createTensorsFromConfigs(folder, truthConfig=truthConfig)
    data_loader = torch.utils.data.DataLoader(adjacency_matrices, batch_size=4, shuffle=True)
    model = DeepGenerativeModel(z_dim=33)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train(data_loader=data_loader, optimizer=optimizer, epochs=1000)
    torch.save(model.state_dict(), os.path.join(folder, "model.pt"))