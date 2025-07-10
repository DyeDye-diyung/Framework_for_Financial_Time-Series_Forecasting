import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        modules = []
        modules.append(
            SelfAttention(config[0])
        )
        for i in range(1, len(config)):
            modules.extend((
                nn.Linear(config[i - 1], config[i]),
                nn.ReLU()
            ))

        self.config = config
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], config[-1])
        self.fc_var = nn.Linear(config[-1], config[-1])
        modules = []
        for i in range(len(config) - 1, 0, -1):
            modules.extend((
                nn.Linear(config[i], config[i - 1]),
                nn.ReLU()
            ))
        modules[-1] = nn.Sigmoid()

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logVar = self.fc_var(result)
        return mu, logVar

    def decode(self, x):
        result = self.decoder(x)
        return result

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        output = self.decode(z)
        return output, z, mu, logVar

    def loss_function(self, recon_x, x, mu, logVar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        return BCE + KLD


class SelfAttention(pl.LightningModule):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, V)
        return attended_values


class Discriminator(pl.LightningModule):
    def __init__(self, input_size, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=1, padding='same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size=kernel_size, stride=1, padding='same')
        self.linear1 = nn.Linear(128, 220)
        self.batch1 = nn.BatchNorm1d(220)
        self.linear2 = nn.Linear(220, 220)
        self.batch2 = nn.BatchNorm1d(220)
        self.linear3 = nn.Linear(220, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        flatten_x = conv3.squeeze()
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out_3 = self.linear3(out_2)
        out = self.sigmoid(out_3)
        return out


class Generator(pl.LightningModule):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.gru_1 = nn.GRU(input_size, 1024, batch_first=True)
        self.gru_2 = nn.GRU(1024, 512, batch_first=True)
        self.gru_3 = nn.GRU(512, 256, batch_first=True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, noise=None):
        h0 = torch.zeros((1, x.size(0), 1024), device=self.device)
        out_1, _ = self.gru_1(x, h0)
        out_1 = self.dropout(out_1)
        h1 = torch.zeros((1, x.size(0), 512), device=self.device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros((1, x.size(0), 256), device=self.device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out


class Generator_CNNBiLSTM(pl.LightningModule):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 48
        self.num_layers = 2
        self.output_size = output_size
        self.cnn_out_channels = 64
        self.out_var = 1
        self.num_directions = 2

        # Add CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.cnn_out_channels,
                            # The number of output channels from CNN is the input size for LSTM
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)  # Bidirectional LSTM
        # Fully connected layer
        self.linear = nn.Linear(self.hidden_size * self.num_directions, self.out_var)

    def forward(self, input_seq: torch.Tensor, noise=None):
        # input shape: (batch_size, seq_len, input_size)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        # Permute input from (batch_size, seq_len, input_size) to (batch_size, input_size, seq_len)
        # because Conv1d expects input shape (batch_size, channels, seq_len)
        input_seq = input_seq.permute(0, 2, 1)

        # Pass through CNN layer
        cnn_output = self.cnn(input_seq)  # shape: (batch_size, out_channels, seq_len_after_cnn)

        # Permute CNN output back to (batch_size, seq_len_after_cnn, out_channels)
        cnn_output = cnn_output.permute(0, 2, 1)

        # Initial hidden and cell states for LSTM
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Pass through LSTM layer
        lstm_output, hidden = self.lstm(cnn_output, (h_0, c_0))  # shape: (batch_size, seq_len_after_cnn, hidden_size)

        # Pass through fully connected layer
        pred = self.linear(lstm_output[:, -1, :])  # shape: (batch_size, seq_len_after_cnn, output_var)

        return pred


# Implementation of the Attention part for CNN_BiLSTM_Attention
class CausalScaledDotProductAttention(nn.Module):
    def __init__(self, input_dim, output_dim, mask=True):
        """
        input_dim: Feature dimension of LSTM output (hidden_size * num_directions)
        output_dim: Dimension of the prediction target (output_var)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask = mask

        # Projection matrices for generating Q/K/V (single head)
        self.Wq = nn.Linear(input_dim, input_dim)  # Projection for Q
        self.Wk = nn.Linear(input_dim, input_dim)  # Projection for K
        self.Wv = nn.Linear(input_dim, output_dim)  # Projection for V, maps directly to the output dimension

        # Scaling factor
        self.scale = 1.0 / math.sqrt(input_dim)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        Returns: (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Generate Q/K/V
        Q = self.Wq(x)  # [B, L, input_dim]
        K = self.Wk(x)  # [B, L, input_dim]
        V = self.Wv(x)  # [B, L, output_dim] (V is directly projected to the target dimension)

        # Calculate scaled dot-product attention scores [B, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Causal mask (to prevent attending to future information)
        if self.mask:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(x.device)
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax normalization
        attn_weights = torch.softmax(scores, dim=-1)  # [B, L, L]

        # Weighted sum (using the projected V)
        # attn_output = torch.matmul(attn_weights, V)  # [B, L, output_dim]
        output = torch.matmul(attn_weights, V)  # [B, L, output_dim]

        return output


class Generator_CNNBiLSTMAttention(pl.LightningModule):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 48
        self.num_layers = 2
        self.output_size = output_size
        self.cnn_out_channels = 64
        self.out_var = 1
        self.num_directions = 2

        # Add CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.cnn_out_channels,
                            # The number of output channels from CNN is the input size for LSTM
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)  # Bidirectional LSTM
        # Attention layer
        self.attention = CausalScaledDotProductAttention(
            input_dim=self.hidden_size * self.num_directions,  # Bidirectional LSTM output dimension
            output_dim=self.hidden_size * self.num_directions,  # Dimension remains unchanged
            mask=True  # Enable causal mask
        )
        # Fully connected layer
        self.linear = nn.Linear(self.hidden_size * self.num_directions, self.out_var)

    def forward(self, input_seq: torch.Tensor, noise=None):
        # input shape: (batch_size, seq_len, input_size)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        # Permute input from (batch_size, seq_len, input_size) to (batch_size, input_size, seq_len)
        # because Conv1d expects input shape (batch_size, channels, seq_len)
        input_seq = input_seq.permute(0, 2, 1)

        # Pass through CNN layer
        cnn_output = self.cnn(input_seq)  # shape: (batch_size, out_channels, seq_len_after_cnn)

        # Permute CNN output back to (batch_size, seq_len_after_cnn, out_channels)
        cnn_output = cnn_output.permute(0, 2, 1)

        # Initial hidden and cell states for LSTM
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Pass through LSTM layer
        lstm_output, hidden = self.lstm(cnn_output, (h_0, c_0))  # shape: (batch_size, seq_len_after_cnn, hidden_size)

        # Pass through attention layer
        attention_output = self.attention(lstm_output)  # shape: (batch_size, seq_len_after_cnn, output_var)

        # Pass through fully connected layer
        pred = self.linear(attention_output[:, -1, :])  # shape: (batch_size, seq_len_after_cnn, output_var)

        return pred
