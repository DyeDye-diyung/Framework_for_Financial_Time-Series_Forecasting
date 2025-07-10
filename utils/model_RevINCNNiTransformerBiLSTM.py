import numpy as np
import torch
import math
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningModule

from sklearn.metrics import mean_squared_error
from .data import StockDataSet
from .iTransformer.iTransformer_new import Model as iTransformer
from .iTransformer.revin import RevIN


class RevINCNNiTransformerBiLSTM(LightningModule):
    def __init__(
            self,
            num_days_for_predict: int,
            num_days_to_predict: int,
            target='Apple',
            learning_rate=0.00002,
            momentum=None,
            num_workers=1,
            batch_size=128,
            train_size=0.8,
            validation_size=0.1,
            optimizer=torch.optim.Adam,
            hidden_size=48,
            num_layers=2,
            cnn_out_channels=16,
            out_var=1,
            eps=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_days_for_predict = num_days_for_predict
        self.num_days_to_predict = num_days_to_predict
        self.target = target
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.optimizer = optimizer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn_out_channels = cnn_out_channels
        self.out_var = out_var
        self.num_directions = 2
        self.eps = eps

        self.prepare_data()
        self.input_size = self.raw_dataset.dim  # Input feature dimension

        # Add RevIN layer
        self.rev_in = RevIN(num_variates=self.input_size, eps=self.eps)
        # Add CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.cnn_out_channels, kernel_size=3, padding=1),
        )
        # iTransformer layer
        self.iTransformer_layer = iTransformer(
            seq_len=self.num_days_for_predict,
            pred_len=self.num_days_for_predict,
        )
        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.cnn_out_channels,  # The number of output channels from iTransformer is the input size for LSTM
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)  # Bidirectional LSTM
        # Fully connected layer
        self.linear = nn.Linear(self.hidden_size * self.num_layers, self.num_days_to_predict)  # Map hidden dimension to the prediction time length

        self.criterion = nn.MSELoss()

    def prepare_data(self) -> None:
        self.raw_dataset = StockDataSet.from_preprocessed(target=self.target)
        X, Y = self.raw_dataset[:]
        X = torch.from_numpy(np.array([X[i:i + self.num_days_for_predict] for i in
                                       range(len(X) - self.num_days_for_predict - self.num_days_to_predict + 1)]))
        Y = torch.from_numpy(np.array([Y[i:i + self.num_days_for_predict + self.num_days_to_predict] for i in
                                       range(len(Y) - self.num_days_for_predict - self.num_days_to_predict + 1)]))
        self.dataset = TensorDataset(X, Y)

    def setup(self, stage: str) -> None:
        train_end = int(len(self.dataset) * self.train_size)
        validation_end = int(len(self.dataset) * (self.train_size + self.validation_size))
        match stage:
            case "fit":
                self.train_dataset = TensorDataset(*self.dataset[:train_end])
                self.val_dataset = TensorDataset(*self.dataset[train_end:validation_end])
            case "test":
                self.test_dataset = TensorDataset(*self.dataset[validation_end:])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def forward(self, input_seq: torch.Tensor):
        # input shape: (batch_size, seq_len, input_size)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        # Permute input from (batch_size, seq_len, input_size) to (batch_size, input_size, seq_len)
        # because Conv1d's input shape is (batch_size, channels, seq_len), and rev_in's input shape is (batch_size, num_variate, seq_len)
        input_seq = input_seq.permute(0, 2, 1)

        # Pass through RevIN layer
        normalized_input_seq, reverse_fn, statistics = self.rev_in(input_seq, return_statistics=True)

        # Pass through CNN layer
        cnn_output = self.cnn(normalized_input_seq)  # shape: (batch_size, out_channels, seq_len_after_cnn)

        # Permute CNN output back to (batch_size, seq_len_after_cnn, out_channels)
        cnn_output = cnn_output.permute(0, 2, 1)

        # Generate data for the iTransformer layer
        enc_input = cnn_output
        dec_zero = torch.zeros_like(cnn_output[:, -self.num_days_for_predict:, :]).float()
        dec_input = torch.cat([cnn_output, dec_zero], dim=1).float()
        mark_enc = None
        mark_dec = None
        # Pass through Transformer layer
        iTransformer_output = self.iTransformer_layer(enc_input, mark_enc, dec_input, mark_dec)  # shape: (batch_size, seq_len_for_predict, out_channels)

        # Initial hidden and cell states for LSTM
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Pass through LSTM layer
        lstm_output, hidden = self.lstm(iTransformer_output, (h_0, c_0))  # shape: (batch_size, seq_len_for_predict, hidden_size)
        # Get the last time step output of the forward LSTM
        lstm_forward_last_output = lstm_output[:, -1:, :self.hidden_size]  # shape: (batch_size, 1, hidden_size)
        # Get the last time step output of the backward LSTM
        lstm_backward_last_output = lstm_output[:, 0:1, -self.hidden_size:]  # shape: (batch_size, 1, hidden_size)
        # Concatenate
        lstm_final_output = torch.cat([lstm_forward_last_output, lstm_backward_last_output], dim=2)  # shape: (batch_size, 1, hidden_size * 2)

        # Pass through fully connected layer
        linear_output = self.linear(lstm_final_output[:, -1:, :])  # shape: (batch_size, 1, seq_len_to_predict)

        # RevIN layer, denormalize the target variable
        target_variable_mean = statistics.mean[:, -1:, :]
        target_variable_variance = statistics.variance[:, -1:, :]
        target_variable_gamma = statistics.gamma[-1:, :]
        target_variable_beta = statistics.beta[-1:, :]
        clamped_gamma = torch.sign(target_variable_gamma) * target_variable_gamma.abs().clamp(min=self.eps)
        unscaled_output = (linear_output - target_variable_beta) / clamped_gamma
        reverse_output = unscaled_output * target_variable_variance.sqrt() + target_variable_mean
        # shape: (batch_size, 1, seq_len_to_predict)

        pred = reverse_output[:, -1, :]  # shape: (batch_size, seq_len_to_predict)

        return pred

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x).reshape(-1, self.num_days_to_predict, 1)  # Reshape to 3D (batch_size, seq_len_to_predict, out_var)
        y_in_and_pred = torch.cat([y[:, :self.num_days_for_predict, :], y_pred], axis=1)  # Concatenate with the original sequence to get (batch_size, seq_len_for_predict + seq_len_to_predict, out_var)
        loss = self.criterion(y_in_and_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        y_pred_reshape = y_pred.reshape(-1, self.num_days_to_predict, 1)  # Reshape to 3D (batch_size, seq_len_to_predict, out_var)
        y_in_and_pred = torch.cat([y[:, :self.num_days_for_predict, :], y_pred_reshape], axis=1)  # Concatenate with the original sequence to get (batch_size, seq_len_for_predict + seq_len_to_predict, out_var)
        loss = self.criterion(y_in_and_pred, y)
        # calculate RMSE
        y_true_inverse = self.raw_dataset.inverse_transform(y[:, self.num_days_for_predict:self.num_days_for_predict + self.num_days_to_predict, 0].cpu()).flatten()
        y_pred_inverse = self.raw_dataset.inverse_transform(y_pred.cpu()).flatten()
        mse = mean_squared_error(y_true_inverse, y_pred_inverse)
        rmse = np.sqrt(mse)
        self.log_dict({'val_loss': loss, 'val_RMSE': rmse}, prog_bar=True)
        return loss


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.raw_dataset.inverse_transform(self(x).cpu()).flatten()

        y_true = self.raw_dataset.inverse_transform(
            y[:, self.num_days_for_predict:self.num_days_for_predict + self.num_days_to_predict, 0].cpu()).flatten()
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        self.log("test_RMSE", rmse, prog_bar=True)
        return rmse

    def configure_optimizers(self):
        if self.momentum is None:
            return self.optimizer(self.parameters(), lr=self.learning_rate)
        else:
            return self.optimizer(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
