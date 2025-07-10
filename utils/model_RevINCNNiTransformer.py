import numpy as np
import torch
import math
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningModule

from sklearn.metrics import mean_squared_error
from .data import StockDataSet
from .iTransformer.iTransformer import iTransformer
from .iTransformer.revin import RevIN


class RevINCNNiTransformer(LightningModule):
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
            num_variates=self.cnn_out_channels,  # The number of output channels from CNN is the number of variates for iTransformer
            lookback_len=self.num_days_for_predict,
            dim=128,  # model dimensions
            depth=2,  # depth
            heads=8,  # attention heads
            dim_head=64,  # head dimension
            attn_dropout=0.05,  # attention dropout
            pred_length=self.num_days_to_predict,  # can be one prediction, or many
            num_tokens_per_variate=1,  # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
            use_reversible_instance_norm=False  # use reversible instance normalization
        )

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

        # Pass through iTransformer layer
        iTransformer_output_dict = self.iTransformer_layer(cnn_output)  # shape: Dict[pred_length_int, Tensor[batch, pred_length, variate]]
        iTransformer_output = iTransformer_output_dict[self.num_days_to_predict]  # shape: (batch_size, seq_len_to_predict, hidden_size)

        # Take the variable of the last out_var dimensions
        target_output = iTransformer_output[:, :, -self.out_var:]  # shape: (batch_size, seq_len_to_predict, 1)

        # Permute dimension to (batch_size, 1, seq_len_to_predict)
        target_output = target_output.permute(0, 2, 1)
        # RevIN layer, denormalize the target variable
        target_variable_mean = statistics.mean[:, -1:, :]
        target_variable_variance = statistics.variance[:, -1:, :]
        target_variable_gamma = statistics.gamma[-1:, :]
        target_variable_beta = statistics.beta[-1:, :]
        clamped_gamma = torch.sign(target_variable_gamma) * target_variable_gamma.abs().clamp(min=self.eps)
        unscaled_output = (target_output - target_variable_beta) / clamped_gamma
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
        # self.log("val_loss", loss, prog_bar=True)
        # calculate RMSE
        y_true_inverse = self.raw_dataset.inverse_transform(
            y[:, self.num_days_for_predict:self.num_days_for_predict + self.num_days_to_predict, 0].cpu()).flatten()
        y_pred_inverse = self.raw_dataset.inverse_transform(y_pred.cpu()).flatten()
        mse = mean_squared_error(y_true_inverse, y_pred_inverse)
        rmse = np.sqrt(mse)
        # self.log("val_RMSE", rmse, prog_bar=True)
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
