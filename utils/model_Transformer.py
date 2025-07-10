import numpy as np
import torch
import math
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningModule

from sklearn.metrics import mean_squared_error
from .data import StockDataSet
from .Transformer.Transformer import Model as Transformer_layer

class Transformer(LightningModule):
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
            cnn_out_channels=32,
            out_var=1,
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

        self.prepare_data()
        self.input_size = self.raw_dataset.dim  # Input feature dimension

        # Transformer layer
        self.Transformer_layer = Transformer_layer(
            enc_in=self.input_size,  # The input feature dimension is the input size for the Transformer
            dec_in=self.input_size,
            c_out=self.input_size,
            # lookback_len=self.num_days_for_predict,
            d_model=128,  # model dimensions
            d_ff=128,
            e_layers=2,  # encoder layer depth
            d_layers=1,  # decoder layer depth
            n_heads=8,  # attention heads
            # dim_head=64,  # head dimension
            dropout=0.05,  # attention dropout
            pred_len=self.num_days_to_predict,
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

        # Generate data for the Transformer layer
        enc_input = input_seq
        dec_zero = torch.zeros_like(input_seq[:, -self.num_days_for_predict:, :]).float()
        dec_input = torch.cat([input_seq, dec_zero], dim=1).float()
        mark_enc = None
        mark_dec = None
        # Pass through Transformer layer
        Transformer_output = self.Transformer_layer(enc_input, mark_enc, dec_input, mark_dec)  # shape: (batch_size, seq_len_to_predict, out_channels)

        # Take the variable of the last out_var dimensions
        pred = Transformer_output[:, :, -self.out_var]  # shape: (batch_size, seq_len_to_predict)

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
        y_true_inverse = self.raw_dataset.inverse_transform(
            y[:, self.num_days_for_predict:self.num_days_for_predict + self.num_days_to_predict, 0].cpu()).flatten()
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
