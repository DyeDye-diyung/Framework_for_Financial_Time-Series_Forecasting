import numpy as np
import os
from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning import seed_everything

from sklearn.metrics import mean_squared_error

from .model import Discriminator, Generator, Generator_CNNBiLSTM, Generator_CNNBiLSTMAttention
from .model_CNNBiLSTMAttention import CNNBiLSTMAttention
from .model_CNNiTransformerBiLSTM import CNNiTransformerBiLSTM
from .model_CNNTransformerBiLSTM import CNNTransformerBiLSTM
from .model_RevINCNNiTransformerBiLSTM import RevINCNNiTransformerBiLSTM
from .model_RevINCNNiTransformer import RevINCNNiTransformer
from .model_RevINCNNTransformerBiLSTM import RevINCNNTransformerBiLSTM
from .model_SCINet import SCINet
from .model_iTransformer import iTransformer
from .model_Transformer import Transformer
from .model_RevINiTransformerBiLSTM import RevINiTransformerBiLSTM
from .model_RevINTransformerBiLSTM import RevINTransformerBiLSTM
from .model_RevINCNNTransformer import RevINCNNTransformer
from .data import StockDataSet


class GAN(LightningModule):
    def __init__(
            self,
            num_days_for_predict,
            num_days_to_predict,
            target='Apple',
            learning_rate=0.00002,
            momentum=None,
            num_workers=1,
            batch_size=128,
            train_size=0.8,
            validation_size=0.1,
            optimizer=torch.optim.Adam,
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
        self.prepare_data()
        self.G = Generator_CNNBiLSTMAttention(
            input_size=self.raw_dataset.dim,
            output_size=num_days_to_predict
        )
        self.D = Discriminator(
            input_size=num_days_for_predict + num_days_to_predict
        )
        # self.criterion = nn.BCELoss()
        self.criterion = nn.MSELoss()
        self.automatic_optimization = False

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def forward(self, x):
        return self.G(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        optG, optD = self.optimizers()

        fake_data = self.G(x).reshape(-1, self.num_days_to_predict, 1)
        fake_data = torch.cat([y[:, :self.num_days_for_predict, :], fake_data], axis=1)

        real_output = self.D(y)
        fake_output = self.D(fake_data)
        real_labels = torch.ones_like(real_output, device=self.device)
        fake_labels = torch.zeros_like(fake_output, device=self.device)

        lossD = self.criterion(real_output, real_labels) \
                + self.criterion(fake_output, fake_labels)

        optD.zero_grad()
        lossD.backward(retain_graph=True)
        optD.step()

        fake_output = self.D(fake_data)
        lossG = self.criterion(fake_output, real_labels)
        optG.zero_grad()
        lossG.backward()
        optG.step()

        self.log_dict({"lossG": lossG, "lossD": lossD, "lossG+lossD": lossG + lossD})

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.G(x)
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
        y_pred = self.raw_dataset.inverse_transform(self.G(x).cpu()).flatten()
        y_true = self.raw_dataset.inverse_transform(
            y[:, self.num_days_for_predict:self.num_days_for_predict + self.num_days_to_predict, 0].cpu()).flatten()
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        self.log("test_RMSE", rmse, prog_bar=True)
        return rmse

    def configure_optimizers(self):
        if self.momentum is None:
            return (self.optimizer(self.G.parameters(), lr=self.learning_rate),
                    self.optimizer(self.D.parameters(), lr=self.learning_rate))
        else:
            return (self.optimizer(self.G.parameters(), lr=self.learning_rate, momentum=self.momentum),
                    self.optimizer(self.D.parameters(), lr=self.learning_rate, momentum=self.momentum))


checkpoint_callback = ModelCheckpoint(
    monitor='val_RMSE',
    # monitor='val_loss',
    dirpath='./model_checkpoint',
    save_top_k=3,
)

early_stop_callback = EarlyStopping(
    # monitor='val_RMSE',
    monitor='val_loss',
    min_delta=0.00,
    patience=50,
    verbose=False,
    mode='min'
)


def config_parser(
        parser: ArgumentParser = ArgumentParser(),
        targets: list[str] = None,
        optimizers: list[str] = None,
) -> ArgumentParser:
    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand', required=True)

    new_parser = subparsers.add_parser('new', help='Train a new model')
    resume_parser = subparsers.add_parser('resume', help='Resume training a model')

    GAN_parser = new_parser.add_argument_group("GAN", "Arguments for GAN")
    GAN_parser.add_argument("target", type=str, default="Apple", choices=targets, help="Target stock to predict")
    GAN_parser.add_argument("--num-days-for-predict", type=int, default=10, help="Number of days used for prediction")
    GAN_parser.add_argument("--num-days-to-predict", type=int, default=1, help="Number of days to predict")
    GAN_parser.add_argument("--learning-rate", type=float, default=0.00002,
                            help="Learning rate for both generator and discriminator")
    GAN_parser.add_argument("--momentum", type=float, default=None,
                            help="Momentum for both generator and discriminator")
    GAN_parser.add_argument("--optimizer", type=str, default="adam", choices=optimizers,
                            help="Optimizer for both generator and discriminator")
    GAN_parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    GAN_parser.add_argument("--train-size", type=float, default=0.8, help="Train size for dataloader")
    GAN_parser.add_argument("--validation-size", type=float, default=0.1, help="Validation size for dataloader")
    GAN_parser.add_argument("--batch-size", type=int, default=128, help="Batch size for dataloader")
    GAN_parser.add_argument('--model_use', type=str, default='CNN_iTransformer',
                            help='model name, options: [GAN, CNN_BiLSTM_Attention, CNN_iTransformer]')

    resume_parser = resume_parser.add_argument_group("Resume")
    resume_parser.add_argument("checkpoint_path", type=str, default=None)
    resume_parser.add_argument('--model_use', type=str, default='CNN_iTransformer',
                               help='model name, options: [GAN, CNN_BiLSTM_Attention, CNN_iTransformer_BiLSTM, RevIN_CNN_iTransformer_BiLSTM, RevIN_CNN_iTransformer, RevIN_CNN_Transformer_BiLSTM, SCINet, iTransformer, Transformer, CNN_Transformer_BiLSTM, RevIN_iTransformer_BiLSTM, RevIN_Transformer_BiLSTM, RevIN_CNN_Transformer]')

    for p in (GAN_parser, resume_parser):
        trainer_parser = p.add_argument_group("Trainer", "Arguments for Trainer")
        trainer_parser.add_argument("--min-epochs", type=int, default=100, help="Minimum number of epochs")
        trainer_parser.add_argument("--max-epochs", type=int, default=-1,
                                    help="Maximum number of epochs")  # -1 infinite
        trainer_parser.add_argument("--early-stop", type=bool, default=True, help="Whether to use early stopping")
        trainer_parser.add_argument("--early-stop-patience", type=int, default=30, help="Patience for early stopping")

    return parser


if __name__ == "__main__":

    from rich import traceback

    traceback.install()
    import warnings

    warnings.filterwarnings("ignore")

    seed_everything(seed=2025, workers=True)  # set global random seed

    optimizer_map = {
        "adam": torch.optim.Adam,
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
        "asgd": torch.optim.ASGD,
        "lbfgs": torch.optim.LBFGS,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD,
    }

    model_map = {
        'CNN_BiLSTM_Attention': CNNBiLSTMAttention,
        'GAN': GAN,
        'CNN_iTransformer_BiLSTM': CNNiTransformerBiLSTM,
        'RevIN_CNN_iTransformer_BiLSTM': RevINCNNiTransformerBiLSTM,
        'RevIN_CNN_iTransformer': RevINCNNiTransformer,
        'RevIN_CNN_Transformer_BiLSTM': RevINCNNTransformerBiLSTM,
        'SCINet': SCINet,
        'iTransformer': iTransformer,
        'Transformer': Transformer,
        'CNN_Transformer_BiLSTM': CNNTransformerBiLSTM,
        'RevIN_iTransformer_BiLSTM': RevINiTransformerBiLSTM,
        'RevIN_Transformer_BiLSTM': RevINTransformerBiLSTM,
        'RevIN_CNN_Transformer': RevINCNNTransformer,
    }

    parser = config_parser(
        # targets = sorted(name[:-16] for name in os.listdir('data') if name.endswith('(2017-2023).csv')),
        # targets=sorted(name[:-16] for name in os.listdir('data') if name.endswith('(2015-2024).csv')),
        targets=sorted(name[:-4] for name in os.listdir('data') if name.endswith('.csv')),
        optimizers=sorted(optimizer_map.keys())
    )
    args = parser.parse_args()

    model_use = model_map[args.model_use]

    callbacks = [checkpoint_callback]
    if args.early_stop:
        early_stop_callback = EarlyStopping(
            # monitor='val_RMSE',
            monitor='val_loss',
            min_delta=0.00,
            patience=args.early_stop_patience,
            verbose=False,
            mode='min'
        )
        callbacks.append(early_stop_callback)

    trainer_profiler = SimpleProfiler(filename='trainer_profiler_logs')

    trainer = Trainer(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        log_every_n_steps=10,
        callbacks=callbacks,
        profiler=trainer_profiler,  # Record training duration
        deterministic=True,  # Ensure experiment reproducibility
    )

    checkpoint_callback.dirpath = os.path.join(
        trainer.logger.log_dir,
        'checkpoints'
    )

    match args.subcommand:
        case "new":
            args.optimizer = optimizer_map[args.optimizer]
            model = model_use(
                target=args.target,
                num_days_for_predict=args.num_days_for_predict,
                num_days_to_predict=args.num_days_to_predict,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                train_size=args.train_size,
                validation_size=args.validation_size,
                optimizer=args.optimizer,
            )
            trainer.fit(model)
            test_RMSE = trainer.test(model)
            print("test RMSE:", test_RMSE)
        case "resume":
            print("resume from", args.checkpoint_path)
            model = model_use.load_from_checkpoint(args.checkpoint_path)
            trainer.fit(model, ckpt_path=args.checkpoint_path)
            test_RMSE = trainer.test(model)
            print("test RMSE:", test_RMSE)

    trainer.save_checkpoint(os.path.join(checkpoint_callback.dirpath, "last.ckpt"))
