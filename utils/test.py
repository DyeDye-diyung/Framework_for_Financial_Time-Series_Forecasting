import numpy as np
import os
from matplotlib.patches import Patch

from utils.train import GAN
from utils.model_CNNBiLSTMAttention import CNNBiLSTMAttention
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
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from argparse import ArgumentParser
from pytorch_lightning import seed_everything

def config_parser(parser: ArgumentParser = ArgumentParser(), targets: list[str] = None,) -> ArgumentParser:
    parser.add_argument('ckpt_path', type=str, help='path to the checkpoint to be tested')
    parser.add_argument('--model_use', type=str, default='CNN_iTransformer',
                        help='model name, options: [GAN, CNN_BiLSTM_Attention, CNN_iTransformer_BiLSTM, RevIN_CNN_iTransformer_BiLSTM, RevIN_CNN_iTransformer, RevIN_CNN_Transformer_BiLSTM, SCINet, iTransformer, Transformer, CNN_Transformer_BiLSTM, RevIN_iTransformer_BiLSTM, RevIN_Transformer_BiLSTM, RevIN_CNN_Transformer]')
    parser.add_argument("--test_target", type=str, default="Apple", choices=targets, help="Target stock to predict")
    return parser

if __name__ == '__main__':
    from rich import print
    from rich import traceback
    traceback.install()
    import warnings
    warnings.filterwarnings("ignore")
    seed_everything(seed=2025, workers=True)  # set global random seed

    parser = config_parser(targets=sorted(name[:-4] for name in os.listdir('data') if name.endswith('.csv')))
    args = parser.parse_args()
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
    model_use = model_map[args.model_use]
    model = model_use.load_from_checkpoint(args.ckpt_path)
    print(model.hparams)
    target = args.test_target

    model.eval()
    model.freeze()
    X, Y = model.dataset[:]
    y_pred = model(X.to(model.device))
    y_true = model.raw_dataset.y_scaler.inverse_transform(Y[:, model.num_days_for_predict].cpu()).flatten()
    y_pred = model.raw_dataset.y_scaler.inverse_transform(y_pred.cpu()).flatten()

    train_end = int(len(y_pred)*model.hparams.train_size)
    validation_end = int(len(y_pred)*(model.hparams.train_size+model.hparams.validation_size))
    y_train_pred, y_validation_pred, y_test_pred = y_pred[:train_end], y_pred[train_end:validation_end], y_pred[validation_end:]
    y_train, y_validation, y_test = y_true[:train_end], y_true[train_end:validation_end], y_true[validation_end:]
    total_MSE = mean_squared_error(y_true, y_pred)
    total_RMSE = np.sqrt(total_MSE)
    total_MAPE = mean_absolute_percentage_error(y_true, y_pred)
    total_R_square = r2_score(y_true, y_pred)
    train_MSE = mean_squared_error(y_train, y_train_pred)
    train_RMSE = np.sqrt(train_MSE)
    train_MAPE = mean_absolute_percentage_error(y_train, y_train_pred)
    train_R_square = r2_score(y_train, y_train_pred)
    validation_MSE = mean_squared_error(y_validation, y_validation_pred)
    validation_RMSE = np.sqrt(validation_MSE)
    validation_MAPE = mean_absolute_percentage_error(y_validation, y_validation_pred)
    validation_R_square = r2_score(y_validation, y_validation_pred)
    test_MSE = mean_squared_error(y_test, y_test_pred)
    test_RMSE = np.sqrt(test_MSE)
    test_MAPE = mean_absolute_percentage_error(y_test, y_test_pred)
    test_R_square = r2_score(y_test, y_test_pred)

    standard_MSE = mean_squared_error(y_true[1:], y_true[:-1])
    standard_RMSE = np.sqrt(standard_MSE)
    train_standard_MSE = mean_squared_error(y_train[1:], y_train[:-1])
    train_standard_RMSE = np.sqrt(train_standard_MSE)
    test_standard_MSE = mean_squared_error(y_test[1:], y_test[:-1])
    test_standard_RMSE = np.sqrt(test_standard_MSE)

    print('target:', target)
    print("total RMSE:", total_RMSE)
    print("train RMSE:", train_RMSE)
    print("validation RMSE:", validation_RMSE)
    print("test RMSE:", test_RMSE)
    print("total MAPE:", total_MAPE)
    print("train MAPE:", train_MAPE)
    print("validation MAPE:", validation_MAPE)
    print("test MAPE:", test_MAPE)
    print("total R square:", total_R_square)
    print("train R square:", train_R_square)
    print("validation R square:", validation_R_square)
    print("test R square:", test_R_square)

    Evaluation_data = [
        [total_RMSE, total_MAPE, total_R_square],
        [train_RMSE, train_MAPE, train_R_square],
        [validation_RMSE, validation_MAPE, validation_R_square],
        [test_RMSE, test_MAPE, test_R_square],
    ]
    Evaluation_df = pd.DataFrame(
        Evaluation_data,
        index=['total', 'train', 'validation', 'test'],
        columns=['RMSE', 'MAPE', 'R^2'],
    )
    # Split path to get version information
    # path_parts = args.ckpt_path.split('/')  # Split the path by slash
    # version_dir = '/'.join(path_parts[:2])  # Take the first two path segments
    version_dir = os.path.normpath(args.ckpt_path).split(os.sep)[:2]  # Split the path and take the first two path segments
    version_dir = os.path.join(*version_dir)  # Compatible with different operating systems
    # Ensure the directory exists
    os.makedirs(version_dir, exist_ok=True)
    # Construct the folder for saving evaluation metrics CSV
    Evaluation_save_path = os.path.join(version_dir, 'test_evaluation')
    # Ensure the directory exists
    os.makedirs(Evaluation_save_path, exist_ok=True)
    # Construct the CSV save path
    Evaluation_save_path_csv = os.path.join(Evaluation_save_path, f'{target}_Evaluation.csv')
    Evaluation_df.to_csv(Evaluation_save_path_csv, index=True)
    print(f'saved evaluation csv to "{Evaluation_save_path_csv}"')

    df = model.raw_dataset.df
    df['Close Pred'] = None
    df.iloc[-len(y_pred):, -1] = y_pred

    # Plotting configuration parameters
    # ===================== Configurable Parameters Area =====================
    # Font configuration
    FONT_CONFIG = {
        'family': 'Times New Roman',  # Global font
        'title_size': 16,  # Title font size
        'label_size': 12,  # Axis label font size
        'legend_size': 10  # Legend font size
    }
    # Grid style
    GRID_STYLE = {
        'alpha': 0.15,  # Alpha (transparency)
        'color': 'lightgray',  # Grid line color
        'linestyle': ':',  # Linestyle (dashed)
        'linewidth': 0.8  # Linewidth
    }
    # Data partition color configuration
    COLOR_SETTINGS = {
        'train': {'color': 'green', 'alpha': 0.1},
        'val': {'color': 'gold', 'alpha': 0.1},
        'test': {'color': 'salmon', 'alpha': 0.1}
    }
    # Legend style
    LEGEND_SETTINGS = {
        'data': {
            'loc': 'upper right',
            'framealpha': 0.9,
            'fontsize': FONT_CONFIG['legend_size']  # Link to font configuration
        },
        'partition': {
            'loc': 'upper left',
            'framealpha': 0.9,
            'fontsize': FONT_CONFIG['legend_size']
        }
    }
    # Split line style
    SPLIT_LINE_STYLE = {
        'color': 'gray',
        'linestyle': '--',
        'alpha': 0.8,
        'linewidth': 1.2
    }
    # Price curve style
    PRICE_STYLE = {
        'true': {'color': '#536897', 'linewidth': 1.5},
        'pred': {'color': '#E17D81', 'linewidth': 1.5}
    }
    # Image save configuration
    SAVE_CONFIG = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'figsize': (16, 8)
    }
    # ===================== End of Configuration =====================
    # Apply global font settings
    plt.rcParams['font.family'] = FONT_CONFIG['family']
    # Plot price curves (explicitly capture legend handles)
    # Apply font configuration when creating the figure
    plt.figure(figsize=SAVE_CONFIG['figsize'])
    ax = df['Close'].plot(**PRICE_STYLE['true'], label='True')
    df['Close Pred'].plot(ax=ax, **PRICE_STYLE['pred'], label='Pred', rot=0)
    # Set title and axis label fonts
    ax.set_title(
        f'{target} Close Price',
        fontsize=FONT_CONFIG['title_size']
    )
    ax.set_xlabel('Date', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('Price', fontsize=FONT_CONFIG['label_size'])
    # Apply grid style
    ax.grid(**GRID_STYLE)
    # Calculate split points
    train_split = df.index[int(len(df) * model.hparams.train_size)]
    validation_split = df.index[int(len(df) * (model.hparams.train_size + model.hparams.validation_size))]
    # Create figure
    plt.figure(figsize=SAVE_CONFIG['figsize'])
    ax = df['Close'].plot(**PRICE_STYLE['true'], label='True')
    df['Close Pred'].plot(ax=ax, **PRICE_STYLE['pred'], label = 'Pred', rot = 0)
    # Plot data partition backgrounds
    plt.axvspan(df.index[0], train_split, **COLOR_SETTINGS['train'])
    plt.axvspan(train_split, validation_split, **COLOR_SETTINGS['val'])
    plt.axvspan(validation_split, df.index[-1], **COLOR_SETTINGS['test'])
    # Plot split lines
    # plt.axvline(train_split, **SPLIT_LINE_STYLE)
    # plt.axvline(validation_split, **SPLIT_LINE_STYLE)
    # Create two legends
    # Data curve legend
    data_legend = plt.legend(**LEGEND_SETTINGS['data'])
    # Partition legend (manually create handles)
    partition_handles = [
        Patch(facecolor=v['color'], alpha=v['alpha'], label=f'{k.capitalize()} Set')
        for k, v in COLOR_SETTINGS.items()
    ]
    plt.legend(handles=partition_handles, **LEGEND_SETTINGS['partition'])
    ax.add_artist(data_legend)  # Ensure both legends are displayed simultaneously

    # Construct the folder for saving images
    plt_save_path = os.path.join(version_dir, 'test_images')
    # Ensure the directory exists
    os.makedirs(plt_save_path, exist_ok=True)
    # Construct the PNG image save path
    plt_save_path_png = os.path.join(plt_save_path, f'{target} Close Price Prediction.png')
    # Save the PNG image
    plt.savefig(
        plt_save_path_png,
        dpi=SAVE_CONFIG['dpi'],
        bbox_inches=SAVE_CONFIG['bbox_inches']
    )
    print(f'saved prediction plot to png "{plt_save_path_png}"')
    # Construct the SVG vector image save path
    plt_save_path_svg = os.path.join(plt_save_path, f'{target} Close Price Prediction.svg')
    # Save the SVG image
    plt.savefig(
        plt_save_path_svg,
        dpi=SAVE_CONFIG['dpi'],
        bbox_inches=SAVE_CONFIG['bbox_inches']
    )
    print(f'saved prediction plot to svg "{plt_save_path_svg}"')

    # Construct the folder for saving predictions
    pred_save_path = os.path.join(version_dir, 'test_prediction')
    # Ensure the directory exists
    os.makedirs(pred_save_path, exist_ok=True)
    # Construct the prediction save path
    pred_save_path_csv = os.path.join(pred_save_path, f'{target}_pred.csv')
    # Build the prediction dataframe and save to CSV
    y_pred_df = pd.DataFrame(y_pred, index=df.index[-len(y_pred):], columns=["y_pred"])  # y_pred: type: np.ndarray, 1d
    y_pred_df.to_csv(pred_save_path_csv, index=True)
    # print(f'shape of y_pred: {y_pred.shape}')
    print(f'{target} Close Price Prediction saved to "{pred_save_path_csv}"')

    # Construct the training set prediction save path
    train_pred_save_path_csv = os.path.join(pred_save_path, f'{target}_train_pred.csv')
    # Build the training set prediction dataframe and save to CSV
    y_train_index_end = len(y_validation_pred) + len(y_test_pred)
    y_train_pred_df = pd.DataFrame(y_train_pred, index=df.index[-len(y_pred):-y_train_index_end], columns=["y_train_pred"])  # y_pred: type: np.ndarray, 1d
    y_train_pred_df.to_csv(train_pred_save_path_csv, index=True)
    print(f'{target} Close Price Train Prediction saved to "{train_pred_save_path_csv}"')

    # Construct the validation set prediction save path
    validation_pred_save_path_csv = os.path.join(pred_save_path, f'{target}_validation_pred.csv')
    # Build the validation set prediction dataframe and save to CSV
    y_validation_pred_df = pd.DataFrame(y_validation_pred, index=df.index[-y_train_index_end:-len(y_test_pred)], columns=["y_validation_pred"])  # y_pred: type: np.ndarray, 1d
    y_validation_pred_df.to_csv(validation_pred_save_path_csv, index=True)
    print(f'{target} Close Price Validation Prediction saved to "{validation_pred_save_path_csv}"')

    # Construct the test set prediction save path
    test_pred_save_path_csv = os.path.join(pred_save_path, f'{target}_test_pred.csv')
    # Build the test set prediction dataframe and save to CSV
    y_test_pred_df = pd.DataFrame(y_test_pred, index=df.index[-len(y_test_pred):], columns=["y_test_pred"])  # y_pred: type: np.ndarray, 1d
    y_test_pred_df.to_csv(test_pred_save_path_csv, index=True)
    print(f'{target} Close Price Test Prediction saved to "{test_pred_save_path_csv}"')
