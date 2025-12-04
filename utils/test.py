import numpy as np
import os
from matplotlib.patches import Patch

from utils.data import StockDataSet

from utils.train import GAN
from utils.model_CNNBiLSTMAttention import CNNBiLSTMAttention
from .model_CNNiTransformerBiLSTM import CNNiTransformerBiLSTM
from .model_CNNTransformerBiLSTM import CNNTransformerBiLSTM
from .model_RevINCNNiTransformerBiLSTM import RevINCNNiTransformerBiLSTM
from .model_RevINCNNiTransformer import RevINCNNiTransformer
from .model_RevINCNNTransformerBiLSTM import RevINCNNTransformerBiLSTM
from .model_SCINet import SCINet
from .model_xLSTM import xLSTM
from .model_Mamba import Mamba
from .model_iTransformer import iTransformer
from .model_Transformer import Transformer
from .model_RevINiTransformerBiLSTM import RevINiTransformerBiLSTM
from .model_RevINTransformerBiLSTM import RevINTransformerBiLSTM
from .model_RevINCNNTransformer import RevINCNNTransformer
from .model_RevINCNNiTransformerBiLSTM_PinballLoss import RevINCNNiTransformerBiLSTM as RevINCNNiTransformerBiLSTM_PinballLoss
from .model_iTransformer_PinballLoss import iTransformer as iTransformer_PinballLoss
from .model_RevINCNNTransformerBiLSTM_PinballLoss import RevINCNNTransformerBiLSTM as RevINCNNTransformerBiLSTM_PinballLoss
from .model_Transformer_PinballLoss import Transformer as Transformer_PinballLoss
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
    parser.add_argument('--model_use', type=str, default='CNN_iTransformer', help='model name, options: [GAN, CNN_BiLSTM_Attention, CNN_iTransformer_BiLSTM, RevIN_CNN_iTransformer_BiLSTM, RevIN_CNN_iTransformer, RevIN_CNN_Transformer_BiLSTM, SCINet, xLSTM, Mamba, iTransformer, Transformer, CNN_Transformer_BiLSTM, RevIN_iTransformer_BiLSTM, RevIN_Transformer_BiLSTM, RevIN_CNN_Transformer, RevIN_CNN_iTransformer_BiLSTM_PinballLoss, iTransformer_PinballLoss, RevIN_CNN_Transformer_BiLSTM_PinballLoss, Transformer_PinballLoss]')
    parser.add_argument("--test_target", type=str, default="Apple", choices=targets, help="Target stock to predict")
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'day-ahead'], help='Set the script to run in test or day-ahead prediction mode.')
    parser.add_argument('--data-csv-path', type=str, default='data/processed_dataset.csv', help='Path to external data CSV for day-ahead mode.')
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
        'xLSTM': xLSTM,
        'Mamba': Mamba,
        'iTransformer': iTransformer,
        'Transformer': Transformer,
        'CNN_Transformer_BiLSTM': CNNTransformerBiLSTM,
        'RevIN_iTransformer_BiLSTM': RevINiTransformerBiLSTM,
        'RevIN_Transformer_BiLSTM': RevINTransformerBiLSTM,
        'RevIN_CNN_Transformer': RevINCNNTransformer,
        'RevIN_CNN_iTransformer_BiLSTM_PinballLoss': RevINCNNiTransformerBiLSTM_PinballLoss,
        'iTransformer_PinballLoss': iTransformer_PinballLoss,
        'RevIN_CNN_Transformer_BiLSTM_PinballLoss': RevINCNNTransformerBiLSTM_PinballLoss,
        'Transformer_PinballLoss': Transformer_PinballLoss,        
    }
    model_use = model_map[args.model_use]
    model = model_use.load_from_checkpoint(args.ckpt_path)
    print(model.hparams)
    target = args.test_target

    model.eval()
    model.freeze()
    
    if args.mode == 'day-ahead':
        print(f"Day-ahead mode: Loading data from '{args.data_csv_path}'")
        # 1. Load external CSV data
        df_day_ahead = pd.read_csv(args.data_csv_path, index_col=0, parse_dates=True)
        
        # 2. Create a new raw_dataset instance using the StockDataSet
        #    Key: Pass in the already fitted scaler extracted from the loaded model
        day_ahead_raw_dataset = StockDataSet(
            df=df_day_ahead.copy(),
            target=target,
            is_scaled=True,
            pre_fitted_x_scaler=model.raw_dataset.x_scaler,
            pre_fitted_y_scaler=model.raw_dataset.y_scaler
        )

        # 3. Accurately reproduce the sliding window creation logic in the training script
        input_len = model.hparams.num_days_for_predict
        output_len = model.hparams.num_days_to_predict
        
        x_scaled, y_scaled = day_ahead_raw_dataset.X, day_ahead_raw_dataset.Y
        
        # Create all possible Windows using list derivations
        X_list = [x_scaled[i : i + input_len] for i in range(len(x_scaled) - input_len - output_len + 1)]
        Y_list = [y_scaled[i : i + input_len + output_len] for i in range(len(y_scaled) - input_len - output_len + 1)]

        # Convert the list to a Numpy Array and then to Torch Tensors
        X = torch.from_numpy(np.array(X_list)).float()
        Y = torch.from_numpy(np.array(Y_list)).float()
        
        # The DataFrame used for subsequent drawing should be this new DataFrame
        df = df_day_ahead
        
        # Predict
        y_pred = model(X.to(model.device))
    else: # standard test mode
        X, Y = model.dataset[:]
        # Plot using the original DataFrame inside the model
        df = model.raw_dataset.df
        y_pred = model(X.to(model.device))
    
    # Unified reverse scaling operation
    # From the long Y window, slice out the true future value portion and perform inverse scaling
    true_future_values_scaled = Y[:, model.hparams.num_days_for_predict : model.hparams.num_days_for_predict + model.hparams.num_days_to_predict]
    y_true = model.raw_dataset.y_scaler.inverse_transform(true_future_values_scaled.cpu().reshape(-1, 1)).flatten()

    y_pred = model.raw_dataset.y_scaler.inverse_transform(y_pred.cpu()).flatten()

    def rmsre(y_true, y_pred):
        """ Root Mean Squared Relative Error """
        epsilon = 1e-10
        return np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + epsilon))))


    def rmspe(y_true, y_pred):
        """ Root Mean Squared Percentage Error """
        return rmsre(y_true, y_pred) * 100


    def mare(y_true, y_pred):
        """ Mean Absolute Relative Error """
        epsilon = 1e-10
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))

    if args.mode == 'day-ahead':
        # In the day-ahead mode, only the overall indicators are calculated
        total_MSE = mean_squared_error(y_true, y_pred)
        total_RMSE = np.sqrt(total_MSE)
        total_MAPE = mean_absolute_percentage_error(y_true, y_pred) * 100
        total_R_square = r2_score(y_true, y_pred)
        total_RMSRE = rmsre(y_true, y_pred)
        total_RMSPE = rmspe(y_true, y_pred)
        total_MARE = mare(y_true, y_pred)

        print('target:', target)
        print("day-ahead RMSE:", total_RMSE)
        print("day-ahead MAPE:", total_MAPE)
        print("day-ahead R square:", total_R_square)
        print("day-ahead RMSRE:", total_RMSRE)
        print("day-ahead RMSPE:", total_RMSPE)
        print("day-ahead MARE:", total_MARE)

        Evaluation_data = [[total_RMSE, total_MAPE, total_R_square, total_RMSRE, total_RMSPE, total_MARE]]
        Evaluation_df = pd.DataFrame(
            Evaluation_data,
            index=['day-ahead'],
            columns=['RMSE', 'MAPE', 'R^2', 'RMSRE', 'RMSPE', 'MARE'],
        )
    else: # standard test mode
        train_end = int(len(y_pred)*model.hparams.train_size)
        validation_end = int(len(y_pred)*(model.hparams.train_size+model.hparams.validation_size))
        y_train_pred, y_validation_pred, y_test_pred = y_pred[:train_end], y_pred[train_end:validation_end], y_pred[validation_end:]
        y_train, y_validation, y_test = y_true[:train_end], y_true[train_end:validation_end], y_true[validation_end:]
        total_MSE = mean_squared_error(y_true, y_pred)
        total_RMSE = np.sqrt(total_MSE)
        total_MAPE = mean_absolute_percentage_error(y_true, y_pred) * 100
        total_R_square = r2_score(y_true, y_pred)
        total_RMSRE = rmsre(y_true, y_pred)
        total_RMSPE = rmspe(y_true, y_pred)
        total_MARE = mare(y_true, y_pred)
        train_MSE = mean_squared_error(y_train, y_train_pred)
        train_RMSE = np.sqrt(train_MSE)
        train_MAPE = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        train_R_square = r2_score(y_train, y_train_pred)
        train_RMSRE = rmsre(y_train, y_train_pred)
        train_RMSPE = rmspe(y_train, y_train_pred)
        train_MARE = mare(y_train, y_train_pred)
        validation_MSE = mean_squared_error(y_validation, y_validation_pred)
        validation_RMSE = np.sqrt(validation_MSE)
        validation_MAPE = mean_absolute_percentage_error(y_validation, y_validation_pred) * 100
        validation_R_square = r2_score(y_validation, y_validation_pred)
        validation_RMSRE = rmsre(y_validation, y_validation_pred)
        validation_RMSPE = rmspe(y_validation, y_validation_pred)
        validation_MARE = mare(y_validation, y_validation_pred)
        test_MSE = mean_squared_error(y_test, y_test_pred)
        test_RMSE = np.sqrt(test_MSE)
        test_MAPE = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        test_R_square = r2_score(y_test, y_test_pred)
        test_RMSRE = rmsre(y_test, y_test_pred)
        test_RMSPE = rmspe(y_test, y_test_pred)
        test_MARE = mare(y_test, y_test_pred)

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
        print("total RMSRE:", total_RMSRE)
        print("train RMSRE:", train_RMSRE)
        print("validation RMSRE:", validation_RMSRE)
        print("test RMSRE:", test_RMSRE)
        print("total RMSPE:", total_RMSPE)
        print("train RMSPE:", train_RMSPE)
        print("validation RMSPE:", validation_RMSPE)
        print("test RMSPE:", test_RMSPE)
        print("total MARE:", total_MARE)
        print("train MARE:", train_MARE)
        print("validation MARE:", validation_MARE)
        print("test MARE:", test_MARE)

        Evaluation_data = [
            [total_RMSE, total_MAPE, total_R_square, total_RMSRE, total_RMSPE, total_MARE],
            [train_RMSE, train_MAPE, train_R_square, train_RMSRE, train_RMSPE, train_MARE],
            [validation_RMSE, validation_MAPE, validation_R_square, validation_RMSRE, validation_RMSPE, validation_MARE],
            [test_RMSE, test_MAPE, test_R_square, test_RMSRE, test_RMSPE, test_MARE],
        ]
        Evaluation_df = pd.DataFrame(
            Evaluation_data,
            index=['total', 'train', 'validation', 'test'],
            columns=['RMSE', 'MAPE', 'R^2', 'RMSRE', 'RMSPE', 'MARE'],
        )
    
    # Split path to get version information
    # path_parts = args.ckpt_path.split('/')  # Split the path by slash
    # version_dir = '/'.join(path_parts[:2])  # Take the first two path segments
    version_dir = os.path.normpath(args.ckpt_path).split(os.sep)[:2]  # Split the path and take the first two path segments
    version_dir = os.path.join(*version_dir)  # Compatible with different operating systems
    # Ensure the directory exists
    os.makedirs(version_dir, exist_ok=True)
    
    # Determine the save folder based on the mode
    if args.mode == 'day-ahead':
        evaluation_dir_name = 'day-ahead_evaluation'
        images_dir_name = 'day-ahead_images'
        prediction_dir_name = 'day-ahead_prediction'
        print("Running in [bold green]day-ahead[/bold green] prediction mode.")
    else:  # 'test' mode
        evaluation_dir_name = 'test_evaluation'
        images_dir_name = 'test_images'
        prediction_dir_name = 'test_prediction'
        print("Running in [bold blue]standard test[/bold blue] mode.")
    
    # Construct the folder for saving evaluation metrics CSV
    Evaluation_save_path = os.path.join(version_dir, evaluation_dir_name)
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
    
    if args.mode == 'day-ahead':
        # In the day-ahead mode, only the part containing the prediction is drawn
        df_plot = df.iloc[-len(y_pred):].copy()

        plt.figure(figsize=SAVE_CONFIG['figsize'])
        ax = df_plot['Close'].plot(**PRICE_STYLE['true'], label='True')
        df_plot['Close Pred'].plot(ax=ax, **PRICE_STYLE['pred'], label='Pred', rot=0)
        
        ax.set_title(f'{target} Day-Ahead Prediction', fontsize=FONT_CONFIG['title_size'])
        ax.set_xlabel('Date', fontsize=FONT_CONFIG['label_size'])
        ax.set_ylabel('Price', fontsize=FONT_CONFIG['label_size'])
        ax.grid(**GRID_STYLE)
        
        # Only display the data legend (True vs Pred)
        plt.legend(**LEGEND_SETTINGS['data'])
        
    else: # standard test mode
        plt.figure(figsize=SAVE_CONFIG['figsize'])
        ax = df['Close'].plot(**PRICE_STYLE['true'], label='True')
        df['Close Pred'].plot(ax=ax, **PRICE_STYLE['pred'], label='Pred', rot=0)
        
        ax.set_title(f'{target} Close Price', fontsize=FONT_CONFIG['title_size'])
        ax.set_xlabel('Date', fontsize=FONT_CONFIG['label_size'])
        ax.set_ylabel('Price', fontsize=FONT_CONFIG['label_size'])
        ax.grid(**GRID_STYLE)

        train_split = df.index[int(len(df) * model.hparams.train_size)]
        validation_split = df.index[int(len(df) * (model.hparams.train_size + model.hparams.validation_size))]
        
        # Draw the background of the data partition
        plt.axvspan(df.index[0], train_split, **COLOR_SETTINGS['train'])
        plt.axvspan(train_split, validation_split, **COLOR_SETTINGS['val'])
        plt.axvspan(validation_split, df.index[-1], **COLOR_SETTINGS['test'])

        data_legend = plt.legend(**LEGEND_SETTINGS['data'])
        partition_handles = [
            Patch(facecolor=v['color'], alpha=v['alpha'], label=f'{k.capitalize()} Set')
            for k, v in COLOR_SETTINGS.items()
        ]
        plt.legend(handles=partition_handles, **LEGEND_SETTINGS['partition'])
        ax.add_artist(data_legend)
    
    # Construct the folder for saving images
    plt_save_path = os.path.join(version_dir, images_dir_name)
    # Ensure the directory exists
    os.makedirs(plt_save_path, exist_ok=True)
    # Construct the PNG image save path
    plt_save_path_png = os.path.join(plt_save_path, f'{target} Close Price Prediction.png')
    # Construct the SVG vector image save path
    plt_save_path_svg = os.path.join(plt_save_path, f'{target} Close Price Prediction.svg')
    
    # Save the parameters according to the mode Settings
    save_kwargs = {
        'dpi': SAVE_CONFIG['dpi'],
        'bbox_inches': SAVE_CONFIG['bbox_inches']
    }
    if args.mode == 'day-ahead':
        save_kwargs['transparent'] = True # Add a transparent background for the day-ahead mode

    plt.savefig(plt_save_path_png, **save_kwargs)
    print(f'saved prediction plot to png "{plt_save_path_png}"')
    plt.savefig(plt_save_path_svg, **save_kwargs)
    print(f'saved prediction plot to svg "{plt_save_path_svg}"')

    # Construct the folder for saving predictions
    pred_save_path = os.path.join(version_dir, prediction_dir_name)
    # Ensure the directory exists
    os.makedirs(pred_save_path, exist_ok=True)
    
    if args.mode == 'day-ahead':
        # In the day-ahead mode, only the overall prediction is saved
        pred_save_path_csv = os.path.join(pred_save_path, f'{target}_pred.csv')
        df_pred = df.iloc[-len(y_pred):].copy() # Only extract the part with the predicted value
        y_pred_df = pd.DataFrame(y_pred, index=df_pred.index, columns=["y_pred"])
        y_pred_df.to_csv(pred_save_path_csv, index=True)
        print(f'{target} Day-Ahead Prediction saved to "{pred_save_path_csv}"')
    else: # standard test mode
        pred_save_path_csv = os.path.join(pred_save_path, f'{target}_pred.csv')
        y_pred_df = pd.DataFrame(y_pred, index=df.index[-len(y_pred):], columns=["y_pred"])
        y_pred_df.to_csv(pred_save_path_csv, index=True)
        print(f'{target} Close Price Prediction saved to "{pred_save_path_csv}"')

        train_pred_save_path_csv = os.path.join(pred_save_path, f'{target}_train_pred.csv')
        y_train_index_end = len(y_validation_pred) + len(y_test_pred)
        y_train_pred_df = pd.DataFrame(y_train_pred, index=df.index[-len(y_pred):-y_train_index_end], columns=["y_train_pred"])
        y_train_pred_df.to_csv(train_pred_save_path_csv, index=True)
        print(f'{target} Close Price Train Prediction saved to "{train_pred_save_path_csv}"')

        validation_pred_save_path_csv = os.path.join(pred_save_path, f'{target}_validation_pred.csv')
        y_validation_pred_df = pd.DataFrame(y_validation_pred, index=df.index[-y_train_index_end:-len(y_test_pred)], columns=["y_validation_pred"])
        y_validation_pred_df.to_csv(validation_pred_save_path_csv, index=True)
        print(f'{target} Close Price Validation Prediction saved to "{validation_pred_save_path_csv}"')

        test_pred_save_path_csv = os.path.join(pred_save_path, f'{target}_test_pred.csv')
        y_test_pred_df = pd.DataFrame(y_test_pred, index=df.index[-len(y_test_pred):], columns=["y_test_pred"])
        y_test_pred_df.to_csv(test_pred_save_path_csv, index=True)
        print(f'{target} Close Price Test Prediction saved to "{test_pred_save_path_csv}"')
