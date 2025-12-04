import matplotlib

matplotlib.use('Agg')  # set before import pyplot
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from argparse import ArgumentParser
from rich import print
from rich import traceback
from pytorch_lightning import seed_everything

from .model_RevINCNNiTransformerBiLSTM_PinballLoss import RevINCNNiTransformerBiLSTM as RevINCNNiTransformerBiLSTM_PinballLoss
from .model_iTransformer_PinballLoss import iTransformer as iTransformer_PinballLoss
from .model_RevINCNNTransformerBiLSTM_PinballLoss import RevINCNNTransformerBiLSTM as RevINCNNTransformerBiLSTM_PinballLoss
from .model_Transformer_PinballLoss import Transformer as Transformer_PinballLoss

from .data import StockDataSet


def config_parser(parser: ArgumentParser = ArgumentParser()) -> ArgumentParser:
    """
    Configures the command-line argument parser.
    """
    parser.add_argument(
        '--median_ckpt_paths',
        type=str,
        nargs='+',
        required=True,
        help='One or more checkpoint file paths for the median prediction model (q=0.5)'
    )
    parser.add_argument(
        '--lower_ckpt_paths',
        type=str,
        nargs='+',
        required=True,
        help='List of checkpoint paths for the lower bound model (q=0.025), corresponding to the median models'
    )
    parser.add_argument(
        '--upper_ckpt_paths',
        type=str,
        nargs='+',
        required=True,
        help='List of checkpoint paths for the upper bound model (q=0.975), corresponding to the median models'
    )
    parser.add_argument(
        '--model_classes',
        type=str,
        nargs='+',
        required=True,
        help='List of Python class names corresponding to each model group'
    )
    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+',
        required=True,
        help='List of custom names for each model group, used for legends and reports'
    )
    parser.add_argument(
        '--data_csv_path',
        type=str,
        default='data/processed_dataset.csv',
        help='Path to the raw data CSV file for testing'
    )
    parser.add_argument(
        '--test_target',
        type=str,
        default='Apple',
        help='Target stock name (used for legends and filenames)'
    )
    parser.add_argument(
        '--date_range',
        type=str,
        nargs=2,
        default=None,
        help='Optional date range for plotting, format "YYYY-MM-DD YYYY-MM-DD"'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ci_comparison_results',
        help='Output directory to save plots and evaluation metrics'
    )
    return parser


def load_predictions(model_class, ckpt_path, X_data):
    """Loads a model and generates predictions."""
    model = model_class.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()
    predictions = model(X_data.to(model.device))
    return predictions.cpu()  # Return a Tensor, will be processed later uniformly


if __name__ == '__main__':
    traceback.install()
    
    # repeatable
    seed_everything(seed=2025, workers=True)
    
    # Dynamic model mapping
    model_map = {
        'RevIN_CNN_iTransformer_BiLSTM_PinballLoss': RevINCNNiTransformerBiLSTM_PinballLoss,
        'iTransformer_PinballLoss': iTransformer_PinballLoss,
        'RevIN_CNN_Transformer_BiLSTM_PinballLoss': RevINCNNTransformerBiLSTM_PinballLoss,
        'Transformer_PinballLoss': Transformer_PinballLoss,
    }

    # 1. Parse command-line arguments
    parser = config_parser()
    args = parser.parse_args()

    # Validate that the lengths of the argument lists are consistent
    num_models = len(args.median_ckpt_paths)
    if not all(len(lst) == num_models for lst in
               [args.lower_ckpt_paths, args.upper_ckpt_paths, args.model_classes, args.model_names]):
        raise ValueError("All model-related parameter lists (paths, classes, names) must have the same length!")

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load and process data independently
    print("Loading hyperparameters from the first model to ensure data consistency...")
    first_model_class = model_map.get(args.model_classes[0])
    if not first_model_class:
        raise ValueError(f"Model class '{args.model_classes[0]}' not found in model_map.")

    base_model = first_model_class.load_from_checkpoint(args.median_ckpt_paths[0])
    hparams = base_model.hparams
    num_days_for_predict = hparams.num_days_for_predict
    num_days_to_predict = hparams.num_days_to_predict

    print(f"Loading and preprocessing data from user-provided CSV: '{args.data_csv_path}'...")

    raw_dataset = StockDataSet.from_preprocessed(path=args.data_csv_path, target=args.test_target)
    X_raw, Y_raw_full = raw_dataset[:]  # Get the scaled raw sequence data

    # Manually perform the same windowing operation as during training
    X = torch.from_numpy(np.array([X_raw[i:i + num_days_for_predict] for i in
                                   range(len(X_raw) - num_days_for_predict - num_days_to_predict + 1)]))
    Y = torch.from_numpy(np.array([Y_raw_full[i:i + num_days_for_predict + num_days_to_predict] for i in
                                   range(len(Y_raw_full) - num_days_for_predict - num_days_to_predict + 1)]))

    y_true = raw_dataset.y_scaler.inverse_transform(Y[:, num_days_for_predict].cpu()).flatten()

    print("Data preparation complete.")

    # 3. Load all models and generate predictions
    all_predictions = {}
    for i in range(num_models):
        model_name = args.model_names[i]
        model_class_name = args.model_classes[i]
        print(f"\nProcessing model [cyan]'{model_name}'[/cyan] (class: {model_class_name})...")

        model_class = model_map.get(model_class_name)
        if not model_class:
            raise ValueError(f"Model class '{model_class_name}' not found in model_map.")

        # Use the unified X data for prediction and the unified scaler for inverse transformation
        median_preds_tensor = load_predictions(model_class, args.median_ckpt_paths[i], X)
        lower_preds_tensor = load_predictions(model_class, args.lower_ckpt_paths[i], X)
        upper_preds_tensor = load_predictions(model_class, args.upper_ckpt_paths[i], X)

        all_predictions[model_name] = {
            'median': raw_dataset.y_scaler.inverse_transform(median_preds_tensor).flatten(),
            'lower': raw_dataset.y_scaler.inverse_transform(lower_preds_tensor).flatten(),
            'upper': raw_dataset.y_scaler.inverse_transform(upper_preds_tensor).flatten(),
        }

    # 4. Create DataFrame and filter by date range
    full_df = pd.DataFrame({'True': y_true}, index=raw_dataset.df.index[-len(y_true):])
    for name, preds in all_predictions.items():
        full_df[f'{name}_median'] = preds['median']
        full_df[f'{name}_lower'] = preds['lower']
        full_df[f'{name}_upper'] = preds['upper']

    if args.date_range:
        try:
            start_date, end_date = pd.to_datetime(args.date_range[0]), pd.to_datetime(args.date_range[1])
            print(
                f"\nFiltering data between [green]{start_date.date()}[/green] and [green]{end_date.date()}[/green]...")
            plot_df = full_df.loc[start_date:end_date].copy()
            if plot_df.empty: raise ValueError("The specified date range resulted in no data.")
        except Exception as e:
            print(f"[red]Error processing date range: {e}[/red]");
            exit(1)
    else:
        plot_df = full_df.copy()

    # Post-processing to correct quantile crossing
    print("\nApplying post-processing to prevent quantile crossing...")
    for name in args.model_names:
        quantile_columns = [f'{name}_lower', f'{name}_median', f'{name}_upper']
        predictions_array = plot_df[quantile_columns].values
        sorted_predictions = np.sort(predictions_array, axis=1)
        plot_df[quantile_columns] = sorted_predictions
    print("Post-processing complete. Quantile order is now guaranteed.")

    # 5. Calculate evaluation metrics for all models
    metrics_list = []
    for name in args.model_names:
        coverage = np.mean(
            (plot_df['True'] >= plot_df[f'{name}_lower']) & (plot_df['True'] <= plot_df[f'{name}_upper'])) * 100
        width = np.mean(plot_df[f'{name}_upper'] - plot_df[f'{name}_lower'])
        metrics_list.append(
            {'Model': name, 'Coverage_Rate (%)': f"{coverage:.2f}", 'Mean_Interval_Width': f"{width:.4f}"})

    metrics_df = pd.DataFrame(metrics_list)
    print("\n--- Comparative Confidence Interval Metrics ---")
    print(metrics_df)

    metrics_save_path = os.path.join(args.output_dir, f'{args.test_target}_CI_Metrics_Comparison.csv')
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f'\nComparative CI metrics saved to "{metrics_save_path}"')

    # 6. Plot the comparison chart
    print("\nGenerating comparison plot...")
    # ===================== Plotting Configuration =====================
    FONT_CONFIG = {
        'family': 'Times New Roman',
        'title_size': 16,
        'label_size': 15,
        'legend_size': 13,
        'tick_size': 14,
    }
    GRID_STYLE = {
        'alpha': 0.2,
        'color': 'gray',
        'linestyle': '--',
        'linewidth': 0.8,
    }
    LEGEND_SETTINGS = {'loc': 'upper left', 'framealpha': 1.0, 'fontsize': FONT_CONFIG['legend_size']}
    COLOR_CYCLE = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    PLOT_STYLE = {
        'true': {'color': '#1f77b4', 'linewidth': 1.2, 'alpha': 1.0, 'zorder': 10, 'label': 'True Value'},
        'median': {'linewidth': 0.8, 'linestyle': '--', 'alpha': 0.8, 'zorder': 5},
        'ci_fill': {'alpha': 0.15, 'zorder': 1}
    }
    FIG_CONFIG = {'figsize': (12, 4)}
    SAVE_CONFIG = {'dpi': 300, 'bbox_inches': 'tight'}
    # ===================== End of Configuration =====================

    plt.rcParams['font.family'] = FONT_CONFIG['family']
    fig, ax = plt.subplots(**FIG_CONFIG)

    ax.plot(plot_df.index, plot_df['True'], **PLOT_STYLE['true'])
    for i, name in enumerate(args.model_names):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        ax.plot(plot_df.index, plot_df[f'{name}_median'], color=color, **PLOT_STYLE['median'], label=f'{name} - Median')
        ax.fill_between(plot_df.index, plot_df[f'{name}_lower'], plot_df[f'{name}_upper'], color=color,
                        **PLOT_STYLE['ci_fill'], label=f'{name} - 95% CI')

    ax.set_xlabel('Date', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('Price', fontsize=FONT_CONFIG['label_size'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick_size'])
    ax.grid(**GRID_STYLE)

    handles, labels = ax.get_legend_handles_labels()
    true_idx = labels.index('True Value')
    handles = [handles[true_idx]] + [h for i, h in enumerate(handles) if i != true_idx]
    labels = [labels[true_idx]] + [l for i, l in enumerate(labels) if i != true_idx]
    # legend = ax.legend(handles, labels, **LEGEND_SETTINGS)
    legend = ax.legend(
        bbox_to_anchor=(0.00, 1),
        loc='upper left',
        framealpha=LEGEND_SETTINGS['framealpha'],
        fontsize=LEGEND_SETTINGS['fontsize']
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.6)

    png_path = os.path.join(args.output_dir, f'{args.test_target}_CI_Comparison.png')
    plt.savefig(png_path, **SAVE_CONFIG, transparent=True)
    print(f'Comparison plot saved as PNG: "{png_path}"')

    svg_path = os.path.join(args.output_dir, f'{args.test_target}_CI_Comparison.svg')
    plt.savefig(svg_path, **SAVE_CONFIG, transparent=True)
    print(f'Comparison plot saved as SVG: "{svg_path}"')
