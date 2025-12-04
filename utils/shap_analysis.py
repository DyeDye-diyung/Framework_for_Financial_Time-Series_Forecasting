import os
import argparse
import pandas as pd
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
from pytorch_lightning import seed_everything

from .data import StockDataSet
from .model_RevINCNNiTransformerBiLSTM import RevINCNNiTransformerBiLSTM
from .model_iTransformer import iTransformer
from .model_RevINCNNTransformerBiLSTM import RevINCNNTransformerBiLSTM
from .model_Transformer import Transformer

import matplotlib
matplotlib.use('Agg')

# Ignore all FutureWarning from NumPy
warnings.filterwarnings("ignore", category=FutureWarning, module='numpy')
# Sometimes shap can also trigger it, so make it more general
warnings.filterwarnings("ignore", category=FutureWarning)


def config_parser() -> argparse.ArgumentParser:
    """Configures command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform feature importance analysis on a trained deep learning model using SHAP DeepExplainer."
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='calculate',
        choices=['calculate', 'plot'],
        help="Select the running mode: 'calculate' (compute and plot) or 'plot' (load data from files and plot only)."
    )
    parser.add_argument(
        '--model-type',
        type=str,
        help="[Required for 'calculate' mode only] The type of model to load (e.g., 'RevIN_CNN_iTransformer_BiLSTM', 'iTransformer')"
    )
    parser.add_argument(
        '--ckpt-path',
        type=str,
        help="[Required for 'calculate' mode only] Path to the model checkpoint (.ckpt) file."
    )
    parser.add_argument(
        '--num-background-samples',
        type=int,
        default=200,
        help="[Required for 'calculate' mode only] Number of training samples to use for creating the SHAP background dataset."
    )
    parser.add_argument(
        '--num-test-samples',
        type=int,
        default=500,
        help="[Required for 'calculate' mode only] Number of test samples to use for calculating SHAP values."
    )
    parser.add_argument(
        '--shap-values-path',
        type=str,
        help="[Required for 'plot' mode only] Path to the pre-calculated SHAP values .npy file."
    )
    parser.add_argument(
        '--test-sample-path',
        type=str,
        help="[Required for 'plot' mode only] Path to the corresponding test samples .npy file."
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed_dataset.csv',
        help="Path to the processed dataset (.csv) file, used to get feature names."
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        default='shap_analysis_results/',
        help="Output folder to save SHAP results (plots and .npy files)."
    )
    parser.add_argument(
        '--max-display',
        type=int,
        default=None,
        help="The number of the most important features displayed in the SHAP graph. By default, all features are displayed."
    )
    return parser


def main():
    # --- 0. Preparations ---
    # repeatable
    seed_everything(seed=2025, workers=True)
    # Define a mapping from model names to class implementations
    model_map = {
        'RevIN_CNN_iTransformer_BiLSTM': RevINCNNiTransformerBiLSTM,
        'iTransformer': iTransformer,
        'RevIN_CNN_Transformer_BiLSTM': RevINCNNTransformerBiLSTM,
        'Transformer': Transformer,
    }

    # Parse command-line arguments
    parser = config_parser()
    args = parser.parse_args()

    # Mode validity check
    if args.mode == 'calculate':
        if not args.model_type or not args.ckpt_path:
            parser.error("--model-type and --ckpt-path are required in 'calculate' mode.")
    elif args.mode == 'plot':
        if not args.shap_values_path or not args.test_sample_path:
            parser.error("--shap-values-path and --test-sample-path are required in 'plot' mode.")

    # Uniformly get feature names
    print(f"Loading data from '{args.data_path}' to get feature names...")
    df_headers = pd.read_csv(args.data_path, index_col=0, nrows=0)
    target_column = 'Close'
    feature_names = [col for col in df_headers.columns if col != target_column] + [target_column]
    print(f"Successfully obtained {len(feature_names)} feature names.")

    # Execute different operations based on the mode
    if args.mode == 'calculate':
        print("Mode: calculate - Starting to compute SHAP values from the model...")
        if args.model_type not in model_map:
            raise ValueError(f"Unknown model type: {args.model_type}. Please register it in the model_map in shap_analysis.py.")

        model_class = model_map[args.model_type]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        torch.set_float32_matmul_precision('high')

        print(f"Loading model from '{args.ckpt_path}'...")
        model = model_class.load_from_checkpoint(args.ckpt_path)
        model.to(device)
        model.eval()
        model.freeze()

        print("Loading full dataset for splitting...")
        raw_dataset = StockDataSet.from_preprocessed(path=args.data_path, target=model.hparams.target)
        X_full, _ = raw_dataset[:]

        num_days_for_predict = model.hparams.num_days_for_predict
        num_days_to_predict = model.hparams.num_days_to_predict
        X_windowed = np.array([X_full[i: i + num_days_for_predict]
                               for i in range(len(X_full) - num_days_for_predict - num_days_to_predict + 1)])

        train_size, validation_size = model.hparams.train_size, model.hparams.validation_size
        train_end_idx = int(len(X_windowed) * train_size)
        validation_end_idx = int(len(X_windowed) * (train_size + validation_size))
        X_train, X_test = X_windowed[:train_end_idx], X_windowed[validation_end_idx:]

        num_available_train = X_train.shape[0]
        num_background_samples = min(num_available_train, args.num_background_samples)

        if num_background_samples < args.num_background_samples:
            print(f"[Warning] The number of training samples ({num_available_train}) is less than the requested number of background samples ({args.num_background_samples}).")
            print(f"Will use all {num_background_samples} available training samples as background data.")
        else:
            print(f"Preparing SHAP background data ({num_background_samples} samples)...")

        background_indices = np.random.choice(num_available_train, num_background_samples, replace=False)
        X_train_background = torch.from_numpy(X_train[background_indices]).float().to(device)

        num_available_test = X_test.shape[0]
        num_test_samples = min(num_available_test, args.num_test_samples)

        if num_test_samples < args.num_test_samples:
            print(f"[Warning] The number of test samples ({num_available_test}) is less than the requested number of SHAP analysis samples ({args.num_test_samples}).")
            print(f"Will use all {num_test_samples} available test samples for the analysis.")
        else:
            print(f"Preparing SHAP test samples ({num_test_samples} samples)...")

        X_test_sample = torch.from_numpy(X_test[:num_test_samples]).float().to(device)

        print("Creating SHAP GradientExplainer...")
        explainer = shap.GradientExplainer(model, X_train_background)

        print("Calculating SHAP values (this may take some time)...")

        # Import tqdm
        from tqdm import tqdm

        # Define a reasonable batch size to balance performance and memory usage
        shap_batch_size = 32
        shap_values_list = []

        # Temporarily switch to train mode to enable cuDNN backward pass
        model.train()

        # Use tqdm to loop through mini-batches
        for i in tqdm(range(0, num_test_samples, shap_batch_size), desc="Calculating SHAP values"):
            start_idx = i
            end_idx = min(i + shap_batch_size, num_test_samples)
            batch = X_test_sample[start_idx:end_idx]

            # Calculate SHAP values for the current batch
            shap_batch = explainer.shap_values(batch)
            shap_values_list.append(shap_batch)

        # After calculation is complete, revert to evaluation mode
        model.eval()

        # Concatenate the results from all batches into one large numpy array
        shap_values = np.concatenate(shap_values_list, axis=0)

        X_test_sample_np = X_test_sample.cpu().numpy()
        model_type_for_filename = args.model_type

    elif args.mode == 'plot':
        print("Mode: plot - Starting to load SHAP values from file and plot...")
        shap_values = np.load(args.shap_values_path)
        X_test_sample_np = np.load(args.test_sample_path)
        model_type_for_filename = os.path.basename(args.shap_values_path).replace('_shap_values.npy', '')

    print(f"Original SHAP values shape: {shap_values.shape}.")
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        print("Detected 4D SHAP values, squeezing to a 3D array for plotting...")
        shap_values = shap_values.squeeze(-1)

    if X_test_sample_np.ndim != 3:
        print(f"[Warning] X_test_sample_np is not 3-dimensional (actual: {X_test_sample_np.ndim}), plotting might fail.")

    print(f"SHAP values shape for plotting: {shap_values.shape}")

    if args.mode == 'calculate':
        os.makedirs(args.output_folder, exist_ok=True)
        shap_values_path = os.path.join(args.output_folder, f"{args.model_type}_shap_values.npy")
        test_sample_path = os.path.join(args.output_folder, f"{args.model_type}_X_test_sample.npy")
        np.save(shap_values_path, shap_values)
        np.save(test_sample_path, X_test_sample_np)
        print(f"SHAP values (3D) saved to: '{shap_values_path}'")

    # --- Unified Plotting and Saving Logic ---
    print("\n--- Starting to Generate Plots ---")

    num_features = len(feature_names)
    if args.max_display is None:
        display_limit = num_features
    else:
        display_limit = args.max_display
    
    FONT_CONFIG = {
        'family': 'Times New Roman',
        'title_size': 19,
        'label_size': 18,
        'legend_size': 18,
        'tick_size': 18,
    }
    GRID_STYLE = {
        'alpha': 0.2,
        'color': 'gray',
        'linestyle': '--',
        'linewidth': 0.8
    }
    plt.rcParams['font.family'] = FONT_CONFIG['family']

    # 1. Generate and save the global feature importance bar plot
    print("Generating global feature importance bar plot...")
    shap_values_for_bar_plot = np.abs(shap_values).mean(axis=1)

    plt.figure()
    shap.summary_plot(shap_values_for_bar_plot, plot_type="bar", feature_names=feature_names, show=False, max_display=display_limit)
    
    ax = plt.gca()
    ax.grid(axis='x', **GRID_STYLE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.0005))
    plt.xlabel('mean(|SHAP value|)', fontsize=FONT_CONFIG['label_size'])
    plt.xticks(fontsize=FONT_CONFIG['tick_size'])
    plt.yticks(fontsize=FONT_CONFIG['tick_size'])

    bar_plot_path_png = os.path.join(args.output_folder, f"{model_type_for_filename}_shap_bar_plot.png")
    bar_plot_path_svg = os.path.join(args.output_folder, f"{model_type_for_filename}_shap_bar_plot.svg")
    plt.savefig(bar_plot_path_png, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(bar_plot_path_svg, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Bar plot saved to: '{bar_plot_path_png}' and '{bar_plot_path_svg}'")

    # 2. Generate and save the detailed summary plot (Beeswarm)
    print("Generating summary plot (Beeswarm)...")
    # For the beeswarm plot, reshape data from (samples, timesteps, features) -> (samples * timesteps, features)
    shap_values_for_beeswarm = shap_values.reshape(-1, num_features)
    X_test_sample_for_beeswarm = X_test_sample_np.reshape(-1, num_features)

    plt.figure()
    # Use the reshaped data and add max_display
    shap.summary_plot(shap_values_for_beeswarm, X_test_sample_for_beeswarm, feature_names=feature_names, show=False, max_display=display_limit)
    
    cbar = plt.gcf()
    cbar_ax = cbar.axes[1]
    cbar_ax.tick_params(labelsize=FONT_CONFIG['tick_size'])
    current_label = cbar_ax.get_ylabel()
    cbar_ax.set_ylabel(current_label, fontsize=FONT_CONFIG['label_size'])
    plt.xlabel('SHAP value (impact on model output)', fontsize=FONT_CONFIG['label_size'])
    plt.xticks(fontsize=FONT_CONFIG['tick_size'])
    plt.yticks(fontsize=FONT_CONFIG['tick_size'])

    beeswarm_plot_path_png = os.path.join(args.output_folder, f"{model_type_for_filename}_shap_beeswarm_plot.png")
    beeswarm_plot_path_svg = os.path.join(args.output_folder, f"{model_type_for_filename}_shap_beeswarm_plot.svg")
    plt.savefig(beeswarm_plot_path_png, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(beeswarm_plot_path_svg, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Summary plot saved to: '{beeswarm_plot_path_png}' and '{beeswarm_plot_path_svg}'")


if __name__ == '__main__':
    main()
