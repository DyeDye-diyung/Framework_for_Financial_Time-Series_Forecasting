import matplotlib
matplotlib.use('Agg') # set before import pyplot
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser


def config_parser(parser: ArgumentParser = ArgumentParser(), targets: list[str] = None) -> ArgumentParser:
    """
    Configures the command-line argument parser.

    Args:
        parser (ArgumentParser): The argument parser object.
        targets (list[str]): A list of optional target stocks.

    Returns:
        ArgumentParser: The configured parser.
    """
    parser.add_argument(
        '--test_target',
        type=str,
        default='Apple',
        choices=targets,
        help='Target stock name'
    )
    parser.add_argument(
        '--true_csv_path',
        type=str,
        default='data/Apple.csv',
        help='Path to the ground truth CSV file'
    )
    parser.add_argument(
        '--pred_csv_paths',
        type=str,
        nargs='+',
        default=[
            'lightning_logs/version_0/test_prediction/Apple_test_pred.csv',
            'lightning_logs/version_1/test_prediction/Apple_test_pred.csv',
            'lightning_logs/version_2/test_prediction/Apple_test_pred.csv',
            'lightning_logs/version_3/test_prediction/Apple_test_pred.csv',
            'lightning_logs/version_4/test_prediction/Apple_test_pred.csv',
            'lightning_logs/version_5/test_prediction/Apple_test_pred.csv',
            'lightning_logs/version_6/test_prediction/Apple_test_pred.csv'
        ],
        help='Paths to multiple prediction CSV files'
    )
    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+',
        default=[
            'RevIN_CNN_iTransformer_BiLSTM',
            'RevIN_CNN_Transformer_BiLSTM',
            'CNN_BiLSTM_Attention',
            'SCINet',
            'GRU_GAN',
            'Transformer',
            'iTransformer'
        ],
        help='Model names corresponding to the prediction CSV files'
    )
    parser.add_argument(
        '--date_range',
        type=str,
        nargs=2,
        default=[
            '2015-01-01',
            '2025-01-01'
        ],
        help='Date range for plotting, format "YYYY-MM-DD YYYY-MM-DD", e.g., "2023-01-01 2023-12-31", the right side is not included'
    )
    parser.add_argument(
        '--highlight_models',
        type=str,
        nargs='+',
        default=[
            'RevIN_CNN_iTransformer_BiLSTM',
            'RevIN_CNN_Transformer_BiLSTM',
        ],
        help='List of model names to be highlighted'
    )
    parser.add_argument(
        '--day-ahead',
        action='store_true',
        help='Enable the day-ahead mode and save the images to the "day-ahead images" folder'
    )
    parser.add_argument(
        '--fig_title',
        type=str,
        default=None,
        help='Figure title suffix'
    )
    return parser


if __name__ == '__main__':
    # Parse command-line arguments
    parser = config_parser(targets=['Apple', 'Microsoft', 'MaoTai', 'HSBC'])  # The target list can be expanded as needed
    args = parser.parse_args()
    
    if args.day_ahead and not args.fig_title:
        parser.error("--fig_title is required when --day-ahead is specified.")

    # Verify that the number of model names and prediction paths are consistent
    if len(args.model_names) != len(args.pred_csv_paths):
        raise ValueError('The number of model names must match the number of prediction CSV file paths')

    # Read the ground truth values
    true_df = pd.read_csv(args.true_csv_path, index_col=0, parse_dates=True)
    # If a date range is specified, filter the data
    if args.date_range:
        try:
            start_date = pd.to_datetime(args.date_range[0])
            end_date = pd.to_datetime(args.date_range[1])
            if start_date > end_date:
                raise ValueError('Start date cannot be later than end date')
            true_df = true_df.loc[start_date:end_date]
        except Exception as e:
            print(f'Error in date range: {e}')
            exit(1)
    true_values = true_df['Close'].values

    # Read the predicted values for each model separately
    pred_values_list = []
    for path in args.pred_csv_paths:
        pred_df = pd.read_csv(path, index_col=0, parse_dates=True)
        # If a date range is specified, filter the data
        if args.date_range:
            try:
                start_date = pd.to_datetime(args.date_range[0])
                end_date = pd.to_datetime(args.date_range[1])
                if start_date > end_date:
                    raise ValueError('Start date cannot be later than end date')
                pred_df = pred_df.loc[start_date:end_date]
            except Exception as e:
                print(f'Error in date range: {e}')
                exit(1)
        pred_values = pred_df.values  # Assume predicted values are in a single column
        pred_values_list.append(pred_values)

    # ===================== Plotting Configuration Parameters =====================
    FONT_CONFIG = {
        'family': 'Times New Roman',  # Global font
        'title_size': 16,  # Title font size
        'label_size': 15,  # Axis label font size
        'legend_size': 15,  # Legend font size
        'tick_size': 14,  # Control the font size of the ticks number/date (usually slightly smaller than the label)
    }
    GRID_STYLE = {
        'alpha': 0.2,  # Alpha (transparency)
        'color': 'gray',  # Grid line color
        'linestyle': '--',  # Linestyle (dashed)
        'linewidth': 0.8  # Linewidth
    }
    COLOR_SETTINGS = {
        'train': {'color': 'green', 'alpha': 0.1},
        'val': {'color': 'gold', 'alpha': 0.1},
        'test': {'color': 'salmon', 'alpha': 0.1}
    }
    LEGEND_SETTINGS = {
        'data': {
            'loc': 'lower right',
            'framealpha': 0.5,
            'fontsize': FONT_CONFIG['legend_size']
        },
        'partition': {
            'loc': 'upper right',
            'framealpha': 0.9,
            'fontsize': FONT_CONFIG['legend_size']
        }
    }
    SPLIT_LINE_STYLE = {
        'color': 'gray',
        'linestyle': '--',
        'alpha': 0.8,
        'linewidth': 1.2
    }
    # Define color scheme for value curves
    COLOR_SCHEME = {
        # 'true': '#d62728',  # Changed to high-saturation red
        'true': '#1f77b4',  # Changed to high-saturation blue
        'highlight': ['#ff7f0e', '#2ca02c'],  # Orange and green as highlight colors
        'normal': ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9467bd', '#c49c94']  # Use low-saturation/neutral colors
    }
    PRICE_STYLE = {
        'true': {
            'color': COLOR_SCHEME['true'],  # Changed to standard blue
            'linewidth': 1.2,  # Bolder line width
            'alpha': 1.0,
            'zorder': 3  # Ensure it is on the top layer
        },
        'highlight': {  # New style for highlighted models
            'linewidth': 1.0,
            'linestyle': '-',
            'alpha': 0.8,
            'zorder': 2
        },
        'normal': {  # Normal prediction style
            'linewidth': 0.35,
            'linestyle': '-',
            'alpha': 0.8,
            'zorder': 1
        }
    }
    SAVE_CONFIG = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'figsize': (12, 4)
    }
    # ===================== End of Configuration =====================

    # Apply global font settings
    plt.rcParams['font.family'] = FONT_CONFIG['family']

    # Create figure and axes
    plt.figure(figsize=SAVE_CONFIG['figsize'])
    ax = plt.gca()

    # Plot the true values
    ax.plot(true_df.index, true_values, **PRICE_STYLE['true'], label='True')

    # Dynamically assign colors for highlighted models
    highlight_colors = {model: COLOR_SCHEME['highlight'][i % len(COLOR_SCHEME['highlight'])]
                        for i, model in enumerate(args.highlight_models)}

    # Plot the predicted values for each model
    for i, (pred_values, model_name) in enumerate(zip(pred_values_list, args.model_names)):
        # Determine the style
        if model_name in args.highlight_models:
            style = {
                **PRICE_STYLE['highlight'],
                'color': highlight_colors[model_name],
                'label': f'* {model_name}'  # Add a special marker
            }
        else:
            style = {
                **PRICE_STYLE['normal'],
                'color': COLOR_SCHEME['normal'][i % len(COLOR_SCHEME['normal'])],
                'label': model_name
            }
        ax.plot(true_df.index[-len(pred_values):], pred_values, **style)

    # Set title and axis labels
    ax.set_xlabel('Date', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('Price', fontsize=FONT_CONFIG['label_size'])
    # Apply the ticks number/date font sizes on the X-axis and Y-axis
    ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick_size'])

    # Apply grid style
    ax.grid(**GRID_STYLE)

    # Add data legend
    # data_legend = ax.legend(**LEGEND_SETTINGS['data'])
    data_legend = ax.legend(
        bbox_to_anchor=(1.00, 1),
        loc='upper left',
        framealpha=LEGEND_SETTINGS['data']['framealpha'],
        fontsize=LEGEND_SETTINGS['data']['fontsize']
    )
    ax.add_artist(data_legend)

    # Save the figure
    if args.day_ahead:
        save_folder = 'day-ahead images'
        file_name_base = f'{args.test_target}_Close_Price_{args.fig_title}_Day-ahead_Prediction'
        print(f"The Day-ahead mode is enabled and the image will be saved to the '{save_folder}' folder.")
    else:
        save_folder = 'images'
        file_name_base = f'{args.test_target}_Close_Price_Prediction'

    os.makedirs(save_folder, exist_ok=True)

    png_save_path = os.path.join(save_folder, f'{file_name_base}.png')
    svg_save_path = os.path.join(save_folder, f'{file_name_base}.svg')

    plt.savefig(png_save_path, dpi=SAVE_CONFIG['dpi'],
                bbox_inches=SAVE_CONFIG['bbox_inches'], transparent=True, bbox_extra_artists=(data_legend,))
    print(f'saved prediction plot to "{png_save_path}"')
    plt.savefig(svg_save_path, dpi=SAVE_CONFIG['dpi'],
                bbox_inches=SAVE_CONFIG['bbox_inches'], transparent=True, bbox_extra_artists=(data_legend,))
    print(f'saved prediction plot to "{svg_save_path}"')
