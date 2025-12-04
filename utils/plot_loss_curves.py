import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from argparse import ArgumentParser
from rich.console import Console
from rich.table import Table
import numpy as np

# Use 'Agg' backend to avoid errors on headless servers
matplotlib.use('Agg')
console = Console()


def config_parser() -> ArgumentParser:
    """Configure the command-line argument parser."""
    parser = ArgumentParser(description="Read loss data from multiple CSVLogger files and plot training/validation loss comparison curves.")

    parser.add_argument(
        '--csv-paths',
        type=str,
        nargs='+',
        required=True,
        help="List of paths to all metrics.csv files."
    )
    parser.add_argument(
        '--model-names',
        type=str,
        nargs='+',
        required=True,
        help="List of names corresponding to the models, order should match --csv-paths."
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        default='loss_curve_results/',
        help="Output folder for the results."
    )
    parser.add_argument(
        '--output-basename',
        type=str,
        default='loss_comparison',
        help="Base name for the output plot files (without extension), will be saved as .png and .svg automatically."
    )
    parser.add_argument(
        '--y-log',
        action='store_true',
        help="If set, the Y-axis will use a logarithmic scale (Log Scale)."
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help="(Optional) Limit the maximum number of epochs to plot."
    )
    return parser


def process_loss_data(paths: list[str], model_names: list[str], max_epochs: int | None) -> pd.DataFrame:
    """
    Read all CSV files, extract and process training/validation losses, 
    and consolidate into a single DataFrame.
    """
    # Critical validation: Ensure the number of provided arguments matches
    if len(paths) != len(model_names):
        raise ValueError(
            f"Argument count mismatch! Provided {len(model_names)} model names but {len(paths)} CSV paths."
        )

    all_results = []
    console.print(f"Found {len(paths)} models, starting processing...")

    for model_name, file_path in zip(model_names, paths):
        try:
            # Automatically skip commented lines at the beginning of the CSV file (if any)
            df = pd.read_csv(file_path, comment='#')

            # (Handle 'lossG+lossD')
            # 1. Process training loss (Step-level -> Epoch-level average)

            # Determine the name of the training loss column
            train_loss_col = None
            if 'train_loss' in df.columns:
                train_loss_col = 'train_loss'
            elif 'lossG+lossD' in df.columns:
                train_loss_col = 'lossG+lossD'
                console.print(f"ℹ️ Info: Model '{model_name}' is using 'lossG+lossD' as training loss.", style="dim")

            if train_loss_col is None:
                console.print(
                    f"⚠️ Warning: Model '{model_name}' did not find 'train_loss' or 'lossG+lossD' column in '{file_path}'.",
                    style="yellow")
                continue

            train_df = df[['epoch', train_loss_col]].dropna()
            if train_df.empty:
                console.print(f"⚠️ Warning: Model '{model_name}' has no data in '{train_loss_col}' column in '{file_path}'.",
                              style="yellow")
                continue

            # Group by epoch and calculate the mean
            avg_train_loss = train_df.groupby('epoch')[train_loss_col].mean().reset_index()
            avg_train_loss.rename(columns={train_loss_col: 'avg_train_loss'}, inplace=True)

            # 2. Process validation loss (Already Epoch-level)
            if 'val_loss' not in df.columns:
                console.print(f"⚠️ Warning: Model '{model_name}' did not find 'val_loss' column in '{file_path}'. Will only plot training loss.",
                              style="yellow")
                val_df = pd.DataFrame(columns=['epoch', 'val_loss'])  # Create an empty one
            else:
                val_df = df[['epoch', 'val_loss']].dropna()
                if val_df.empty:
                    console.print(f"ℹ️ Info: Model '{model_name}' has no data in 'val_loss' column in '{file_path}'.",
                                  style="dim")

            # 3. Merge training and validation losses
            model_df = pd.merge(avg_train_loss, val_df, on='epoch', how='left')
            model_df['model'] = model_name

            # 4. (Optional) Truncate Epochs
            if max_epochs is not None:
                model_df = model_df[model_df['epoch'] < max_epochs]  # Use < instead of <=

            all_results.append(model_df)
            console.print(f"  [green]✓[/green] Successfully processed model: [cyan]{model_name}[/cyan] (Total {len(model_df)} epochs)")

        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            console.print(f"❌ Error: Failed to process file '{file_path}' (Model: {model_name}). Error: {e}", style="bold red")
        except Exception as e:
            console.print(f"❌ Unexpected error: Error processing '{file_path}': {e}", style="bold red")

    if not all_results:
        raise ValueError("No CSV files were processed successfully.")

    return pd.concat(all_results, ignore_index=True)


def get_plot_style(model_name: str) -> dict:
    """
    Return specific color and linestyle based on the model name.
    """
    # Default style
    style = {'color': 'gray', 'linestyle': ':'}

    # Dimension 1: Color (Component Type)
    if 'iTransformer' in model_name:
        style['color'] = '#0072B2'  # Blue
    elif 'Transformer' in model_name:
        style['color'] = '#D55E00'  # Orange

    # Dimension 2: Linestyle (Framework Type)
    if 'RevIN-CNN' in model_name:
        style['linestyle'] = '-'  # Solid
    elif 'Transformer' in model_name:  # Ensure baseline model is dashed
        style['linestyle'] = '--'  # Dashed

    return style


def plot_train_loss_curves(df: pd.DataFrame, output_basename: str, output_folder: str, y_log: bool):
    """
    Plot and save the *training* loss curves based on the consolidated DataFrame.
    """
    # ===================== Plotting Configuration (Emulating your style) =====================
    FONT_CONFIG = {
        'family': 'Times New Roman',
        'title_size': 16,
        'label_size': 15,
        'legend_size': 15,
        'tick_size': 14,
    }
    GRID_STYLE = {
        'alpha': 0.2,
        'color': 'gray',
        'linestyle': '--',
        'linewidth': 0.8
    }
    SAVE_CONFIG = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'figsize': (10, 6)
    }
    # =========================================================================

    plt.rcParams['font.family'] = FONT_CONFIG['family']
    fig, ax = plt.subplots(figsize=SAVE_CONFIG['figsize'])

    model_names = df['model'].unique()
    # Remove color_cycle
    # color_cycle = plt.cm.get_cmap('tab10', len(model_names) if len(model_names) > 0 else 1)

    for i, model_name in enumerate(model_names):
        model_data = df[df['model'] == model_name].sort_values(by='epoch')

        # Get style using the new function
        style = get_plot_style(model_name)

        # Plot only training loss (solid line)
        if 'avg_train_loss' in model_data.columns and not model_data['avg_train_loss'].isnull().all():
            ax.plot(
                model_data['epoch'],
                model_data['avg_train_loss'],
                label=model_name,  # Simplified legend
                color=style['color'],  # <-- Apply custom color
                linestyle=style['linestyle'],  # <-- Apply custom linestyle
                alpha=0.9
            )

    # Set chart elements
    ax.set_xlabel('Epoch', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('Train Loss', fontsize=FONT_CONFIG['label_size'])  # Differentiate Y-axis

    if y_log:
        ax.set_yscale('log')
        ax.set_ylabel('Train Loss (Log Scale)', fontsize=FONT_CONFIG['label_size'])  # Differentiate Y-axis

    ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick_size'])
    ax.legend(title='Model', fontsize=FONT_CONFIG['legend_size'], loc='upper right')
    ax.grid(**GRID_STYLE)

    if not df.empty:
        ax.set_xlim(left=0)

    # Modify save filename
    png_path = os.path.join(output_folder, f"{output_basename}_train_loss.png")
    svg_path = os.path.join(output_folder, f"{output_basename}_train_loss.svg")

    plt.savefig(png_path, **{k: v for k, v in SAVE_CONFIG.items() if k != 'figsize'}, transparent=True)
    console.print(f"📈 Training loss plot saved as PNG: [bold cyan]{png_path}[/bold cyan]")

    plt.savefig(svg_path, **{k: v for k, v in SAVE_CONFIG.items() if k != 'figsize'}, transparent=True)
    console.print(f"📈 Training loss plot saved as SVG: [bold cyan]{svg_path}[/bold cyan]")

    plt.close(fig)


def plot_val_loss_curves(df: pd.DataFrame, output_basename: str, output_folder: str, y_log: bool):
    """
    Plot and save the *validation* loss curves based on the consolidated DataFrame.
    """
    # Check if there is any plottable validation data
    if 'val_loss' not in df.columns or df['val_loss'].isnull().all():
        console.print("⚠️ Warning: No valid 'val_loss' data found, skipping validation loss plot generation.", style="yellow")
        return

    # ===================== Plotting Configuration (Emulating your style) =====================
    FONT_CONFIG = {
        'family': 'Times New Roman',
        'title_size': 16,
        'label_size': 15,
        'legend_size': 15,
        'tick_size': 14,
    }
    GRID_STYLE = {
        'alpha': 0.2,
        'color': 'gray',
        'linestyle': '--',
        'linewidth': 0.8
    }
    SAVE_CONFIG = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'figsize': (10, 6)
    }
    # =========================================================================

    plt.rcParams['font.family'] = FONT_CONFIG['family']
    fig, ax = plt.subplots(figsize=SAVE_CONFIG['figsize'])

    model_names = df['model'].unique()
    # Remove color_cycle
    # color_cycle = plt.cm.get_cmap('tab10', len(model_names) if len(model_names) > 0 else 1)

    for i, model_name in enumerate(model_names):
        model_data = df[df['model'] == model_name].sort_values(by='epoch')

        # Get style using the new function
        style = get_plot_style(model_name)

        # Plot only validation loss (dashed line)
        if 'val_loss' in model_data.columns and not model_data['val_loss'].isnull().all():
            ax.plot(
                model_data['epoch'],
                model_data['val_loss'],
                label=model_name,  # Simplified legend
                color=style['color'],  # <-- Apply custom color
                linestyle=style['linestyle'],  # <-- Apply custom linestyle
                alpha=0.9
            )

    # Set chart elements
    ax.set_xlabel('Epoch', fontsize=FONT_CONFIG['label_size'])
    ax.set_ylabel('Validation Loss', fontsize=FONT_CONFIG['label_size'])  # Differentiate Y-axis

    if y_log:
        ax.set_yscale('log')
        ax.set_ylabel('Validation Loss (Log Scale)', fontsize=FONT_CONFIG['label_size'])  # Differentiate Y-axis

    ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick_size'])
    ax.legend(title='Model', fontsize=FONT_CONFIG['legend_size'], loc='upper right')
    ax.grid(**GRID_STYLE)

    if not df.empty:
        ax.set_xlim(left=0)

    # Modify save filename
    png_path = os.path.join(output_folder, f"{output_basename}_val_loss.png")
    svg_path = os.path.join(output_folder, f"{output_basename}_val_loss.svg")

    plt.savefig(png_path, **{k: v for k, v in SAVE_CONFIG.items() if k != 'figsize'}, transparent=True)
    console.print(f"📈 Validation loss plot saved as PNG: [bold cyan]{png_path}[/bold cyan]")

    plt.savefig(svg_path, **{k: v for k, v in SAVE_CONFIG.items() if k != 'figsize'}, transparent=True)
    console.print(f"📈 Validation loss plot saved as SVG: [bold cyan]{svg_path}[/bold cyan]")

    plt.close(fig)


def print_summary_table(df: pd.DataFrame):
    """Print a summary table of the best losses using Rich."""
    table = Table(title="Model Loss Final Results Summary (Based on Minimum Validation Loss)")
    table.add_column("Model", justify="right", style="cyan", no_wrap=True)
    table.add_column("Best Val Epoch", justify="center", style="magenta")
    table.add_column("Min Val Loss", justify="center", style="green")
    table.add_column("Train Loss (at Best Val)", justify="center", style="yellow")

    model_names = df['model'].unique()
    for model_name in model_names:
        model_data = df[df['model'] == model_name]

        if 'val_loss' in model_data.columns and not model_data['val_loss'].isnull().all():
            # Find the row corresponding to the minimum validation loss
            best_row = model_data.loc[model_data['val_loss'].idxmin()]
            epoch = int(best_row['epoch'])
            min_val_loss = best_row['val_loss']
            train_loss_at_best = best_row['avg_train_loss']
            table.add_row(
                model_name,
                str(epoch),
                f"{min_val_loss:.4e}",  # Use scientific notation
                f"{train_loss_at_best:.4e}"
            )
        else:
            # If no validation loss, report only the final training loss
            last_row = model_data.sort_values(by='epoch').iloc[-1]
            epoch = int(last_row['epoch'])
            train_loss = last_row['avg_train_loss']
            table.add_row(
                model_name,
                str(epoch),
                "[dim]N/A[/dim]",
                f"{train_loss:.4e}"
            )
    console.print(table)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    # Create the output folder immediately to ensure it exists before saving any files
    try:
        os.makedirs(args.output_folder, exist_ok=True)
        console.print(f"📁 Ensured output directory exists: [bold cyan]{args.output_folder}[/bold cyan]")
    except Exception as e:
        console.print(f"❌ Failed to create directory: {e}", style="bold red")
        exit(1)  # Exit if directory creation fails

    console.rule("[bold blue]1. Processing loss CSV files...[/bold blue]")
    try:
        results_df = process_loss_data(args.csv_paths, args.model_names, args.max_epochs)

        # (Optional) Save the consolidated CSV file, now that the directory exists, it's safe to save
        csv_path = os.path.join(args.output_folder, f"{args.output_basename}_summary_data.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.8f')
        console.print(f"📄 Consolidated loss data saved to: [bold cyan]{csv_path}[/bold cyan]")

        # Print summary table
        print_summary_table(results_df)

        console.rule("[bold blue]2. Generating loss curve plots...[/bold blue]")
        if not results_df.empty:
            # Call the two plotting functions separately
            plot_train_loss_curves(
                results_df,
                args.output_basename,
                args.output_folder,
                args.y_log
            )
            plot_val_loss_curves(
                results_df,
                args.output_basename,
                args.output_folder,
                args.y_log
            )
        else:
            console.print("❌ No data available for plotting, skipping plotting step.", style="bold red")

    except (ValueError, FileNotFoundError) as e:
        console.print(f"❌ Error during execution: {e}", style="bold red")
    except Exception as e:
        console.print(f"❌ An unexpected error occurred: {e}", style="bold red")

    console.rule("[bold green]✅ All tasks completed![/bold green]")