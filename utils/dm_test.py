import os
import pandas as pd
import numpy as np
import argparse
import statsmodels.api as sm


def calculate_errors(y_true, y_pred, crit='MSE'):
    """
    Calculates the error series according to the specified criterion.

    Args:
        y_true (pd.Series): The series of true values.
        y_pred (pd.Series): The series of predicted values.
        crit (str, optional): The loss criterion ('MSE', 'MAD', 'MAPE'). Defaults to 'MSE'.

    Returns:
        pd.Series: The calculated error series.

    Raises:
        ValueError: If crit is not a supported type.
    """
    if crit == 'MSE':
        # Calculate the component of Mean Squared Error for each data point (i.e., squared error)
        return (y_true - y_pred) ** 2
    elif crit == 'MAD':
        # Calculate the component of Mean Absolute Deviation for each data point (i.e., absolute error)
        return abs(y_true - y_pred)
    elif crit == 'MAPE':
        # Calculate the component of Mean Absolute Percentage Error for each data point
        # To prevent division by zero when the true value is 0, add a very small number epsilon to the denominator
        return abs((y_true - y_pred) / (y_true + 1e-8))
    else:
        raise ValueError(f"Unsupported criterion: {crit}. Please choose from 'MSE', 'MAD', 'MAPE'.")


def diebold_mariano_test(ref_errors, comp_errors, h=1):
    """
    Implements the Diebold-Mariano test using the statsmodels library.
    This method determines if there is a significant difference in the predictive accuracy of two models
    by testing whether the mean of the difference between their error series (the loss differential) is significantly different from zero.

    Args:
        ref_errors (pd.Series): The error series of the reference model.
        comp_errors (pd.Series): The error series of the comparison model.
        h (int, optional): The forecast horizon. Defaults to 1.

    Returns:
        tuple: (DM statistic, p-value)
    """
    # 1. Calculate the loss differential series d = e1 - e2
    loss_diff = ref_errors - comp_errors

    # 2. Perform an OLS regression of the loss differential series against a constant term
    # and use HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors to handle potential autocorrelation in the errors.
    model = sm.OLS(loss_diff, np.ones(len(loss_diff)))

    # maxlags = h-1. For a single-step forecast (h=1), maxlags would theoretically be 0.
    # For more robust results, it is set to 1 here to capture potential MA(1) errors.
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})

    # 3. Extract the t-statistic and p-value from the regression results, which correspond to the DM statistic and p-value, respectively
    dm_stat = results.tvalues[0]
    p_value = results.pvalues[0]

    return dm_stat, p_value


def main():
    """
    Main execution function:
    1. Parses command-line arguments, including selecting a specific subset to test.
    2. Loads the ground truth data.
    3. Loads the predictions of various models and performs the DM test based on the specified subset.
    4. Prints the results to the console and saves them to a CSV file.
    """
    # Step 1: Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Perform Diebold-Mariano test to compare forecast accuracy of models.")
    parser.add_argument('--root_dir', type=str, default='lightning_logs', help='Root directory of the lightning logs.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the ground truth CSV files.')
    parser.add_argument('--output_dir', type=str, default='DM_Test_results',
                        help='Directory to save the DM test result CSV files.')
    parser.add_argument('--test_set', type=str, required=True, help='Name of the test set (e.g., Apple, Microsoft).')
    parser.add_argument('--reference_model', type=str, required=True, help='Name of the reference model.')
    parser.add_argument('--comparison_models', type=str, nargs='+', required=True,
                        help='List of comparison model names.')
    parser.add_argument('--crit', type=str, default='MSE', choices=['MSE', 'MAD', 'MAPE'],
                        help='The loss criterion for the DM test (default: MSE).')
    # --- New Feature: Select the data subset on which to perform the test ---
    parser.add_argument('--subset', type=str, required=True, choices=['All', 'Train', 'Validation', 'Test'],
                        help="The data subset to perform the test on (e.g., 'Test').")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 2: Define the mapping from model names to their log folders (version numbers)
    model_mapping = {
        "RevIN-CNN-iTransformer-BiLSTM": "version_0",
        "RevIN-CNN-Transformer-BiLSTM": "version_1",
        "CNN-BiLSTM-Attention": "version_2",
        "SCINet": "version_3",
        "GAN": "version_4",
        "Transformer": "version_5",
        "iTransformer": "version_6",
        "CNN-iTransformer-BiLSTM": "version_7",
        "RevIN-iTransformer-BiLSTM": "version_8",
        "RevIN-CNN-iTransformer": "version_9",
        "CNN-Transformer-BiLSTM": "version_10",
        "RevIN-Transformer-BiLSTM": "version_11",
        "RevIN-CNN-Transformer": "version_12"
    }

    # Step 3: Load the complete ground truth data
    try:
        ground_truth_path = os.path.join(args.data_dir, f"{args.test_set}.csv")
        ground_truth_df = pd.read_csv(ground_truth_path)
        ground_truth_df['Date'] = pd.to_datetime(ground_truth_df['Date'])
        ground_truth_df.set_index('Date', inplace=True)
        print(f"Successfully loaded ground truth data from: {ground_truth_path}")
    except FileNotFoundError:
        print(f"ERROR: Ground truth file not found at {ground_truth_path}.")
        return

    # Step 4: Determine the subset name and file suffix for the test based on user input
    subset_map = {
        "All": ("All_Data", "pred"),
        "Train": ("Train_Set", "train_pred"),
        "Validation": ("Validation_Set", "validation_pred"),
        "Test": ("Test_Set", "test_pred")
    }
    subset_name, subset_suffix = subset_map[args.subset]

    # Step 5: Perform the DM test
    print(
        f"\n========== DM Test for '{args.test_set}' - {subset_name.replace('_', ' ')} (Criterion: {args.crit}) ==========")
    print(f"Reference Model: {args.reference_model}\n")

    try:
        ref_version = model_mapping[args.reference_model]
        ref_pred_path = os.path.join(args.root_dir, ref_version, 'test_prediction',
                                     f'{args.test_set}_{subset_suffix}.csv')
        ref_preds_df = pd.read_csv(ref_pred_path)
        ref_preds_df['Date'] = pd.to_datetime(ref_preds_df.iloc[:, 0])
        ref_preds_df.set_index('Date', inplace=True)
    except (KeyError, FileNotFoundError):
        print(
            f"ERROR: Could not load predictions for reference model '{args.reference_model}' for subset '{subset_name}'. Aborting.")
        return

    aligned_true_values = ground_truth_df.loc[ref_preds_df.index, 'Close']
    ref_preds = ref_preds_df.iloc[:, -1]

    if len(aligned_true_values) != len(ref_preds):
        print(f"ERROR: Length mismatch for reference model. Aborting.")
        return

    ref_errors = calculate_errors(aligned_true_values, ref_preds, crit=args.crit)
    results = []

    for model_name in args.comparison_models:
        if model_name == args.reference_model:
            continue

        try:
            comp_version = model_mapping[model_name]
            comp_pred_path = os.path.join(args.root_dir, comp_version, 'test_prediction',
                                          f'{args.test_set}_{subset_suffix}.csv')
            comp_preds_df = pd.read_csv(comp_pred_path)
            comp_preds_df['Date'] = pd.to_datetime(comp_preds_df.iloc[:, 0])
            comp_preds_df.set_index('Date', inplace=True)
            comp_preds = comp_preds_df.loc[ref_preds_df.index].iloc[:, -1]

            if len(aligned_true_values) != len(comp_preds):
                print(f"Warning: Length mismatch for '{model_name}'. Skipping.")
                continue

            comp_errors = calculate_errors(aligned_true_values, comp_preds, crit=args.crit)
            dm_stat, p_value = diebold_mariano_test(ref_errors, comp_errors)

            results.append({
                'Comparison Model': model_name,
                'DM Statistic': f"{dm_stat:.4f}",
                'p-value': f"{p_value:.4f}"
            })

        except (KeyError, FileNotFoundError):
            print(f"Warning: Prediction file for '{model_name}' (subset: {subset_name}) not found. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing '{model_name}': {e}. Skipping.")

    # Step 6: Format, print, and save the results
    if results:
        results_df = pd.DataFrame(results)
        print(f"DM Test Results (comparing against {args.reference_model}):\n")
        print(results_df.to_string(index=False))

        output_filename = f"{args.reference_model}_{args.test_set}_{subset_name}_DM_Test_Summary.csv"
        output_path = os.path.join(args.output_dir, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    else:
        print("No comparison models were processed.")

    print("\n================== Test Completed ==================")


if __name__ == "__main__":
    main()
