import os
import pandas as pd


def aggregate_results(root_dir="lightning_logs"):
    # Mapping from version number to model name
    model_mapping = [
        ("version_0", "RevIN-CNN-iTransformer-BiLSTM"),
        ("version_1", "RevIN-CNN-Transformer-BiLSTM"),
        ("version_2", "CNN-BiLSTM-Attention"),
        ("version_3", "SCINet"),
        ("version_4", "GAN"),
        ("version_5", "Transformer"),
        ("version_6", "iTransformer"),
        ("version_7", "CNN-iTransformer-BiLSTM"),
        ("version_8", "RevIN-iTransformer-BiLSTM"),
        ("version_9", "RevIN-CNN-iTransformer"),
        ("version_10", "CNN-Transformer-BiLSTM"),
        ("version_11", "RevIN-Transformer-BiLSTM"),
        ("version_12", "RevIN-CNN-Transformer")
    ]

    # Convert to an OrderedDict to maintain order
    from collections import OrderedDict
    model_mapping_od = OrderedDict(model_mapping)

    # All test sets and their corresponding row names
    test_sets = {
        "Apple": "test",
        "Microsoft": "total",
        "MaoTai": "total",
        "HSBC": "total"
    }

    # Create a summary table for each test set
    for test_set, row_name in test_sets.items():
        # Initialize the summary data structure
        metrics = ["RMSE", "MAPE", "R^2"]
        summary_data = []

        # Process according to the specified model order
        for version, model_name in model_mapping_od.items():
            file_path = os.path.join(root_dir, version, "test_evaluation", f"{test_set}_Evaluation.csv")

            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                # If the file does not exist, fill with NaN values
                row_values = {metric: float('nan') for metric in metrics}
            else:
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path, index_col=0)

                    # Extract data from the specified row
                    row_data = df.loc[row_name]

                    # Collect metric values
                    row_values = {metric: row_data[metric] for metric in metrics}
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    row_values = {metric: float('nan') for metric in metrics}

            # Add model name and metric values
            summary_data.append((model_name, row_values))

        # Create DataFrame (maintaining original order)
        model_names = [x[0] for x in summary_data]
        metric_data = {metric: [x[1][metric] for x in summary_data] for metric in metrics}

        summary_df = pd.DataFrame(metric_data, index=model_names).T

        # Save as CSV
        output_path = f"evaluate_results/{test_set}_Summary.csv"
        summary_df.to_csv(output_path)
        print(f"Saved summary for {test_set} (using {row_name} row) to {output_path}")
        print("Model order preserved as specified in model_mapping")


if __name__ == "__main__":
    aggregate_results()
