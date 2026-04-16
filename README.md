# Enhancing stock price forecasting with a modular deep learning framework incorporating plug-and-play Transformer variants

This repository contains the official source code and data for the research paper: **"Enhancing stock price forecasting with a modular deep learning framework incorporating plug-and-play Transformer variants"**.

Our proposed framework, **RevIN-CNN-(Transformer Variants)-BiLSTM**, is a modular and robust deep learning architecture designed to tackle the inherent challenges of financial time series forecasting, such as high noise, non-stationarity, and complex dependencies.

---

## 🏛️ Framework Architecture

The core of our work is a synergistic, modular framework where each component targets a specific challenge in financial forecasting:

<p align="center">
  <img src="model_architecture.svg" width="800">
</p>

-   **RevIN (Reversible Instance Normalization):** Mitigates distribution shifts common in non-stationary time series.
-   **CNN (Convolutional Neural Network):** Extracts local, short-term patterns and features.
-   **Transformer Variants:** A pluggable core (e.g., vanilla Transformer, iTransformer) to capture long-range dependencies.
-   **BiLSTM (Bidirectional LSTM):** Models bidirectional sequential dynamics, capturing information from both past and future contexts.

---

## ✨ Key Features

-   **High Versatility:** The modular design allows for easy substitution of the Transformer core, making the framework adaptable to future advancements in Transformer architectures.
-   **Statistical Robustness:** Effectively handles non-stationary data with extreme values, validated by strong cross-market generalization performance.
-   **Proven Superiority:** Outperforms various baseline models significantly, with performance gains validated by rigorous Diebold-Mariano statistical tests.
-   **Full Reproducibility:** All code, sample data, and experimental configurations are provided to ensure the full reproducibility of our findings.

---

## 🚀 Getting Started

Follow these instructions to set up the project environment and reproduce our results.

### Prerequisites

-   Python 3.12
-   Conda or venv for environment management
-   An NVIDIA GPU with CUDA support is recommended for training.

### Installation

Due to the project's reliance on specific GPU hardware acceleration (CUDA), the installation process requires `conda` to properly manage the environment. Please follow these steps carefully.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DyeDye-diyung/Framework_for_Financial_Time-Series_Forecasting.git](https://github.com/DyeDye-diyung/Framework_for_Financial_Time-Series_Forecasting.git)
    cd Framework_for_Financial_Time-Series_Forecasting
    ```

2.  **Create and activate a Conda environment:**
    
    It is highly recommended to create a new Conda environment to avoid conflicts with existing packages.

    ```bash
    conda create -n stock_forecast python=3.12
    conda activate stock_forecast
    ```
    
3.  **Install Core GPU and PyTorch Dependencies:**

    This is the most critical step for ensuring hardware acceleration works correctly. This project requires a specific build of PyTorch (v2.3.1) that is compatible with CUDA 11.8 and Python 3.12.

    *Note: This specific version may only be available through particular Conda channels (in this case, the Tsinghua University mirror for PyTorch). Official channels may not host this exact build. Therefore, it is **essential** to use the following command which explicitly points to the correct channels to install PyTorch, the CUDA Toolkit, and cuDNN together. This ensures perfect compatibility between all core libraries.*

    ```bash
    conda install cudatoolkit=11.8.0 cudnn=8.8.0 pytorch=2.3.1=py3.12_cuda11.8_cudnn8_0 torchvision=0.18.1=py312_cu118 torchaudio=2.3.1=py312_cu118 pytorch-cuda=11.8 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64 -c nvidia
    ```

4. **Install Remaining Dependencies:**

    Once the core libraries are installed, you can install the remaining Python packages using `pip` and the `requirements.txt` file.

    ```bash
    python -m pip install -r requirements.txt
    ```
   
    Or,  you can install the remaining Python packages using `conda`.

    ```bash
    conda install --yes --file requirements.txt
    ```

5.  **(Fallback Option) Alternate Installation Methods:**

    If you encounter persistent dependency conflicts with the standard installation, you can use one of the following fallback methods to replicate the exact project environment.

    **Method A: Create from Environment File (Recommended)**

    This method uses the `environment.yml` file included in the repository to automatically create a new Conda environment with all specified packages and versions.

    ```bash
    # This command creates an environment named 'transformer_stock_prediction' as specified in the file
    conda env create -f environment.yml
    conda activate transformer_stock_prediction
    ```

    **Method B: Direct Environment Cloning**

    This method involves downloading a complete, pre-built environment folder. It is faster but may be less portable across different operating systems.

    1.  Download the compressed environment folder from: **[Google Drive Link](https://drive.google.com/drive/folders/1KNErZCA-WFEPURbE0Mudm3GM_Glctggv?usp=drive_link)**. *(If the link is inaccessible and you require this file, please contact the corresponding author of our paper via email.)*
    2.  Extract the `.7z` file and copy the resulting folder (e.g., `transformer_stock_prediction`) into your main Conda environments directory (e.g., `C:\Users\Admin\anaconda3\envs`, you can find the path by running `conda info --envs`). Example path after copying: `C:\Users\Admin\anaconda3\envs\transformer_stock_prediction`.
    3.  Run the following commands to register and activate the environment:

    ```bash
    # Add your envs directory to conda's configuration to ensure it's recognized
    # Replace the path below with your actual envs path
    conda config --append envs_dirs C:\Users\Admin\anaconda3\envs
    
    # Verify that conda now sees the environment
    conda env list
    
    # Activate the environment (the folder name is the environment name)
    conda activate transformer_stock_prediction
    ```

---

## 📈 Usage & Reproducing Results

The workflow is divided into four main steps: data preparation, model training, evaluation, and statistical testing.

### Step 1: Data Preparation

First, you need to download the raw stock data and then preprocess it to generate features.

1.  **Download Raw Data:**
    The `get_data.py` script uses the `akshare` library to download historical data for the four stocks used in the paper. The data will be saved in the `/data` directory.
    ```bash
    python -m utils.get_data
    ```

2.  **Preprocess Data:**
    The `preprocess.py` script takes the raw data, computes all technical and time-series features, and saves a unified `processed_dataset.csv` file in the `/data` directory. This file will be used for training and testing.
    ```bash
    python -m utils.preprocess --targets Apple --processors technical_indicators fourier_components
    ```

### Step 2: Model Training

The `train.py` script is used to train all models. You can specify the model architecture and target dataset via command-line arguments. To reproduce our paper's results, all 13 models must be trained.

**Example: Training our best model (`RevIN-CNN-iTransformer-BiLSTM`) on the Apple dataset:**
```bash
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use RevIN_CNN_iTransformer_BiLSTM
```

Training logs and model checkpoints will be saved to the `lightning_logs/` directory, organized by version number (e.g., `version_0`, `version_1`, etc.).

**To train all models used in the paper, you can run the commands sequentially:**

```bash
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use RevIN_CNN_iTransformer_BiLSTM
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use RevIN_CNN_Transformer_BiLSTM
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use CNN_BiLSTM_Attention
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use SCINet
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use GAN
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use Transformer
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use iTransformer
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use CNN_iTransformer_BiLSTM
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use RevIN_iTransformer_BiLSTM
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use RevIN_CNN_iTransformer
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use CNN_Transformer_BiLSTM
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use RevIN_Transformer_BiLSTM
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use RevIN_CNN_Transformer
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use xLSTM
python -m utils.train new Apple --num-days-for-predict 96 --batch-size 32 --learning-rate 0.0001 --model_use Mamba
```

### Step 3: Predict and Test

After training, use the `test.py` script to load a checkpoint and generate predictions, evaluation metrics, and result plots.

**Example: Evaluating the trained model on the Apple dataset:**

```bash
# Replace 'path/to/your/checkpoint.ckpt' with the actual path from lightning_logs
python -m utils.test lightning_logs/version_0/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer_BiLSTM --test_target Apple
```

This will generate prediction CSVs and images in the corresponding `lightning_logs/version_0/` subdirectories. Repeat this for all models and datasets to generate all necessary evaluation files.

**To reproduce all evaluation results from the paper, you must run the tests for all models across all four datasets. Preprocessing should be done before a new dataset is tested. For your convenience, we have listed all these commands here:**

```bash
python -m utils.preprocess --targets Apple --processors technical_indicators fourier_components
python -m utils.test lightning_logs/version_0/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer_BiLSTM --test_target Apple
python -m utils.test lightning_logs/version_1/checkpoints/last.ckpt  --model_use RevIN_CNN_Transformer_BiLSTM --test_target Apple
python -m utils.test lightning_logs/version_2/checkpoints/last.ckpt  --model_use CNN_BiLSTM_Attention --test_target Apple
python -m utils.test lightning_logs/version_3/checkpoints/last.ckpt  --model_use SCINet --test_target Apple
python -m utils.test lightning_logs/version_4/checkpoints/last.ckpt  --model_use GAN --test_target Apple
python -m utils.test lightning_logs/version_5/checkpoints/last.ckpt  --model_use Transformer --test_target Apple
python -m utils.test lightning_logs/version_6/checkpoints/last.ckpt  --model_use iTransformer --test_target Apple
python -m utils.test lightning_logs/version_7/checkpoints/last.ckpt  --model_use CNN_iTransformer_BiLSTM --test_target Apple
python -m utils.test lightning_logs/version_8/checkpoints/last.ckpt  --model_use RevIN_iTransformer_BiLSTM --test_target Apple
python -m utils.test lightning_logs/version_9/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer --test_target Apple
python -m utils.test lightning_logs/version_10/checkpoints/last.ckpt  --model_use CNN_Transformer_BiLSTM --test_target Apple
python -m utils.test lightning_logs/version_11/checkpoints/last.ckpt  --model_use RevIN_Transformer_BiLSTM --test_target Apple
python -m utils.test lightning_logs/version_12/checkpoints/last.ckpt  --model_use RevIN_CNN_Transformer --test_target Apple
python -m utils.test lightning_logs/version_13/checkpoints/last.ckpt  --model_use xLSTM --test_target Apple
python -m utils.test lightning_logs/version_14/checkpoints/last.ckpt  --model_use Mamba --test_target Apple
python -m utils.preprocess --targets Microsoft --processors technical_indicators fourier_components
python -m utils.test lightning_logs/version_0/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer_BiLSTM --test_target Microsoft
python -m utils.test lightning_logs/version_1/checkpoints/last.ckpt  --model_use RevIN_CNN_Transformer_BiLSTM --test_target Microsoft
python -m utils.test lightning_logs/version_2/checkpoints/last.ckpt  --model_use CNN_BiLSTM_Attention --test_target Microsoft
python -m utils.test lightning_logs/version_3/checkpoints/last.ckpt  --model_use SCINet --test_target Microsoft
python -m utils.test lightning_logs/version_4/checkpoints/last.ckpt  --model_use GAN --test_target Microsoft
python -m utils.test lightning_logs/version_5/checkpoints/last.ckpt  --model_use Transformer --test_target Microsoft
python -m utils.test lightning_logs/version_6/checkpoints/last.ckpt  --model_use iTransformer --test_target Microsoft
python -m utils.test lightning_logs/version_7/checkpoints/last.ckpt  --model_use CNN_iTransformer_BiLSTM --test_target Microsoft
python -m utils.test lightning_logs/version_8/checkpoints/last.ckpt  --model_use RevIN_iTransformer_BiLSTM --test_target Microsoft
python -m utils.test lightning_logs/version_9/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer --test_target Microsoft
python -m utils.test lightning_logs/version_10/checkpoints/last.ckpt  --model_use CNN_Transformer_BiLSTM --test_target Microsoft
python -m utils.test lightning_logs/version_11/checkpoints/last.ckpt  --model_use RevIN_Transformer_BiLSTM --test_target Microsoft
python -m utils.test lightning_logs/version_12/checkpoints/last.ckpt  --model_use RevIN_CNN_Transformer --test_target Microsoft
python -m utils.test lightning_logs/version_13/checkpoints/last.ckpt  --model_use xLSTM --test_target Microsoft
python -m utils.test lightning_logs/version_14/checkpoints/last.ckpt  --model_use Mamba --test_target Microsoft
python -m utils.preprocess --targets MaoTai --processors technical_indicators fourier_components
python -m utils.test lightning_logs/version_0/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer_BiLSTM --test_target MaoTai
python -m utils.test lightning_logs/version_1/checkpoints/last.ckpt  --model_use RevIN_CNN_Transformer_BiLSTM --test_target MaoTai
python -m utils.test lightning_logs/version_2/checkpoints/last.ckpt  --model_use CNN_BiLSTM_Attention --test_target MaoTai
python -m utils.test lightning_logs/version_3/checkpoints/last.ckpt  --model_use SCINet --test_target MaoTai
python -m utils.test lightning_logs/version_4/checkpoints/last.ckpt  --model_use GAN --test_target MaoTai
python -m utils.test lightning_logs/version_5/checkpoints/last.ckpt  --model_use Transformer --test_target MaoTai
python -m utils.test lightning_logs/version_6/checkpoints/last.ckpt  --model_use iTransformer --test_target MaoTai
python -m utils.test lightning_logs/version_7/checkpoints/last.ckpt  --model_use CNN_iTransformer_BiLSTM --test_target MaoTai
python -m utils.test lightning_logs/version_8/checkpoints/last.ckpt  --model_use RevIN_iTransformer_BiLSTM --test_target MaoTai
python -m utils.test lightning_logs/version_9/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer --test_target MaoTai
python -m utils.test lightning_logs/version_10/checkpoints/last.ckpt  --model_use CNN_Transformer_BiLSTM --test_target MaoTai
python -m utils.test lightning_logs/version_11/checkpoints/last.ckpt  --model_use RevIN_Transformer_BiLSTM --test_target MaoTai
python -m utils.test lightning_logs/version_12/checkpoints/last.ckpt  --model_use RevIN_CNN_Transformer --test_target MaoTai
python -m utils.test lightning_logs/version_13/checkpoints/last.ckpt  --model_use xLSTM --test_target MaoTai
python -m utils.test lightning_logs/version_14/checkpoints/last.ckpt  --model_use Mamba --test_target MaoTai
python -m utils.preprocess --targets HSBC --processors technical_indicators fourier_components
python -m utils.test lightning_logs/version_0/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer_BiLSTM --test_target HSBC
python -m utils.test lightning_logs/version_1/checkpoints/last.ckpt  --model_use RevIN_CNN_Transformer_BiLSTM --test_target HSBC
python -m utils.test lightning_logs/version_2/checkpoints/last.ckpt  --model_use CNN_BiLSTM_Attention --test_target HSBC
python -m utils.test lightning_logs/version_3/checkpoints/last.ckpt  --model_use SCINet --test_target HSBC
python -m utils.test lightning_logs/version_4/checkpoints/last.ckpt  --model_use GAN --test_target HSBC
python -m utils.test lightning_logs/version_5/checkpoints/last.ckpt  --model_use Transformer --test_target HSBC
python -m utils.test lightning_logs/version_6/checkpoints/last.ckpt  --model_use iTransformer --test_target HSBC
python -m utils.test lightning_logs/version_7/checkpoints/last.ckpt  --model_use CNN_iTransformer_BiLSTM --test_target HSBC
python -m utils.test lightning_logs/version_8/checkpoints/last.ckpt  --model_use RevIN_iTransformer_BiLSTM --test_target HSBC
python -m utils.test lightning_logs/version_9/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer --test_target HSBC
python -m utils.test lightning_logs/version_10/checkpoints/last.ckpt  --model_use CNN_Transformer_BiLSTM --test_target HSBC
python -m utils.test lightning_logs/version_11/checkpoints/last.ckpt  --model_use RevIN_Transformer_BiLSTM --test_target HSBC
python -m utils.test lightning_logs/version_12/checkpoints/last.ckpt  --model_use RevIN_CNN_Transformer --test_target HSBC
python -m utils.test lightning_logs/version_13/checkpoints/last.ckpt  --model_use xLSTM --test_target HSBC
python -m utils.test lightning_logs/version_14/checkpoints/last.ckpt  --model_use Mamba --test_target HSBC
```

### Step 4: Aggregate Results and Statistical Tests

Once all models have been evaluated, you can aggregate the results and perform statistical tests.

1. **Aggregate Evaluation Metrics:**
   The `evaluate.py` script collects all individual `_Evaluation.csv` files and creates summary tables.

   ```bash
   python -m utils.evaluate
   ```

   This will generate summary files like `Apple_Summary.csv` in the `/evaluate_results` folder.

2. **Perform Diebold-Mariano Test:**
   The `dm_test.py` script performs statistical tests to verify if the performance improvement of our model is significant.

   **Example: Comparing our best model against baselines on the Apple test set:**

   ```bash
   python -m utils.dm_test --test_set Apple --reference_model "RevIN-CNN-iTransformer-BiLSTM" --comparison_models "RevIN-CNN-Transformer-BiLSTM" "CNN-BiLSTM-Attention" "SCINet" "GAN" "Transformer" "iTransformer" "xLSTM" "Mamba" --crit MSE --subset Test
   ```

   The results will be printed to the console and saved to the `/DM_Test_results` folder.

   **To perform all Diebold-Mariano Tests used in the paper, these commands are necessary:**

   ```bash
   python -m utils.dm_test --test_set Apple --reference_model "RevIN-CNN-iTransformer-BiLSTM" --comparison_models "RevIN-CNN-Transformer-BiLSTM" "CNN-BiLSTM-Attention" "SCINet" "GAN" "Transformer" "iTransformer" "xLSTM" "Mamba" --crit MSE --subset Test
   python -m utils.dm_test --test_set Microsoft --reference_model "RevIN-CNN-iTransformer-BiLSTM" --comparison_models "RevIN-CNN-Transformer-BiLSTM" "CNN-BiLSTM-Attention" "SCINet" "GAN" "Transformer" "iTransformer" "xLSTM" "Mamba" --crit MSE --subset All
   python -m utils.dm_test --test_set MaoTai --reference_model "RevIN-CNN-iTransformer-BiLSTM" --comparison_models "RevIN-CNN-Transformer-BiLSTM" "CNN-BiLSTM-Attention" "SCINet" "GAN" "Transformer" "iTransformer" "xLSTM" "Mamba" --crit MSE --subset All
   python -m utils.dm_test --test_set HSBC --reference_model "RevIN-CNN-iTransformer-BiLSTM" --comparison_models "RevIN-CNN-Transformer-BiLSTM" "CNN-BiLSTM-Attention" "SCINet" "GAN" "Transformer" "iTransformer" "xLSTM" "Mamba" --crit MSE --subset All
   ```

3. **Generate Comparison Plots 🎨:**
   The `draw_figure.py` script is provided to visualize and compare the predictions of multiple models against the true values on a single plot. This is useful for creating the final figures for the paper.

   **Example: Generating the comparison plot for the Apple test set within a specific date range:**
   ```bash
   python -m utils.draw_figure --test_target Apple --true_csv_path data/Apple.csv \
     --pred_csv_paths lightning_logs/version_0/test_prediction/Apple_test_pred.csv lightning_logs/version_1/test_prediction/Apple_test_pred.csv lightning_logs/version_2/test_prediction/Apple_test_pred.csv lightning_logs/version_3/test_prediction/Apple_test_pred.csv lightning_logs/version_4/test_prediction/Apple_test_pred.csv lightning_logs/version_5/test_prediction/Apple_test_pred.csv     lightning_logs/version_6/test_prediction/Apple_test_pred.csv  lightning_logs/version_13/test_prediction/Apple_test_pred.csv lightning_logs/version_14/test_prediction/Apple_test_pred.csv \
     --model_names RevIN_CNN_iTransformer_BiLSTM RevIN_CNN_Transformer_BiLSTM CNN_BiLSTM_Attention SCINet GRU_GAN Transformer iTransformer xLSTM Mamba \
     --date_range 2024-01-17 2025-01-01
   ```

   **To generate all four comparison plots used in the paper, you can use the provided commands:**
   ```bash
   python -m utils.draw_figure --test_target Apple --true_csv_path data/Apple.csv --pred_csv_paths lightning_logs/version_0/test_prediction/Apple_test_pred.csv lightning_logs/version_1/test_prediction/Apple_test_pred.csv lightning_logs/version_2/test_prediction/Apple_test_pred.csv lightning_logs/version_3/test_prediction/Apple_test_pred.csv lightning_logs/version_4/test_prediction/Apple_test_pred.csv lightning_logs/version_5/test_prediction/Apple_test_pred.csv lightning_logs/version_6/test_prediction/Apple_test_pred.csv lightning_logs/version_13/test_prediction/Apple_test_pred.csv lightning_logs/version_14/test_prediction/Apple_test_pred.csv --model_names RevIN_CNN_iTransformer_BiLSTM RevIN_CNN_Transformer_BiLSTM CNN_BiLSTM_Attention SCINet GRU_GAN Transformer iTransformer xLSTM Mamba --date_range 2024-01-17 2025-01-01
   python -m utils.draw_figure --test_target Microsoft --true_csv_path data/Microsoft.csv --pred_csv_paths lightning_logs/version_0/test_prediction/Microsoft_pred.csv lightning_logs/version_1/test_prediction/Microsoft_pred.csv lightning_logs/version_2/test_prediction/Microsoft_pred.csv lightning_logs/version_3/test_prediction/Microsoft_pred.csv lightning_logs/version_4/test_prediction/Microsoft_pred.csv lightning_logs/version_5/test_prediction/Microsoft_pred.csv lightning_logs/version_6/test_prediction/Microsoft_pred.csv lightning_logs/version_13/test_prediction/Microsoft_test_pred.csv lightning_logs/version_14/test_prediction/Microsoft_test_pred.csv --model_names RevIN_CNN_iTransformer_BiLSTM RevIN_CNN_Transformer_BiLSTM CNN_BiLSTM_Attention SCINet GRU_GAN Transformer iTransformer xLSTM Mamba
   python -m utils.draw_figure --test_target MaoTai --true_csv_path data/MaoTai.csv --pred_csv_paths lightning_logs/version_0/test_prediction/MaoTai_pred.csv lightning_logs/version_1/test_prediction/MaoTai_pred.csv lightning_logs/version_2/test_prediction/MaoTai_pred.csv lightning_logs/version_3/test_prediction/MaoTai_pred.csv lightning_logs/version_4/test_prediction/MaoTai_pred.csv lightning_logs/version_5/test_prediction/MaoTai_pred.csv lightning_logs/version_6/test_prediction/MaoTai_pred.csv lightning_logs/version_13/test_prediction/MaoTai_test_pred.csv lightning_logs/version_14/test_prediction/MaoTai_test_pred.csv --model_names RevIN_CNN_iTransformer_BiLSTM RevIN_CNN_Transformer_BiLSTM CNN_BiLSTM_Attention SCINet GRU_GAN Transformer iTransformer xLSTM Mamba
   python -m utils.draw_figure --test_target HSBC --true_csv_path data/HSBC.csv --pred_csv_paths lightning_logs/version_0/test_prediction/HSBC_pred.csv lightning_logs/version_1/test_prediction/HSBC_pred.csv lightning_logs/version_2/test_prediction/HSBC_pred.csv lightning_logs/version_3/test_prediction/HSBC_pred.csv lightning_logs/version_4/test_prediction/HSBC_pred.csv lightning_logs/version_5/test_prediction/HSBC_pred.csv lightning_logs/version_6/test_prediction/HSBC_pred.csv lightning_logs/version_13/test_prediction/HSBC_test_pred.csv lightning_logs/version_14/test_prediction/HSBC_test_pred.csv --model_names RevIN_CNN_iTransformer_BiLSTM RevIN_CNN_Transformer_BiLSTM CNN_BiLSTM_Attention SCINet GRU_GAN Transformer iTransformer xLSTM Mamba
   ```

   **To generate loss comparison plots used in the paper, you can use the provided commands:**
   ```bash
   python -m utils.plot_loss_curves --csv-paths lightning_logs/version_0/metrics.csv lightning_logs/version_1/metrics.csv lightning_logs/version_5/metrics.csv lightning_logs/version_6/metrics.csv --model-names RevIN-CNN-iTransformer-BiLSTM RevIN-CNN-Transformer-BiLSTM Transformer iTransformer --output-folder loss_curve_results/ --output-basename loss_comparison --y-log
   ```

4. **Feature Importance and Model Interpretability with SHAP:**

   To understand *why* the model makes specific predictions, we provide a SHAP (SHapley Additive exPlanations) analysis script.

   ```bash
   python -m utils.preprocess --targets Apple --processors technical_indicators fourier_components
   python -m utils.shap_analysis --model-type RevIN_CNN_iTransformer_BiLSTM --ckpt-path lightning_logs/version_0/checkpoints/last.ckpt --data-path data/processed_dataset.csv --output-folder shap_analysis/RevIN_CNN_iTransformer_BiLSTM/ --num-background-samples 200 --num-test-samples 500 --max-display 10
   python -m utils.shap_analysis --model-type iTransformer --ckpt-path lightning_logs/version_6/checkpoints/last.ckpt --data-path data/processed_dataset.csv --output-folder shap_analysis/iTransformer/ --num-background-samples 200 --num-test-samples 500 --max-display 10
   ```

5. **Uncertainty Quantification (Prediction Interval Evaluation):**

   Beyond point forecasting, our framework supports **Probabilistic Forecasting** using Pinball Loss to generate 95% confidence intervals.

   ```bash
   python -m utils.PinballLoss_test_and_draw_figure --median_ckpt_paths lightning_logs/version_17/checkpoints/last.ckpt lightning_logs/version_20/checkpoints/last.ckpt --lower_ckpt_paths lightning_logs/version_15/checkpoints/last.ckpt lightning_logs/version_18/checkpoints/last.ckpt --upper_ckpt_paths lightning_logs/version_16/checkpoints/last.ckpt lightning_logs/version_19/checkpoints/last.ckpt --model_classes RevIN_CNN_iTransformer_BiLSTM_PinballLoss iTransformer_PinballLoss --model_names RevIN_CNN_iTransformer_BiLSTM iTransformer_PinballLoss --test_target Apple --output_dir PinballLoss_results/RevIN_CNN_iTransformer_BiLSTM --data_csv_path data/processed_dataset.csv --date_range 2024-01-17 2025-01-01
   ```

6. **Day-Ahead Rolling Forecast 🔄:**

   To simulate a real-world trading scenario, we perform a rolling forecast on the latest 360 trading days without retraining (Zero-shot generalization):

   ```bash
   python -m utils.preprocess --targets Apple --processors technical_indicators fourier_components --day-ahead 360 --input-horizon 96
   python -m utils.test lightning_logs/version_0/checkpoints/last.ckpt  --model_use RevIN_CNN_iTransformer_BiLSTM --test_target Apple --mode day-ahead
   python -m utils.test lightning_logs/version_6/checkpoints/last.ckpt  --model_use iTransformer --test_target Apple --mode day-ahead
   python -m utils.draw_figure --test_target Apple --true_csv_path data/Apple.csv --pred_csv_paths lightning_logs/version_0/day-ahead_prediction/Apple_pred.csv lightning_logs/version_6/day-ahead_prediction/Apple_pred.csv --model_names RevIN_CNN_iTransformer_BiLSTM iTransformer --date_range 2024-05-14 2025-10-20 --fig_title iTransformer --day-ahead
   
   ```

---

## How to Cite

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{HUANG2026131572,
title = {Enhancing stock price forecasting with a modular deep learning framework incorporating plug-and-play transformer variants},
journal = {Expert Systems with Applications},
volume = {315},
pages = {131572},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2026.131572},
url = {https://www.sciencedirect.com/science/article/pii/S0957417426004859},
author = {Yuming Huang and Chenxue Yang},
keywords = {Stock price forecasting, Modular deep learning, Plug-and-play transformer variants, Reversible instance normalization (RevIN), Non-stationary data, Cross-market generalization},
abstract = {Stock price forecasting remains challenging due to highly nonstationary, noisy financial time-series punctuated by rare extreme events with persistent impacts. Although recent deep-learning models, especially CNNs, RNNs, and Transformers, have improved predictive accuracy, they still face significant limitations regarding distributional drift, tail-risk robustness, and computational demands for long sequences. To bridge these gaps, we propose a modular, knowledge-driven deep learning framework integrating Reversible Instance Normalization (RevIN), Convolutional Neural Networks (CNN), plug-and-play Transformer variants, and Bidirectional LSTM (BiLSTM). RevIN acts as an expert-driven decision-support module, dynamically normalizing data to mitigate regime shifts while maintaining outputs in economically meaningful scales. CNN captures local temporal micro-patterns, Transformer variants (standard Transformer, iTransformer, etc.) handle global temporal and cross-variable dependencies, and BiLSTM fuses forward-backward temporal dynamics. The fully modular, plug-and-play architecture enables effortless incorporation of emerging Transformer variants without retraining existing components, ensuring long-term adaptability and scalability. Empirical evaluations on four cross-market datasets (NASDAQ-AAPL, NASDAQ-MSFT, SSE-600519, and HKEX-00005) confirm that our framework significantly lowers RMSE and MAPE by up to 50% and 54%, respectively, compared with five strong baselines. These performance gains are achieved with only a moderate (about 22%) increase in GPU training time and comparable inference latency to lightweight models. Cross-market tests further demonstrate robust generalization across distinct market regimes and currencies, significantly outperforming baseline models even without retraining. Ablation studies validate the complementary contribution of each module, underscoring the effectiveness of integrating expert-driven normalization and modular deep-learning components for robust financial decision-support systems. The code and data used in this study are available at: https://github.com/DyeDye-diyung/Framework_for_Financial_Time-Series_Forecasting.}
}
```

-----

## 📄 License

This project is licensed under the Apache-2.0 License. See the `LICENSE` file for details.
