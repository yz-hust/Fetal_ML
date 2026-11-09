# Explainable Machine Learning for Prenatal Detection of Coarctation of the Aorta With Ventricular Septal Defect Detection

This directory contains two trained random forest models, an example Excel dataset, and a batch inference script. The script outputs a predicted class and class probabilities for each sample.

## Models

- Model_1 is the original parameter model.
- Model_2 is the new-parameter model and is used by default.

Class labels:

- 0: VSD
- 1: CoA/VSD

## Installation

Python 3.12 is recommended.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-inference.txt

On Windows PowerShell, activate the environment with:

    .venv\Scripts\Activate.ps1

## Usage

Run inference with the default Model_2:

    python Fetal_ML_inference.py

Run inference with Model_1:

    python Fetal_ML_inference.py --model-version Model_1

Specify custom input, model, and output files:

    python Fetal_ML_inference.py --input test.xlsx --model Model_1.pkl --output predictions.csv

An explicit --model path takes precedence over --model-version. Run the following command to view all options:

    python Fetal_ML_inference.py --help

By default, the script reads test.xlsx and writes inference_predictions.csv in the script directory. It can therefore be launched from any working directory.

## Input Data

The input must be an Excel file. Feature columns must be numeric and cannot contain missing or infinite values.

| Model | Required features |
| --- | --- |
| Model_1 | MPA/TAA, LVLS, AoI Z score |
| Model_2 | MPA/TAA, LVLS, AoI/VSD |

The optional CoA/VSD Group column is included as the true label in the output but is not passed to the model. Additional columns are ignored.

## Output

The UTF-8-SIG CSV output contains the sample number, optional true label, predicted numeric class, predicted class name, and probability for each class.

## Security

Pickle files can execute code when loaded. Only use model files from trusted sources.
