# Predict Bike Sharing Demand with AutoGluon

## Introduction to AWS Machine Learning Final Project

This project aims to apply the knowledge and techniques acquired in the Introduction to Machine Learning course to compete in a Kaggle competition utilizing the AutoGluon library.

## Overview
In this project, participants will:
- Create a Kaggle account if they don't already have one.
- Download the Bike Sharing Demand dataset.
- Train a model using AutoGluon and submit initial results for ranking.
- Iterate on the process by enhancing the dataset with additional features and tuning hyperparameters to improve the score.
- Submit all work and write a report detailing the methods that provided the best score improvement and the reasoning behind them. A template of the report can be found [here](report-template.md).

To meet project requirements, the following files are necessary:
- Jupyter notebook with code executed to completion
- HTML export of the Jupyter notebook
- Markdown or PDF file of the report

Additional images or files required to complete the notebook or report can also be included.

## Getting Started
- Clone this template repository `git clone git@github.com:udacity/nd009t-c1-intro-to-ml-project-starter.git` into AWS Sagemaker Studio (or a local development environment).
- Proceed with the project within the [Jupyter notebook](project-template.ipynb).
- Visit the [Kaggle Bike Sharing Demand Competition](https://www.kaggle.com/c/bike-sharing-demand) page for competition details, including overview, data, code, discussion, leaderboard, and rules.

### Dependencies
```plaintext
Python 3.7
MXNet 1.8
Pandas >= 1.2.4
AutoGluon 0.2.0 
```

### Installation
For this project, it is recommended to use Sagemaker Studio from the course-provided AWS workspace, simplifying the necessary installation steps.

For local development:
- Setup a Jupyter Lab instance:
  - Follow the [Jupyter installation](https://jupyter.org/install.html) guide for best practices.
  - If a Python virtual environment is already installed, simply use `pip` to install it:
    ```bash
    pip install jupyterlab
    ```
  - Alternatively, use Docker containers containing Jupyter Lab from [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html).

## Project Instructions
1. Create an account with Kaggle.
2. Download the Kaggle dataset using the Kaggle Python library.
3. Train a model using AutoGluon’s Tabular Prediction and submit predictions to Kaggle for ranking.
4. Use Pandas to conduct exploratory analysis, create new features, and save new versions of the train and test datasets.
5. Rerun the model and submit the new predictions for ranking.
6. Tune at least 3 different hyperparameters from AutoGluon and resubmit predictions to achieve a higher ranking on Kaggle.
7. Write a report detailing improvements (or lack thereof) made by either creating additional features or tuning hyperparameters, and why one approach may be more effective than the other.

## License
[License](LICENSE.txt)