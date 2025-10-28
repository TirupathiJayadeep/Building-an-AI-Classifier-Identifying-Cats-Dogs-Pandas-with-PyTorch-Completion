# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch-Completion
A deep learning project using Transfer Learning with PyTorch to classify images of Cats 🐱, Dogs 🐶, and Pandas 🐼.
This project demonstrates efficient model fine-tuning and GPU acceleration for high-accuracy image classification.

## Overview

This notebook (DL_Classification_project.ipynb) showcases:

Transfer Learning using a pre-trained ResNet model

Data preprocessing & augmentation with torchvision.transforms

Training, validation, and evaluation with loss/accuracy visualization

CUDA acceleration for faster computation

Support for both local and Kaggle environments

## Setup Instructions
1. Create a New GitHub Repository

Go to GitHub → New Repository

Name it something like DL_Classification_Project

Add a .gitignore for Python and optionally a license

After creation, open a terminal and run:
``
git init
git remote add origin https://github.com/<your-username>/DL_Classification_Project.git
``

2. Add Project Files

Upload the following:

DL_Classification_project.ipynb

README.md

Optionally, include:

/dataset/
    ├── train/
    ├── test/
    └── valid/

3. Install Dependencies

Make sure PyTorch and other required libraries are installed:
```
pip install torch torchvision torchaudio matplotlib numpy pandas tqdm
```
4. Open the Notebook

Launch Jupyter Notebook and open the file:

jupyter notebook DL_Classification_project.ipynb



You can use datasets from Kaggle such as:
👉 Cats and Dogs and Pandas Dataset

After downloading, extract and arrange it as:

dataset/
├── train/
├── test/
└── valid/

⚡ CUDA Check

Before training, verify GPU availability:

import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


If CUDA is unavailable, the code automatically switches to CPU mode.

🧾 Running on Kaggle

Upload the notebook to your Kaggle account.

Enable GPU:

Go to Settings → Accelerator → GPU (P100 or T4).

Add the dataset via Add Data → Search “Cats and Dogs and Pandas”.

Set the path in code:

data_dir = "/kaggle/input/cats-and-dogs-and-pandas/"


Run all cells — training, evaluation, and plots will execute automatically.
