# Brain Tumor Classification using Freeze-5 VGG16

MRI brain tumor classification using a freeze-5 fine-tuned VGG16 model with dataset fusion (Kaggle + BraTS 2020). Includes preprocessing, augmentation, 5-fold cross-validation, and benchmarking. Achieves 99.16% accuracy and robust, reproducible results.

## Features

- Transfer learning with VGG16 (freeze first 5 convolutional layers)
- Dataset fusion from Kaggle and BraTS 2020
- Image preprocessing: resizing, normalization, grayscale to RGB
- Data augmentation: rotation, translation, zoom, brightness adjustment
- 5-fold cross-validation for robust evaluation
- Performance metrics: accuracy, precision, recall, F1-score, confusion matrix

## Installation

1. Clone this repository

   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name

   ```

2. Create and activate virtual environment (optional but recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

   ```

3. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Prepare dataset according to instructions in data/README.md
- Run training script:

  ```python train.py

  ```

- Evaluate model with cross-validation and generate reports

## License

This project is licensed under the MIT License

## Contact

For questions or collaborations, please contact:
Vicky
Email: vicky@students.mikroskil.ac.id
LinkedIn: https://www.linkedin.com/in/vicky-profile
