# Titanic - EDA
Exploratory Data Analysis (EDA) on the Titanic dataset

## Overview

This project is focused on performing **Exploratory Data Analysis (EDA)** on the **Titanic dataset**. The primary goal of the project is to explore, clean, visualize, and analyze the data in order to understand the underlying patterns. This analysis helps in building machine learning models that predict survival chances of passengers aboard the Titanic.

Key steps in the project:
- **Data Loading**: Load the Titanic dataset and inspect the data.
- **Data Cleaning**: Handle missing values, encode categorical variables, and normalize numerical values.
- **Exploratory Data Analysis**: Visualize the relationships between features and survival, and identify patterns in the data.
- **Modeling**: Train machine learning models to predict passenger survival and evaluate model performance.
---

## Project Structure

The project is organized as follows:

titanic-eda/
├── data/ # Contains the dataset and prediction output files
│ ├── train.csv # The main training dataset (downloaded from Kaggle)
│ ├── test.csv # The test dataset (downloaded from Kaggle)
│ ├── gender_submission.csv # A sample submission file for Kaggle (downloaded from Kaggle)
│ └── titanic_predictions.csv # The predicted survival outcomes (generated after running the model)
├── venv/ # Virtual environment folder
├── eda.py # Main Python script for EDA
├── titanic_eda.ipynb # Jupyter notebook for exploratory analysis
├── requirements.txt # List of required Python packages
├── README.md # Project documentation (this file)
└── .gitignore # Git ignore file

---

## Data Files

The following datasets are used in this project:

- **train.csv**: The primary dataset used for training the model. It contains information about passengers (e.g., age, sex, passenger class, etc.) and whether they survived.
- **test.csv**: The dataset used for testing model predictions. It does not include the target variable (Survived).
- **gender_submission.csv**: A sample submission file used for Kaggle competitions.
- **titanic_predictions.csv**: This file is generated after running the model. It contains the predicted survival outcomes for the `test.csv` dataset.

---

## Requirements

The project requires the following Python libraries:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- jupyter

To install the required packages, use the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Setting Up the Virtual Environment

### 1. Create a Virtual Environment

To create a virtual environment, run the following command:
```bash
python -m venv venv

```
### 2. Activate the Virtual Environment

- On **Windows**:
```bash
.\venv\Scripts\activate

- On **macOS/Linux**:

source venv/bin/activate

```

### 3. Install Dependencies

Once the virtual environment is activated, install the required packages:
```bash
pip install -r requirements.txt

```
## Running the Project

### 1. Running the EDA Script (`eda.py`)

To perform the exploratory data analysis, you can run the `eda.py` script. This script includes:

- Data loading and inspection
- Data cleaning and preprocessing
- Visualization of data distributions
- Model training and evaluation

You can run the script using:
```bash
python eda.py

```
### 2. Running the Jupyter Notebook (`titanic_eda.ipynb`)

Alternatively, you can open the Jupyter notebook for an interactive analysis:
```bash
jupyter notebook titanic_eda.ipynb

```
This allows you to run the code step-by-step and visualize the data interactively.

---

## Model and Evaluation

The project includes a **Random Forest Classifier** model, which is used to predict passenger survival based on the features provided in the dataset. The model is trained using the `train.csv` data and evaluated on the `test.csv` data.

Key evaluation metrics include:

- **Accuracy**
- **Confusion Matrix**
- **Classification Report**

The model also uses **GridSearchCV** to perform hyperparameter tuning.

---

## Author

- **Elen Tesfai**  
- GitHub: [https://github.com/Elen-tesfai/titanic-eda](https://github.com/Elen-tesfai/titanic-eda)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

