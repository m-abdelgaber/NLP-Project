
# Instructions to run this code for evaluation
The code was mainly run on Colab, so you need to modify the path to point at the FINAL_AAQAD-v1.0.json on Colab. You should download the FINAL_AAQAD-v1.0.json from [here](https://github.com/adelmeleka/AQAD/blob/master/AQQAD%201.0/FINAL_AAQAD-v1.0.json).

# Arabic Question-Answering DistilBERT Model
This repository contains code for a Question-Answering (QA) model based on fine-tuning DistilBERT using Arabic QA data. We compare between 2 tokenizers, the Arabert tokenizer, and the DistilBERT tokenizer. The goal of this project is to build a QA model that can accurately answer questions posed in Arabic. We fine-tune the DistilBERT model, a smaller and faster version of the popular BERT model, using Arabic QA data to create a model that can understand the nuances of the Arabic language and provide accurate answers.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)

## Installation

To use this code, you need to have the following dependencies installed:

- ijson
- arabert
- transformers
- pandas
- matplotlib
- numpy
- torch
- DistilBertTokenizerFast (from transformers)
- DistilBertForQuestionAnswering (from transformers)
- DataLoader (from torch.utils.data)
- AdamW (from transformers)
- tqdm
- sklearn
- nltk
- seaborn
- gensim

You can install these dependencies by running the following command:

```shell
pip install -r requirements.txt
```

Additionally, you need to download the stopwords corpus from NLTK. You can do this by running the following Python code:

```shell
import nltk
nltk.download('stopwords')
```

Before running the code, make sure to clone the repository to get the training data:

```shell
git clone https://github.com/adelmeleka/AQAD
```

## Usage

To use the question-answering model, follow these steps:

1. Clone this repository:

   ```shell
   git clone https://github.com/m-abdelgaber/NLP-Project
   ```

2. Navigate to the repository directory:

   ```shell
   cd NLP-Project
   ```

3. Open the Jupyter Notebook:

   ```shell
   jupyter notebook
   ```

4. Open the `QA.ipynb` notebook and run the cells sequentially to train the model and evaluate its performance.

## Data Preparation

The training and evaluation datasets are expected to be in a JSON format similar to the [AQAD dataset file](https://github.com/adelmeleka/AQAD/blob/master/AQQAD%201.0/FINAL_AAQAD-v1.0.json).

Before training the model, make sure to prepare your data in the required format and specify the path to the JSON file in the notebook.

## Training

The training process involves the following steps:

1. Loading the data from the JSON file and preprocessing it.
2. Splitting the data into training and validation sets.
3. Tokenizing the data using the pre-trained tokenizer.
4. Setting up the model architecture and optimizer.
5. Training the model using the training dataset.
6. Evaluating the model's performance on the validation dataset.

## Evaluation

The evaluation process involves the following steps:

1. Loading the pre-trained model and tokenizer.
2. Defining utility functions for text normalization and computing evaluation metrics (e.g., EM and F1 scores).
3. Calculating the EM and F1 scores for the validation dataset.
4. Analyzing the performance of the model on different subsets of the data (e.g., short, medium, and long contexts).
5. Saving the evaluation results to an Excel file for further analysis.

## Results

The evaluation results include metrics such as EM and F1 scores, average predicted answer length, and performance on different subsets of the data. These results provide insights into the model's effectiveness and can be used to compare different configurations or models.

## Conclusion

This code provides a framework for training and evaluating a question-answering model using a pre-trained DistilBERT model and tokenizer. It allows you to customize the training and evaluation process based on your specific dataset and requirements. By using this code, you can build and deploy a question-answering system for a wide range of applications.

## Contributing

Contributions to this project are welcome. You can contribute by submitting bug reports, suggesting new features, or proposing improvements to the existing code.
