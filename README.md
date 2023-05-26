# Arabic QA Model using Fine-Tuning DistilBERT
This repository contains code for a Question-Answering (QA) model based on fine-tuning DistilBERT using Arabic QA data. We use the Arabert tokenizer for the tokenization part.

## Overview

The goal of this project is to build a QA model that can accurately answer questions posed in Arabic. We fine-tune the DistilBERT model, a smaller and faster version of the popular BERT model, using Arabic QA data to create a model that can understand the nuances of Arabic language and provide accurate answers.

We use the Arabert tokenizer, which is specifically designed for Arabic text, to preprocess the data before feeding it into the model. This tokenizer is based on WordPiece, a subword tokenization algorithm that splits words into smaller units based on frequency and context.

To run the code simply install the required libraries and run the code. 
The code was mainly run on colab so you need to modify the path to point at the FINAL_AAQAD-v1.0.json on colab
The main notebook is QA. The other notebooks are used for Data analysis only.
