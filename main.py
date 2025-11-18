import torch
import transformers
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from transformers import DistilBertModel, Trainer, TrainingArguments

from src.load_data import load_data
from src.train_test_split import train_test_split
from src.baseline import baseline, plot_baseline
from src.tokenizer import tokenizer_processing
from src.dataset import MyDataset
from src.model import DistilBertForSentimentClassification
from src.training import compute_metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open("config.yaml", "r") as file:
        data = yaml.safe_load(file)

    filename = data["filename"]  
    classes = data["classes"]
    model_name = data["model"]
    freeze_pretrained_model = data["freeze_pretrained_model"]
    
    df_data = load_data(filename, classes)
    
    train_test_split(df_data)
    print(df_data['Split'].value_counts())
    
    baseline_f1_train, baseline_f1_test, baseline_accuracy = baseline(df= df_data, max_features= 100, model= 'bn')
    print(f"Baseline model:\n F1_train: {baseline_f1_train}, F1_test: {baseline_f1_test}, Accuracy: {baseline_accuracy}")
    plot_baseline(df_data)
    
    # Correggi la selezione dei dati
    x_train = df_data[df_data.Split=='Trai']['sentence'].values
    y_train = df_data[df_data.Split=='Trai']['label'].values
    x_test = df_data[df_data.Split=='Test']['sentence'].values
    y_test = df_data[df_data.Split=='Test']['label'].values
    
    tokenizer = tokenizer_processing(model= model_name, df_data= df_data)
  
    # Mappatura string -> int
    label_mapping = {'neutral': 0, 'positive': 1, 'negative': 2}

    # Converti le label
    y_train_num = [label_mapping[label] for label in y_train]
    y_test_num = [label_mapping[label] for label in y_test]

        
    # Instanzia i dataset
    train_dataset = MyDataset(x_train, y_train_num, tokenizer)
    test_dataset = MyDataset(x_test, y_test_num, tokenizer)

    # Controllo rapido
    print(f"Lenght train set: {len(train_dataset)}, Lenght test set: {len(test_dataset)}")
    
    # instantiate model
    model = DistilBertForSentimentClassification(
        config= DistilBertModel.from_pretrained(model_name).config,
        num_labels=len(classes),
        freeze_encoder = freeze_pretrained_model, model_name= model_name
        )

    # print info about model's parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print('model total params: ', total_params)
    print('model trainable params: ', trainable_params)
    print('\n', model)
    
    training_args = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',
    logging_first_step=True,
    logging_steps=50,
    num_train_epochs=16,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01)
    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics)
    
    train_results = trainer.train()
    test_results = trainer.predict(test_dataset=test_dataset)
    
    print('Predictions: \n', test_results.predictions)
    print('\nAccuracy: ', test_results.metrics['test_accuracy'])
    print('Precision: ', test_results.metrics['test_precision'])
    print('Recall: ', test_results.metrics['test_recall'])
    print(classes)


if __name__ == "__main__":
    print("Started main.py")
    main()