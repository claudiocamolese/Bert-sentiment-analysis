# Bert-Sentiment-Analysis

Project to train and evaluate a **DistilBERT-based model** for sentiment classification.  
The repository includes a custom DistilBERT classification head, a PyTorch Dataset wrapper, a TF‑IDF baseline and a HuggingFace Trainer-based training flow.

This project demonstrates **fine-tuning a pretrained Transformer model** on a custom sentiment analysis dataset.

---

## About BERT and Fine-Tuning
<img width="1384" height="548" alt="1_5cQlEV_7WuzUfE1B__jR5Q" src="https://github.com/user-attachments/assets/661ab6a2-e891-4834-a27e-4ee10b5d3c65" />

**BERT (Bidirectional Encoder Representations from Transformers)** is a Transformer-based language model that learns contextual representations of text using a bidirectional attention mechanism. Pretrained on large corpora, BERT captures semantic relationships and can be adapted to various NLP tasks.

**DistilBERT** is a smaller, faster, and lighter version of BERT with ~40% fewer parameters but retaining most of the original performance, making it ideal for lightweight applications.

**Fine-tuning** refers to adapting a pretrained model to a specific downstream task (here, sentiment classification). Instead of training from scratch, we attach a task-specific head (e.g., a linear classifier) to the pretrained encoder and train on labeled data, often achieving high performance with limited task-specific data.

---

## How to run
If you want to track your training, build a key of W&B and paste it during colab training cell.
Simply go to colab, and run

```python
!python main.py
```
and past the key in the input cell that appears.

If you don't want to track your model training, simply add to the training argument the option

```python
training_args = TrainingArguments(
    ...,
    report_to=["none"])
```

## Custom
If you want change model simply go in ```config.yaml``` and write your hugging face model name in the model value.
You can also change your data directory and number of classes with the ```filename``` and```classes``` keys.
