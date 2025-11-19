import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Custom PyTorch Dataset for tokenized text classification.

    This dataset takes raw sentences and labels, applies a HuggingFace tokenizer
    to each sentence, and returns tensors suitable for Transformer-based models
    (e.g., BERT, RoBERTa). Each item includes input IDs, attention mask, and the
    numerical label.

    Args:
        sentences (list or pandas.Series): Collection of raw text sentences.
        labels (list or pandas.Series): Corresponding class labels.
        tokenizer: HuggingFace tokenizer used to convert text into token IDs.
        max_len (int, optional): Maximum sequence length for tokenization.
            Sentences longer than this are truncated, shorter ones are padded.
            Defaults to 150.

    Attributes:
        sentences: Stored list/series of input sentences.
        labels: Stored list/series of labels.
        tokenizer: HuggingFace tokenizer instance.
        max_len: Maximum allowed sequence length.

    """
    def __init__(self, sentences, labels, tokenizer, max_len = 150):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        """Retrieve a tokenized and encoded sample.

        Tokenizes the sentence at the given index, producing:
        - `input_ids`: tensor of token indices padded/truncated to `max_len`
        - `attention_mask`: tensor marking true tokens vs. padding
        - `label`: integer class label as a tensor

        Args:
            item (int): Index of the sample to access.

        Returns:
            dict: A dictionary containing:
                - 'input_ids' (torch.Tensor): Token ID sequence.
                - 'attention_mask' (torch.Tensor): Attention mask.
                - 'label' (torch.Tensor): Class label.
        """
        sentence = str(self.sentences[item])
        label = self.labels[item]
        
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length', 
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
