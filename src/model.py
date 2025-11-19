from transformers import DistilBertPreTrainedModel, DistilBertConfig, DistilBertModel
import torch

class DistilBertForSentimentClassification(DistilBertPreTrainedModel):
    """Custom DistilBERT model for sentiment (or general text) classification.

    This class extends `DistilBertPreTrainedModel` and adds a classifier head
    on top of the pooled representation of DistilBERT.  
    It optionally allows freezing the encoder for faster or low-resource training.

    Args:
        config (DistilBertConfig): Model configuration specifying hidden size, dropout, etc.
        num_labels (int): Number of output classes.
        freeze_encoder (bool, optional): If True, DistilBERT encoder parameters 
            are frozen and only the classifier head is trained. Defaults to False.
        model_name (str, optional): Name or path of the pretrained DistilBERT model 
            to load. Defaults to 'distilbert-base-uncased'.

    Attributes:
        encoder (DistilBertModel): Pretrained DistilBERT encoder.
        classifier (torch.nn.Linear): Final classification layer mapping hidden 
            representations to label logits.
        dropout (torch.nn.Dropout): Dropout applied before the classifier.
        num_labels (int): Number of classes for the task.
        model_name (str): Underlying pretrained model identifier.
    """
    def __init__(self, config, num_labels, freeze_encoder=False, model_name = 'istilbert-base-uncased'):
        super().__init__(config)

        self.num_labels = num_labels
        self.model_name = model_name
        self.encoder = DistilBertModel.from_pretrained(self.model_name)
        
        if freeze_encoder:
          for param in self.encoder.parameters():
              param.requires_grad = False
        
        self.classifier = torch.nn.Linear(in_features=config.dim, 
                                          out_features=self.num_labels, 
                                          bias=True)
        self.dropout = torch.nn.Dropout(p=0.1)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """Perform a forward pass through DistilBERT and the classifier head.

        This method runs the input through the DistilBERT encoder, extracts the
        representation of the `[CLS]` token (position 0), applies dropout, and 
        feeds it into a linear classifier.  
        If `labels` are provided, a suitable loss function is computed:

        - MSELoss for regression (`num_labels == 1`)
        - CrossEntropyLoss for classification (`num_labels > 1`)

        Args:
            input_ids (torch.Tensor, optional): Token ID sequence of shape (batch, seq_len).
            attention_mask (torch.Tensor, optional): Mask distinguishing real tokens vs padding.
            head_mask (torch.Tensor, optional): Mask for attention heads (rarely used).
            inputs_embeds (torch.Tensor, optional): Optional precomputed embeddings.
            labels (torch.Tensor, optional): Target labels for computing loss.
            output_attentions (bool, optional): If True, returns attention matrices.
            output_hidden_states (bool, optional): If True, returns all hidden states.

        Returns:
            tuple:
                - loss (torch.Tensor): Returned only if `labels` is provided.
                - logits (torch.Tensor): Raw output scores of shape (batch, num_labels).
                - additional outputs from DistilBERT (hidden states, attentions, etc.)
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_state = encoder_output[0] 
        
        pooled_output = hidden_state[:, 0]  
        pooled_output = self.dropout(pooled_output)  
        
        logits = self.classifier(pooled_output)  

        outputs = (logits,) + encoder_output[1:]

        if labels is not None:
            
            if self.num_labels == 1:               
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            
            else:                
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        outputs = (loss,) + outputs
        return outputs
