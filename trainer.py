from transformers import Trainer
from torch.nn import BCEWithLogitsLoss
import torch

class CustomTrainer(Trainer):
    def __init__(self, *args, is_multi_label=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_multi_label = is_multi_label

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.is_multi_label:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        else:
            loss_fct = self.label_smoother
            loss = loss_fct(outputs, labels) if loss_fct else None
        return (loss, outputs) if return_outputs else loss