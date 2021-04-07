import argparse
import os
from typing import Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from seqeval.metrics import accuracy_score
from seqeval.metrics.sequence_labeling import precision_recall_fscore_support
from seqeval.metrics.v1 import (
    precision_recall_fscore_support as precision_recall_fscore_support_v1,
)
from seqeval.scheme import BILOU
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput

from data import *
from tokenizer import (
    LabelTokenAligner,
    custom_tokenizer_from_pretrained,
)

# huggingface/tokenizers: Disabling parallelism to avoid deadlocks.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TokenClassificationModule(pl.LightningModule):
    """
    Initialize a model and config for token-classification
    """

    def __init__(
        self,
        hparams: Union[Dict, argparse.Namespace],
        bilou: bool = True,
        use_datasets: bool = False,
    ):
        self.bilou = bilou
        self.use_datasets = use_datasets
        # NOTE: internal code may pass hparams as dict **kwargs
        if isinstance(hparams, Dict):
            hparams = argparse.Namespace(**hparams)
        super().__init__()
        # Enable to access arguments via self.hparams
        self.save_hyperparameters(hparams)

        self.step_count = 0
        self.cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        if self.cache_dir is not None and not os.path.exists(self.hparams.cache_dir):
            os.mkdir(self.cache_dir)
        # tokenizer & label-aligner
        tokenizer_path = (
            hparams.model_name_or_path
            if hparams.tokenizer_path is None
            else hparams.tokenizer_path
        )
        self.tokenzier = custom_tokenizer_from_pretrained(
            tokenizer_path, cache_dir=self.cache_dir
        )
        self.label_ids_to_label = LabelTokenAligner.get_ids_to_label(
            labels_path=hparams.labels, bilou=self.bilou
        )
        num_labels = len(self.label_ids_to_label)
        # config
        config_path = (
            hparams.model_name_or_path
            if hparams.config_path is None
            else hparams.config_path
        )
        self.config: PretrainedConfig = AutoConfig.from_pretrained(
            config_path,
            **({"num_labels": num_labels}),
            cache_dir=self.cache_dir,
        )
        extra_model_params = (
            "encoder_layerdrop",
            "decoder_layerdrop",
            "dropout",
            "attention_dropout",
        )
        for p in extra_model_params:
            if getattr(self.hparams, p, None) and hasattr(self.config, p):
                setattr(self.config, p, getattr(self.hparams, p, None))
        # model
        model_checkpoint = hparams.model_name_or_path
        self.model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            config=self.config,
            cache_dir=self.cache_dir,
            from_tf=bool(".ckpt" in model_checkpoint),
        )
        if self.hparams.freeze_pretrained:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        if hparams.do_train:
            outputdir = self.hparams.output_dir
            self.train_loss_log = os.path.join(outputdir, "train_loss.csv")
            self.dev_loss_log = os.path.join(outputdir, "dev_loss.csv")
            self.test_loss_log = os.path.join(outputdir, "test_loss.csv")
            self.loss_log_format = "{:.05},{:.05},{:.05}"
            with open(self.train_loss_log, "w") as fp:
                fp.write("PRECISION,RECALL,F1")
                fp.write("\n")
            with open(self.dev_loss_log, "w") as fp:
                fp.write("PRECISION,RECALL,F1")
                fp.write("\n")
            with open(self.test_loss_log, "w") as fp:
                fp.write("PRECISION,RECALL,F1")
                fp.write("\n")

            # self.model.train()

    def forward(self, input_ids, attention_mask, labels) -> TokenClassifierOutput:
        """BertForTokenClassification.forward"""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def convert_labels_pair(self, label_ids_batch, logits_batch):
        label_ids_batch = label_ids_batch.detach().cpu().numpy()
        logits_batch = logits_batch.detach().cpu().numpy()

        golds_batch: StrListList = [
            [
                self.label_ids_to_label[label_id]
                for label_id in label_ids
                if label_id != PAD_TOKEN_LABEL_ID
            ]
            for label_ids in label_ids_batch
        ]
        preds_batch: StrListList = [
            [
                self.label_ids_to_label[label_id]
                for label_id in label_ids
                if label_id != PAD_TOKEN_LABEL_ID
            ]
            for label_ids in np.argmax(logits_batch, axis=2)
        ]

        preds_batch = [
            labels[: len(golds)] for golds, labels in zip(golds_batch, preds_batch)
        ]

        return golds_batch, preds_batch

    def eval_f1(self, outputs):
        target_list: StrListList = []
        preds_list: StrListList = []
        for output in outputs:
            golds_batch, preds_batch = self.convert_labels_pair(
                output["target"], output["prediction"]
            )
            target_list.extend(golds_batch)
            preds_list.extend(preds_batch)
        accuracy = accuracy_score(target_list, preds_list)
        if self.bilou:
            precision, recall, f1, support = precision_recall_fscore_support_v1(
                target_list, preds_list, scheme=BILOU, average="micro"
            )
        else:
            precision, recall, f1, support = precision_recall_fscore_support(
                target_list, preds_list, average="micro"
            )
        return accuracy, precision, recall, f1, support

    def log_prf(self, p, r, f, output_file):
        if output_file is not None and os.path.exists(output_file):
            with open(output_file, "a") as fp:
                fp.write(self.loss_log_format.format(p, r, f))
                fp.write("\n")

    def predict_step(self, batch) -> Dict[str, StrListList]:
        output = self.forward(
            input_ids=batch.input_ids.to(self.device),
            attention_mask=batch.attention_mask.to(self.device),
            labels=batch.label_ids.to(self.device),
        )
        tokens_batch: StrListList = [
            [
                tok
                for tok in self.tokenizer.convert_ids_to_tokens(ids)
                if tok != PAD_TOKEN
            ]
            for ids in batch.input_ids
        ]
        golds_batch, preds_batch = self.convert_labels_pair(
            output.label_ids, output.logits
        )
        # preds_batch = [
        #     preds[: len(tokens)] for preds, tokens in zip(preds_batch, tokens_batch)
        # ]
        assert len(preds_batch[0]) == len(tokens_batch[0])
        assert len(preds_batch[0]) == len(golds_batch[0])
        return {
            "input_ids": tokens_batch,
            "target": golds_batch,
            "prediction": preds_batch,
        }

    def training_step(self, train_batch, batch_idx) -> Dict[str, torch.Tensor]:
        if self.use_datasets:
            output = self.forward(
                input_ids=train_batch["input_ids"].to(self.device),
                attention_mask=train_batch["attention_mask"].to(self.device),
                labels=train_batch["labels"].to(self.device),
            )
            targets = train_batch["labels"]
        else:
            output = self.forward(
                input_ids=train_batch.input_ids.to(self.device),
                attention_mask=train_batch.attention_mask.to(self.device),
                labels=train_batch.label_ids.to(self.device),
            )
            targets = train_batch.label_ids
        loss = output.loss
        self.log("train_loss", loss, prog_bar=True)
        if self.hparams.monitor_training:
            return {
                "loss": loss,
                "prediction": output.logits,
                "target": targets,
            }
        else:
            return {"loss": loss}

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        if self.hparams.monitor_training:
            accuracy, precision, recall, f1, support = self.eval_f1(outputs)
            self.log_prf(precision, recall, f1, self.train_loss_log)
            self.log("train_accuracy", accuracy)
            self.log("train_precision", precision)
            self.log("train_recall", recall)
            self.log("train_f1", f1)
            self.log("train_support", support)

    def validation_step(self, val_batch, batch_idx) -> Dict[str, torch.Tensor]:
        # print(val_batch.input_ids.shape)
        # print(val_batch.label_ids.shape)
        if self.use_datasets:
            output = self.forward(
                input_ids=val_batch["input_ids"].to(self.device),
                attention_mask=val_batch["attention_mask"].to(self.device),
                labels=val_batch["labels"].to(self.device),
            )
            targets = val_batch["labels"]
        else:
            output = self.forward(
                input_ids=val_batch.input_ids.to(self.device),
                attention_mask=val_batch.attention_mask.to(self.device),
                labels=val_batch.label_ids.to(self.device),
            )
            targets = val_batch.label_ids

        if self.hparams.monitor == "loss":
            return {
                "val_step_loss": output.loss,
            }
        else:
            return {
                "val_step_loss": output.loss,
                "prediction": output.logits,
                "target": targets,
            }

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        if self.hparams.monitor == "loss":
            avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
            self.log("val_loss", avg_loss, sync_dist=True)
        else:
            accuracy, precision, recall, f1, support = self.eval_f1(outputs)
            self.log_prf(precision, recall, f1, self.dev_loss_log)
            self.log("val_accuracy", accuracy)
            self.log("val_precision", precision)
            self.log("val_recall", recall)
            self.log("val_f1", f1)
            self.log("val_support", support)

    def test_step(self, test_batch, batch_idx) -> Dict[str, torch.Tensor]:
        if self.use_datasets:
            output = self.forward(
                input_ids=test_batch["input_ids"].to(self.device),
                attention_mask=test_batch["attention_mask"].to(self.device),
                labels=test_batch["labels"].to(self.device),
            )
            targets = test_batch["labels"]
        else:
            output = self.forward(
                input_ids=test_batch.input_ids.to(self.device),
                attention_mask=test_batch.attention_mask.to(self.device),
                labels=test_batch.label_ids.to(self.device),
            )
            targets = test_batch.label_ids

        return {"target": targets, "prediction": output.logits}

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        accuracy, precision, recall, f1, support = self.eval_f1(outputs)
        self.log_prf(precision, recall, f1, self.test_loss_log)
        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_support", support)

    @staticmethod
    def get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
    ):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            optimizer (:class:`~torch.optim.Optimizer`):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`):
                The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
                The total number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
            last_epoch,
        )

    def configure_optimizers(self):
        """Prepare optimizer and scheduler (linear warmup and decay)"""

        adafactor = False
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_epsilon = 1e-8

        lr_scheduler = "linear"
        warmup_steps = 0

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if adafactor:
            optimizer_cls = torch.optim.Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = torch.optim.AdamW
            optimizer_kwargs = {
                "betas": (adam_beta1, adam_beta2),
                "eps": adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.hparams.learning_rate
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if lr_scheduler == "linear":
            self.lr_scheduler = self.get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.hparams.max_epochs,
            )
            return [self.optimizer], [self.lr_scheduler]
        else:
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min" if self.hparams.monitor == "loss" else "max",
                factor=self.hparams.anneal_factor,
                patience=self.hparams.patience,
                min_lr=1e-5,
                verbose=True,
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "monitor": "val_loss" if self.hparams.monitor == "loss" else "val_f1",
            }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--learning_rate",
            default=1e-3,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.",
        )
        parser.add_argument(
            "--patience",
            default=3,
            type=int,
            help="Number of epochs with no improvement after which learning rate will be reduced.",
        )
        parser.add_argument(
            "--anneal_factor",
            default=5e-5,
            type=float,
            help="Factor by which the learning rate will be reduced.",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--monitor",
            default="loss",
            type=str,
            help="Metrics to monitor on the validation dataset.",
        )
        parser.add_argument(
            "--monitor_training",
            action="store_true",
            help="Whether to monitor train metrics.",
        )
        parser.add_argument("--freeze_pretrained", action="store_true")
        return parser
