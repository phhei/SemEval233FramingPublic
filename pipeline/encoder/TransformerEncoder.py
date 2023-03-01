from typing import Union, Any, Dict, Iterable

import transformers
from transformers.modeling_outputs import SequenceClassifierOutput
from argparse import ArgumentParser
import torch
from loguru import logger

from pipeline.encoder.EncoderInterface import Encoder, list_batch_to_tensor_batch


class Huggingface(Encoder):
    transformer_parser = ArgumentParser(parents=[Encoder.encoder_parser])
    transformer_parser.add_argument("-t", "--transformer_name", action="store", default="xlm-roberta-base", type=str)
    transformer_parser.add_argument("-n", "--num_frame_classes", action="store", default=14, type=int)

    def __init__(self, predict_number_of_frames: bool = False, transformer_name: str = "xlm-roberta-base",
                 num_frame_classes: int = 14) -> None:
        super().__init__(predict_number_of_frames)

        self.model: transformers.PreTrainedModel = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=transformer_name,
            num_labels=num_frame_classes+int(predict_number_of_frames),
            problem_type="multi_label_classification",
            return_dict=True
        )
        logger.success("Successfully loaded our model: {} ({})", self.model.name_or_path, self.model.config)
        if torch.cuda.is_available():
            logger.trace("GPU available, put model on it!")
            self.model.to("cuda")

    def forward(self, x: Union[Any, Dict[str, torch.Tensor], Iterable[Dict[str, torch.Tensor]]]) -> torch.Tensor:
        x = list_batch_to_tensor_batch(x=x)

        if isinstance(x, Dict) and "labels" in x:
            labels = x.pop("labels")
            logger.debug("We want prevent the model to compute its own loss (leading to errors in the text-"
                         "splitting-approach), hence we pop it: {}", labels)

        output: SequenceClassifierOutput = self.model(**x)

        logger.debug("Successfully receives an output: {}", output)

        return self.sigmoid_outputs(output.logits)
