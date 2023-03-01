from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Any, Dict, Iterable, Optional

import torch

from pipeline.encoder.EncoderInterface import Encoder
from loguru import logger

dict_activation = {
    "Linear": torch.nn.Identity,
    "ReLU": torch.nn.ReLU,
    "GeLU": torch.nn.GELU,
    "Sigmoid": torch.nn.Sigmoid,
    "TanH": torch.nn.Tanh
}


def str_to_activation(f_input: str) -> torch.nn.Module:
    return dict_activation[f_input]()


class LinearNNEncoder(Encoder):
    linear_parser = ArgumentParser(parents=[Encoder.encoder_parser])
    linear_parser.add_argument("--in_features", action="store", type=int, default=300, required=False)
    linear_parser.add_argument("--activation_module", action="store", type=str_to_activation, default=torch.nn.ReLU(),
                               required=False)
    linear_parser.add_argument("--num_frame_classes", action="store", type=int, default=14, required=False)
    linear_parser.add_argument("--enable_dropout", action="store_true", default=False, required=False)
    linear_parser.add_argument("--pretrained_params", action="store", default=None, required=False, type=Path)

    def __init__(self, activation_module: torch.nn.Module, in_features: int = 300,
                 predict_number_of_frames: bool = False, num_frame_classes: int = 14,
                 enable_dropout: bool = True, pretrained_params: Optional[Path] = None) -> None:
        super().__init__(predict_number_of_frames=predict_number_of_frames)

        self.in_layer = torch.nn.Linear(in_features=in_features, out_features=num_frame_classes*10, bias=True,
                                        device="cuda" if torch.cuda.is_available() else "cpu")

        self.activation_module = activation_module

        if enable_dropout:
            self.dropout = torch.nn.Dropout(p=.25)
        else:
            self.dropout = torch.nn.Identity()

        self.out_layer = torch.nn.Linear(in_features=2*num_frame_classes*10,
                                         out_features=num_frame_classes+int(self.predict_number_of_frames), bias=True,
                                         device="cuda" if torch.cuda.is_available() else "cpu")

        if pretrained_params is not None:
            logger.debug("OK, let's pre-load the params from \"{}\"", pretrained_params)
            try:
                self.load_state_dict(state_dict=torch.load(f=pretrained_params), strict=True)
            except RuntimeError:
                logger.opt(exception=True).error("The architectures don't fit together -- no pretraining!")
            except IOError:
                logger.opt(exception=True).error("\"{}\" doesn't exist", pretrained_params.absolute())

    def forward(self, x: Union[Any, Dict[str, torch.Tensor], Iterable[Dict[str, torch.Tensor]]]) -> torch.Tensor:
        def _single_sample_forward(tensor_for_single_sample_2d: torch.Tensor) -> torch.Tensor:
            conv_in_tensors = []

            for tensor_slice in tensor_for_single_sample_2d:
                if torch.all(tensor_slice == -1.).item():
                    logger.trace("Detected padding slice -- ignore")
                else:
                    conv_in_tensors.append(
                        self.dropout(self.activation_module(self.in_layer(tensor_slice)))
                    )

            logger.debug("Encoded {} tensor slices (e.g. word embeddings)", len(conv_in_tensors))

            if len(conv_in_tensors) == 1:
                logger.warning("Only a single slice (one token in this sample/ text split) - "
                               "we can't calculate a derivation!")
                mean = conv_in_tensors[0]
                std = torch.zeros_like(mean)
            else:
                std, mean = torch.std_mean(torch.stack(conv_in_tensors, dim=0), dim=0)
            logger.trace("Combining std({}) and mean({}) now", std.cpu().tolist(), mean.cpu().tolist())
            std_mean = torch.concat((std, mean), dim=0)

            final = self.out_layer(std_mean)

            return final

        predictions = []

        for _x in (x["input_tensors"] if isinstance(x, Dict) else x):
            if isinstance(_x, Dict):
                predictions.append(
                    _single_sample_forward(_x["input_tensors"])
                )
            elif isinstance(_x, torch.Tensor):
                predictions.append(
                    _single_sample_forward(_x)
                )

        logger.trace("Combine {} predictions now", len(predictions))

        stacked_predictions = torch.stack(predictions, dim=0)
        if torch.any(torch.isnan(stacked_predictions)).item():
            logger.warning("This prediction-tensor is partially invalid: {}", stacked_predictions)
            stacked_predictions = \
                torch.masked_fill(stacked_predictions, mask=torch.isnan(stacked_predictions), value=0.)

        return self.sigmoid_outputs(stacked_predictions)
