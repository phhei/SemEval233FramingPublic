from abc import ABC, abstractmethod
from typing import Union, Any, Iterable, Dict
from argparse import ArgumentParser

import torch
from loguru import logger
from torch.nn import Module


def list_batch_to_tensor_batch(
        x: Union[Any, Dict[str, torch.Tensor], Iterable[Dict[str, torch.Tensor]]],
        padding_value: float = 0.
) -> Union[Any, torch.Tensor, Dict]:
    if isinstance(x, Iterable):
        if all(map(lambda _x: isinstance(_x, torch.Tensor), x)):
            logger.trace("You have {} loose tensors, stack them together", len(x))
            x = torch.stack(x, dim=0)
        elif all(map(lambda _x: isinstance(_x, Dict), x)):
            logger.trace("You have {} batched encodings, stack them together", len(x))
            x_dict = {k: [v] for k, v in x[0].items() if k != "labels"}
            if len(x) >= 2:
                for _x_d in x[1:]:
                    for k, v in _x_d.items():
                        if k != "labels":
                            x_dict[k] += [v]

            x = {
                k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=padding_value)
                for k, v in x_dict.items()
            }
    return x


class Encoder(ABC, Module):
    encoder_parser = ArgumentParser(prog="Aggregator", add_help=False, allow_abbrev=True)
    encoder_parser.add_argument("--predict_number_of_frames", action="store_true", default=False)

    def __init__(self, predict_number_of_frames: bool = False) -> None:
        """
        Initializing a Decoder module (may be more than one module running in parallel)
         (Input-->Dataset-->ENCODER (should produce probabilities)-->aggregator-->final output).

        :param predict_number_of_frames: should the expected number of frames also be computed or not?
        """
        super().__init__()
        self.predict_number_of_frames = predict_number_of_frames

    @abstractmethod
    def forward(self, x: Union[Any, Dict[str, torch.Tensor], Iterable[Dict[str, torch.Tensor]]]) -> torch.Tensor:
        """
        Encodes the raw machine-readable input (provided as List[Sample]) by predicting frame probabilities
        :param x: a batch of inputs. However, if you need special stuff for your encoder,
        because, e.g., it's a GNN, you can define more fany stuff here
        :return: predictions in a shape of (batch, frames)
        (if predict_number_of_frames, each frame prediction has in the first
        place an estimation how many frames occur in the text)
        """
        pass

    def sigmoid_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        We need to have the outputs in a range from 0 to 1 for each frame (probabilities).
        If you underlying module can't accomplish this, please perform to this option
        :param x: the predicted tensor
        :return: the predicted tensor (sigmoid)
        """
        logger.trace("We sigmoid following logits: {}", x)
        if self.predict_number_of_frames:
            x = torch.concat([torch.unsqueeze(x[:, 0], dim=-1), torch.sigmoid(x[:, 1:])], dim=-1)
        else:
            x = torch.sigmoid(x)

        return x
