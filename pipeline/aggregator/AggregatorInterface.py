from abc import ABC, abstractmethod
from typing import Optional, Union, List
from argparse import ArgumentParser

import torch
from torch.nn import Module, Parameter

from loguru import logger


def weight_str_to_parse(x: Optional[List[str]]) -> Optional[Union[int, torch.Tensor]]:
    if x is None:
        return x

    if len(x) == 1:
        try:
            return int(x[0])
        except ValueError:
            return torch.tensor(
                data=float(x[0]), dtype=torch.float,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

    return torch.tensor(
        data=[float(_x) for _x in x],
        dtype=torch.float, device="cuda" if torch.cuda.is_available() else "cpu"
    )


class Aggregator(ABC, Module):
    general_argument_parser = ArgumentParser(prog="Aggregator", add_help=False, allow_abbrev=True)
    general_argument_parser.add_argument(
        "--weight_vector", action="store", nargs="+", default=None,
        type=str, required=False
    )

    def __init__(self, weight_vector: Optional[List[str]] = None) -> None:
        """
        Initializing an Aggregation module
        (Input-->Dataset-->Encoder (should produce probabilities)-->AGGREGATOR-->final output)
        :param weight_vector: of you want to treat your modules with different strengths
        (emphasizing different modules), you can do it here (tensor of size (#modules, )).
        If you insert a number (of modules), the weighting become trainable, so will adapt during training.
        """
        super().__init__()
        self.module_weight_vector = weight_str_to_parse(weight_vector)
        if isinstance(self.module_weight_vector, int):
            self.module_weight_vector = Parameter(
                data=torch.ones(
                    (self.module_weight_vector, ),
                    dtype=torch.float,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            )
            logger.debug("Created a parameter out of your weight_vector-param ({}): {}",
                         weight_vector, self.module_weight_vector)

    @abstractmethod
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        A method aggregating multiple module frame predictions to a single frame prediction (per each batch)
        :param x: a concatenated prediction tensor of shape (batch, #modules, frames)
        OR a list of unequal-length-predictions [(#text_parts, frames)]
        :return: a aggregation of shape (batch, frames)
        """
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                logger.warning("Got only a single frame prediction!")
                x = torch.unsqueeze(x, dim=0)
            if len(x.shape) == 2:
                logger.debug("You got predictions only for a single sample!")
                x = torch.unsqueeze(x, dim=0)

            assert len(x.shape) == 3
            logger.trace("Got {} predicted samples with {} predictions each", x.shape[0], x.shape[1])
        else:
            logger.info("Your predictions-tensor as List of {} tensors "
                        "(probably because of an aggregation of an text-split batch with unequal length)", len(x))
            if self.module_weight_vector is not None:
                return [torch.permute(self.module_weight_vector * torch.permute(t, dims=(1, 0)), dims=(1, 0))
                        for t in x]

        if self.module_weight_vector is not None:
            x = torch.permute(self.module_weight_vector * torch.permute(x, dims=(0, 2, 1)), dims=(0, 2, 1))

        return x
