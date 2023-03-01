from typing import Optional, Union, List, Callable
from argparse import ArgumentParser
from abc import ABC

import torch

from loguru import logger
from pipeline.aggregator.AggregatorInterface import Aggregator


def pooling(x: Union[torch.Tensor, List[torch.Tensor]], f: Callable = torch.max) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        val, ind = f(x, dim=1, keepdim=False)
    else:
        val_ind = [f(t, dim=0, keepdim=False) for t in x]
        val = torch.stack(tensors=[_val for _val, _ in val_ind], dim=0)
        ind = torch.stack(tensors=[_ind for _, _ind in val_ind], dim=0)

    logger.debug("Found the {} values at {}", str(f).split(" ")[2], ind.cpu().tolist())

    return val


class PoolingMax(Aggregator):
    def __init__(self, weight_vector: Optional[Union[int, torch.Tensor]] = None) -> None:
        super().__init__(weight_vector)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        x = super().forward(x=x)

        return pooling(x=x, f=torch.max)


class PoolingMin(Aggregator):
    def __init__(self, weight_vector: Optional[Union[int, torch.Tensor]] = None) -> None:
        super().__init__(weight_vector)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        x = super().forward(x=x)

        return pooling(x=x, f=torch.min)


class PoolingSmooth(Aggregator, ABC):
    smooth_pooling_parser = ArgumentParser(parents=[Aggregator.general_argument_parser])
    smooth_pooling_parser.add_argument("--exponent", action="store", type=float, default=2., required=False)
    smooth_pooling_parser.add_argument("--impact", action="store", type=float, default=.25, required=False)

    def __init__(self, weight_vector: Optional[Union[int, torch.Tensor]] = None,
                 exponent: float = 2., impact: float = .25) -> None:
        """
        Initializing an Aggregation module
        (Input-->Dataset-->Encoder (should produce probabilities)-->AGGREGATOR-->final output)
        :param weight_vector: of you want to treat your modules with different strengths
        (emphasizing different modules), you can do it here (tensor of size (#modules, )).
        If you insert a number (of modules), the weighting become trainable, so will adapt during training.
        :param exponent: an exponent that regulates how mich the other modules are allowed to differ from the highest/
        lowest value. A small positive value means: every little derivation matters, a large one will lead to tolerance
        to minor derivations
        :param impact: How much should the other modules impact the return value of the maximal/
        minimal predicted probability? Should be between 0 (no) and 1 (much)
        """
        super().__init__(weight_vector)
        self.exponent = exponent
        self.impact = impact

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        logger.trace("Smoothing forward")
        return super().forward(x=x)


class PoolingSmoothMax(PoolingSmooth):
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        x = super().forward(x=x)

        x_max = pooling(x=x, f=torch.max)

        if isinstance(x, torch.Tensor):
            diff = torch.negative(torch.permute(torch.permute(x, dims=(1, 0, 2))-x_max, dims=(1, 0, 2)))
            subtrahend = torch.mean(torch.clip(diff, max=1)**self.exponent, dim=1, keepdim=False)
        else:
            diff = [-(t - x_max[i]) for i, t in enumerate(x)]
            subtrahend = torch.stack(
                tensors=[torch.mean(torch.clip(d, max=1)**self.exponent, dim=0, keepdim=False) for d in diff],
                dim=0
            )
        logger.trace("OK, looking to the other values, we have a subtrahend of {}",
                     subtrahend.cpu().tolist())

        return x_max - self.impact*subtrahend


class PoolingSmoothMin(PoolingSmooth):
    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        x = super().forward(x=x)

        x_min = pooling(x=x, f=torch.min)

        if isinstance(x, torch.Tensor):
            diff = torch.permute(torch.permute(x, dims=(1, 0, 2))-x_min, dims=(1, 0, 2))
            add = torch.mean(torch.clip(diff, max=1)**self.exponent, dim=1, keepdim=False)
        else:
            diff = [t - x_min[i] for i, t in enumerate(x)]
            add = torch.stack(
                tensors=[torch.mean(torch.clip(d, max=1) ** self.exponent, dim=0, keepdim=False) for d in diff],
                dim=0
            )
        logger.trace("OK, looking to the other values, we have a plus of {}",
                     add.cpu().tolist())

        return x_min + self.impact*add
