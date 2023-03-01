from typing import Optional, Union, List

import torch

from loguru import logger
from pipeline.aggregator.AggregatorInterface import Aggregator


class Average(Aggregator):
    def __init__(self, weight_vector: Optional[Union[int, torch.Tensor]] = None) -> None:
        super().__init__(weight_vector)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        x = super().forward(x=x)

        if isinstance(x, torch.Tensor):
            val = torch.mean(x, dim=1, keepdim=False)
        else:
            val = torch.stack(tensors=[torch.mean(t, dim=0, keepdim=False) for t in x], dim=0)

        logger.debug("The mean values are {}", val.cpu().tolist())
        return val


class HarmonicAverage(Aggregator):
    def __init__(self, weight_vector: Optional[Union[int, torch.Tensor]] = None) -> None:
        super().__init__(weight_vector)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        x = super().forward(x=x)

        if isinstance(x, torch.Tensor):
            val = x.shape[1]/torch.sum(1/torch.permute(x, dims=(0, 2, 1)), dim=-1, keepdim=False)
        else:
            val = torch.stack(
                tensors=[t.shape[0]/torch.sum(1/torch.permute(t, dims=(1, 0)), dim=-1, keepdim=False) for t in x],
                dim=0
            )

        logger.debug("The harmonic mean values are {}", val.cpu().tolist())

        return val
