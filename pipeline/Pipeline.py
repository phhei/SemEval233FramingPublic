from functools import reduce
from pathlib import Path
from typing import Union, List, Tuple, Literal, Optional, Callable, Dict, Any, Iterator

import torch
from loguru import logger
from torch.nn import Module, Parameter
from torch.utils.data import Dataset

from pipeline.aggregator.AggregatorInterface import Aggregator
from pipeline.encoder.EncoderInterface import Encoder


class Pipeline(Module):
    def __init__(
            self, datasets_and_modules: List[Union[Tuple[Tuple[Dataset, Dataset, Dataset], Encoder],
                                                   Tuple[Tuple[Dataset, Dataset, Dataset], Encoder, Aggregator]]],
            final_aggregator: Aggregator,
            loss_for_final_aggregation: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
            loss_for_each_module: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            verbose: bool = False
    ) -> None:
        """
        Initializes a pipeline returning a final prediction probability and optionally the loss over all frames
        (for each sample)

        :param datasets_and_modules: A list of all datasets (preprocessing) and Modules
        :param final_aggregator: at the end, we have #modules predictions for each sample - how to aggregate?
        Define here ;)
        :param loss_for_each_module: you can define a loss function [(ground_truth, prediction) -> loss-number] for the
        prediction of each module
        :param loss_for_final_aggregation: you can define a loss function [(ground_truth, prediction) -> loss-number]
        for the final aggregated prediction (late fusion)
        :param verbose: enables more logging information :)
        """
        super().__init__()

        logger.info("Receiving {} pipelines", len(datasets_and_modules))

        try:
            assert len(datasets_and_modules) >= 1
            assert all([
                all([len(dm[0][i]) == len(datasets_and_modules[0][0][i]) for dm in datasets_and_modules])
                for i in range(3)
            ])
        except AssertionError:
            logger.opt(exception=True).critical("Your input (datasets_and_modules) doesn't fit together")
            exit(-100)
        except IndexError:
            logger.opt(exception=True).critical("You have to give a train, dev and test split, hence 3 splits")
            exit(-1000)

        self.datasets_and_modules: List[Union[Tuple[Tuple[Dataset, Dataset, Dataset], Encoder],
                                              Tuple[Tuple[Dataset, Dataset, Dataset], Encoder, Aggregator]]] = \
            datasets_and_modules
        self.final_aggregator: Aggregator = final_aggregator

        self.loss_for_each_module = loss_for_each_module
        self.loss_for_final_aggregation = loss_for_final_aggregation
        self.verbose = verbose

    def save_modules(self, path: Path):
        for i, line in enumerate(self.datasets_and_modules):
            encoder = line[1]
            path_expanded = path.joinpath("line-{}".format(i))
            path_expanded.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), path_expanded.joinpath("encoder.pt"))
            if len(line) >= 3:
                aggregator = line[2]
                torch.save(aggregator.state_dict(), path_expanded.joinpath("aggregator.pt"))

    def load_modules(self, path: Path):
        for line in range(len(self.datasets_and_modules)):
            path_expanded = path.joinpath("line-{}".format(line))
            if path_expanded.joinpath("encoder.pt").exists():
                encoder_weights = torch.load(path_expanded.joinpath("encoder.pt"))
                if self.verbose:
                    logger.debug("Load {} weight tensors (overwrite actual weight tensors)", len(encoder_weights))
                self.datasets_and_modules[line][1].load_state_dict(encoder_weights)
            else:
                logger.warning("No saved encoder found at line {}", line)
            encoder = self.datasets_and_modules[line][1]
            if len(self.datasets_and_modules[line]) >= 3:
                if path_expanded.joinpath("aggregator.pt").exists():
                    self.datasets_and_modules[line][2].load_state_dict(
                        torch.load(path_expanded.joinpath("aggregator.pt"))
                    )
                else:
                    logger.warning("No saved aggregator found at line {}", line)
                aggregator = self.datasets_and_modules[line][2]
            else:
                aggregator = None
            if aggregator is None:
                self.datasets_and_modules[line] = (self.datasets_and_modules[line][0], encoder)
            else:
                self.datasets_and_modules[line] = (self.datasets_and_modules[line][0], encoder, aggregator)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if recurse:
            parameter_modules = [list(dm[1].parameters(recurse=recurse)) for dm in self.datasets_and_modules]
            parameter_modules.extend(
                list(dm[2].parameters(recurse=recurse)) for dm in self.datasets_and_modules if len(dm) >= 3
            )
            parameter_modules.append(list(self.final_aggregator.parameters(recurse=recurse)))

            logger.info("Found {} torch.Modules having (potentially trainable) parameters", len(parameter_modules))

            return reduce(lambda l1, l2: l1+l2, parameter_modules)

        logger.warning("A pipeline-module don't have trainable parameters itself usually, "
                       "please call the method with \"recurse=True\"")
        return super().parameters(recurse)

    def forward(
            self,
            split: Literal["train", "dev", "test"],
            index: Union[int, Tuple[int, int]],
            enable_backpropagation: Optional[bool] = None,
            module_loss_enabled: Union[bool, float] = True,
            final_aggregation_loss_enabled: Union[bool, float] = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a forward pass using all defined modules.

        :param split: train, dev or test
        :param index: the indexes (batch)
        :param enable_backpropagation: if you are in training/ fine-tuning, True, else False
        :param module_loss_enabled: do you want (if you want to compute the loss) consider the loss
        regarding the single module predictions (float: scale->learning rate)
        :param final_aggregation_loss_enabled: do you want (if you want to compute the loss) consider the loss
        regarding the final aggregation (float: scale-> learning rate)
        :return: a not-post-processed frame-probability-distribution with shape (batch, frames),
        the ground-truth multi-hot-encoded frame vector with shape (batch, frames)
        + optional a loss number (for backpropagation)
        """
        if enable_backpropagation is None:
            enable_backpropagation = split == "train"
            logger.debug("Backpropagation: {}", enable_backpropagation)

        if enable_backpropagation:
            logger.debug("Training mode")

            for dea in self.datasets_and_modules:
                dea[1].train()

            return self._forward(split=split, index=index, compute_loss=True,
                                 module_loss_scale=float(module_loss_enabled),
                                 final_aggregation_loss_scale=float(final_aggregation_loss_enabled))
        else:
            logger.debug("Inference mode")

            for dea in self.datasets_and_modules:
                dea[1].eval()

            with torch.no_grad():
                return self._forward(split=split, index=index, compute_loss=False,
                                     module_loss_scale=float(module_loss_enabled),
                                     final_aggregation_loss_scale=float(final_aggregation_loss_enabled))

    def _forward(
            self,
            split: Literal["train", "dev", "test"],
            index: Union[int, Tuple[int, int]],
            compute_loss: bool,
            module_loss_scale: float,
            final_aggregation_loss_scale: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        logger.debug("Let's compute the forward pass through {} parallel module-lines", len(self.datasets_and_modules))
        mapping = {
            "train": 0,
            "dev": 1,
            "test": 2
        }

        if isinstance(index, int):
            index = (index, index+1)

        single_predictions = []
        ground_truths_from_single_datasets = []
        if compute_loss:
            loss = 0
            ground_truth_labels_for_entire_batch = None
        else:
            loss = None
            ground_truth_labels_for_entire_batch = None

        for dm in self.datasets_and_modules:
            try:
                data_batch: List[Dict[str, Any]] = [dm[0][mapping[split]][i] for i in range(*index)]
                ground_truth_labels_for_entire_batch = \
                    torch.stack(tensors=[sample["labels"] for sample in data_batch], dim=0)\
                        if any(map(lambda sample: "labels" in sample, data_batch)) else None
                if len(dm) == 2:
                    logger.debug("Perform single forward pass")

                    single_module_batch_prediction = dm[1](data_batch)
                else:
                    logger.debug("Perform multiple forward pass")
                    split_prediction = []
                    for i, datapoint in enumerate(data_batch):
                        logger.trace("{}. data point ({}%)", i, (i*100)/len(data_batch))

                        split_prediction.append(dm[1](datapoint))

                    single_module_batch_prediction = dm[2](
                        split_prediction
                    )

                if self.verbose:
                    logger.debug(
                        "ground_truth_labels_for_entire_batch: {} / module_prediction: {}",
                        "not given" if ground_truth_labels_for_entire_batch is None else
                        ground_truth_labels_for_entire_batch.cpu().tolist(),
                        single_module_batch_prediction.cpu().tolist()
                    )

                if compute_loss and self.loss_for_each_module is not None and module_loss_scale != 0.:
                    logger.trace("Let's compute the loss for the module {}", dm)
                    module_loss = self.loss_for_each_module(ground_truth_labels_for_entire_batch,
                                                            single_module_batch_prediction)
                    if self.verbose:
                        logger.info("The loss of this module is: {} (will be scaled by {})",
                                    module_loss.cpu().item(), module_loss_scale)
                    loss = loss + module_loss_scale * module_loss
                ground_truths_from_single_datasets.append(ground_truth_labels_for_entire_batch)
                single_predictions.append(
                    single_module_batch_prediction.detach()
                    if self.loss_for_final_aggregation is None and final_aggregation_loss_scale != 0.
                    else single_module_batch_prediction
                )
            except IndexError:
                logger.opt(exception=True).critical("Your dataset hasn't enough samples ({})",
                                                    len(dm[0][mapping[split]]))
                return torch.zeros((index[1]-index[0], 15), dtype=torch.float, device="cpu"), None, None
            except KeyError:
                logger.opt(exception=True).warning("Your dataset doesn't define the ground truth by providing an entry "
                                                   "for \"labels\" - but we have to compute the loss! "
                                                   "Skip this module ({})", (m := str(dm))[:min(len(m), 50)])
            except TypeError:
                logger.opt(exception=True).critical("Your dataset has to return a dictionary for each sample (skip {})",
                                                    (m := str(dm))[:min(len(m), 50)])
            except RuntimeError:
                logger.opt(exception=True).error("Pytorch-error, skip {}", (m := str(dm))[:min(len(m), 50)])

        if self.verbose:
            logger.success("Successfully computed {} lines of modules (each {} samples)",
                           len(single_predictions), index[1]-index[0])

        try:
            final_prediction = self.final_aggregator(torch.stack(single_predictions, dim=1))
            if self.verbose:
                logger.debug("final_prediction: {}", final_prediction.cpu().tolist())
            if compute_loss:
                if self.loss_for_final_aggregation is not None and final_aggregation_loss_scale != 0.:
                    logger.trace("Let's compute the loss for all")
                    overall_loss = self.loss_for_final_aggregation(
                        # we receive from each dataset (module) the ground truth (ideally the same from each dataset),
                        # hence having a shape of (modules, batch_size, frames). However, we need a tensor of
                        # (batch_size, frames), therefore the mean-operation
                        torch.mean(
                            torch.stack(tensors=tuple(
                                filter(lambda sgt: sgt is not None, ground_truths_from_single_datasets)
                            ), dim=0),
                            dim=0,
                            keepdim=False
                        ),
                        final_prediction
                    )
                    if self.verbose:
                        logger.info("The loss for the final prediction is: {} (will be scaled by {})",
                                    overall_loss.cpu().item(), final_aggregation_loss_scale)

                    loss = loss + final_aggregation_loss_scale * overall_loss
                elif self.verbose:
                    logger.warning("This pipeline hasn't a loss function for the overall aggregated prediction "
                                   "(or it's not enabled), hence, it's just an ensemble method with seperated "
                                   "trained module lines.")
        except RuntimeError:
            logger.opt(exception=True).critical("Either your aggregation module ({}) is broken or your loss function! "
                                                "- or you ran out of memory!",
                                                self.final_aggregator)
            return torch.zeros_like(ground_truth_labels_for_entire_batch, device="cpu") \
                       if ground_truth_labels_for_entire_batch else None, ground_truth_labels_for_entire_batch, None

        if isinstance(loss, int):
            logger.error("Our plan was to compute a loss, but both loss components (module loss and final loss) were "
                         "disabled! Please set either \"--compute_loss_for_each_module\" or \"--compute_final_loss\"")
            loss = None

        return final_prediction, ground_truth_labels_for_entire_batch, loss
