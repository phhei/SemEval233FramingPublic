from abc import ABC, abstractmethod
from pathlib import Path

import torch
from loguru import logger
from pandas import DataFrame
from torch.cuda import is_available as gpu_available
from torch.utils.data.dataset import Dataset, T_co
import torch_geometric as gtc

from scorers.scorer_subtask_2 import read_frame_list_from_file

frame_set_path = Path("scorers/frames_subtask2.txt")


class FrameDataset(Dataset, ABC):
    def __init__(self):
        """
        Initializing a preprocessing module
        (Input-->DATASET-->Encoder (should produce probabilities)-->aggregator-->final output)
        """
        if frame_set_path.exists():
            self.label_positions = read_frame_list_from_file(str(frame_set_path.absolute()))
            logger.success("Read {} frames from \"{}\"", len(self.label_positions), frame_set_path.name)
        else:
            logger.warning("\"{}\" doesn't exist, can't load the Frame-set!", frame_set_path.absolute())
            self.label_positions = ["n/a"]
        self.conv_data = []

        self.return_language: bool = False

    @abstractmethod
    def init_data(self, df: DataFrame) -> None:
        """
        Fills the conv_data with data getting by a dataframe. The dataframe has a row for each article which should
        be classified:

        --------------------------
        |(Article-ID)|text|frames|
        --------------------------

        - ATTENTION: if the representation consists of Tensors, the tensors must not have a BATCH-dimension!
        - IMPORTANT: if you use tensor, please store them in the CPU (tensor.cpu()) to save memory on the GPU.
        If something should be processed, it is put to the GPU by this class

        :param df: the dataframe which should be put in conv_data row by row (preserve the order!)
        :return: nothing
        """
        pass

    def convert_frame_text(self, frame_text: str) -> torch.Tensor:
        """
        The frames (ground truth) are given in a form of a comma-seperated list,
        e.g. Crime_and_punishment,Policy_prescription_and_evaluation.
        This method converts such strings into a multi-hot encoded vector.

        :param frame_text: the frame-string in the dataframe
        :return: multi-hot encoded vector
        """
        logger.trace("Process \"{}\"...", frame_text)
        frame_parts = {frame.strip() for frame in frame_text.split(sep=",")}

        return torch.tensor(
            data=[int(label in frame_parts) for label in self.label_positions],
            dtype=torch.float,
            device="cpu"
        )

    def __getitem__(self, index) -> T_co:
        """
        Get the machine-readable representation of one sample

        USE ONLY AFTER CALLING init_data

        :param index: the index of the sample
        :return: the representation (should be a dict containing the key "labels" for the labels).
        If self.return_language = True, the representation contains a cue on the langauge, too.
        ATTENTION: if the representation consists of Tensors, the tensors must not have a BATCH-dimension!
        """
        if self.return_language:
            ret_dict = self.conv_data[index]
        else:
            ret_dict = self.conv_data[index].copy()
            try:
                language = ret_dict.pop("language")
                logger.debug("Drop the language label: {}", language)
            except KeyError:
                logger.opt(exception=False).warning("Your data doesn't contain a language-tag ({})", index)

        return {k: (t.to("cuda" if gpu_available() else "cpu") if isinstance(t, (torch.Tensor, gtc.data.Data)) else t)
                for k, t in ret_dict.items()}

    def __len__(self):
        return len(self.conv_data)
