from abc import ABC, abstractmethod
from pathlib import Path
from argparse import ArgumentParser
import os
from typing import Optional, Dict, Union

import torch
import torch_geometric as gtc
import torch_geometric.nn as gnn
from loguru import logger
from torch.cuda import is_available as gpu_available
from torch.utils.data.dataset import Dataset, T_co
from pandas import DataFrame

from scorers.scorer_subtask_2 import read_frame_list_from_file
from pipeline.preprocessing.FrameDatasetInterface import FrameDataset

frame_set_path = Path("scorers/frames_subtask2.txt")

# assignment of edge type to index for RGCN
# only includes edge types that occur in train or dev set. Might need to extend for test set. 
COMBINE = {
    'fine_grained': [
        ['/r/MadeOf'],
        ['/r/CreatedBy'],
        ['/r/DefinedAs', '/r/SymbolOf'],
        ['/r/ReceivesAction'],
        ['/r/CausesDesire'],
        ['/r/MotivatedByGoal'],
        ['/r/DistinctFrom'], 
        ['/r/MannerOf'],
        ['/r/Desires'],
        ['/r/Causes'],
        ['/r/HasA', '/r/PartOf'],  # edges are used undirected, so we can combine them
        ['/r/HasSubevent', '/r/HasLastSubevent', '/r/HasFirstSubevent'],
        ['/r/HasPrerequisite'], 
        ['/r/Antonym'],
        ['/r/HasProperty'],
        ['/r/UsedFor'],
        ['/r/CapableOf'],
        ['/r/AtLocation', '/r/LocatedNear'],
        ['/r/Synonym', '/r/FormOf'],  # \r\FormOf is semantically close to synonym
        ['/r/HasContext'],
        ['/r/IsA'],
    ],
    'coarse_grained': [
        ['/r/MadeOf', '/r/DefinedAs', '/r/HasA', '/r/PartOf', '/r/MannerOf', '/r/HasProperty', '/r/Synonym', '/r/FormOf', '/r/IsA', '/r/SymbolOf'], # some kind of version / subpart  # evtl add causes here
        ['/r/CreatedBy', '/r/UsedFor', '/r/ReceivesAction', '/r/CausesDesire', '/r/MotivatedByGoal', '/r/Desires', '/r/CapableOf', '/r/HasContext', '/r/AtLocation', '/r/LocatedNear'],  # things that interact. Also include locations, as things need to be in the same location to interact
        ['/r/HasSubevent', '/r/HasLastSubevent', '/r/HasFirstSubevent', '/r/HasPrerequisite', '/r/Causes'],  # temporal stuff
        ['/r/DistinctFrom', '/r/Antonym'], # negative connection
    ],
    'pos_neg': [
        ['/r/MadeOf', '/r/CreatedBy', '/r/DefinedAs', '/r/ReceivesAction', '/r/CausesDesire', '/r/MotivatedByGoal', '/r/MannerOf', '/r/Desires', '/r/Causes', '/r/HasA', '/r/PartOf', '/r/HasSubevent', '/r/HasLastSubevent', '/r/HasFirstSubevent', '/r/HasPrerequisite', '/r/HasProperty', '/r/UsedFor', '/r/CapableOf', '/r/AtLocation', '/r/LocatedNear', '/r/Synonym', '/r/FormOf', '/r/HasContext', '/r/IsA', '/r/SymbolOf'],  # positive edges
        ['/r/Antonym', '/r/DistinctFrom'] # negative edges
    ],
}

class FrameDatasetForGnn(FrameDataset):
    dataset_parser = ArgumentParser(prog="Dataset-parser", add_help=False, allow_abbrev=True)
    dataset_parser.add_argument("--data_dir", type=str, required=False, default="data/GraphData/lookup=sbert_1_r2nl=free_kg=CNsmall_pruning=True_k=1_oPCp=False_alledges=False", help="Path to the data directory")
    dataset_parser.add_argument("--edge_type_combination", 
    type=str,  # should be Optional[str], but then it is not possible to set a non-None value
    required=False, default=None, help="Path to the data directory")

    def __init__(
        self, 
        data_dir: str = "data/GraphData/lookup=sbert_1_r2nl=free_kg=CNsmall_pruning=True_k=1_oPCp=False_alledges=False",
        edge_type_combination: Optional[str] = None
    ):
        """
        :param data_dir: path to data
        :param edge_type_combination: how edge types are combined. Only relevant for relational-GCN (RGCNConv and FastRGCNConv). Must be `None` or a key of `COMBINE`
        """
        super().__init__()
        self.data_dir = data_dir

        if edge_type_combination == None:
            self.edge_type_dict = None
        else:
            self.edge_type_dict = {e: i for i, es in enumerate(COMBINE[edge_type_combination]) for e in es}

    def init_data(self, df: DataFrame) -> None:
        assert len(self.conv_data) == 0, 'need to append to self.conv_data instead of overwriting it.'

        self.conv_data = [self._load_one_instance(article_id, row) for (article_id, row) in df.iterrows()]  

    def _load_one_instance(self, article_id:str, row) -> Dict[str, gtc.data.Data]:
        """
        :param article_id: id of the article
        :param row: row of the dataframe
        """
        data = torch.load(self._get_fn(article_id))

        if self.edge_type_dict != None:
            data.edge_type = torch.tensor([self.edge_type_dict[e] for e in data.edge_type])

        out_data = {
            "data": data
        }

        if "frames" in row:
            out_data["labels"] = self.convert_frame_text(frame_text=row["frames"])

        if "language" in row:
            out_data["language"] = row["language"]
        else:
            logger.warning("No langauge information available in row {}", _id)
            out_data["language"] = "n/a"

        return out_data

    def _get_fn(self, article_id: int) -> str:
        return os.path.join(self.data_dir, f"{article_id}.pt")
