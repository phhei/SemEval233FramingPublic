from typing import Optional, List, Dict, Iterable
from argparse import ArgumentParser

import torch
import transformers
from loguru import logger
from nltk import sent_tokenize
from pandas import DataFrame

from config_languages import LANGUAGE_ABBREVIATION_TO_FULL
from pipeline.preprocessing.FrameDatasetInterface import FrameDataset


class FrameDatasetForTransformers(FrameDataset):
    dataset_parser = ArgumentParser(prog="Dataset-parser", add_help=False, allow_abbrev=True)
    dataset_parser.add_argument("--tokenizer", action="store", default="xlm-roberta-base", type=str, required=False)
    dataset_parser.add_argument("--max_length", action="store", default=None, type=int, required=False)
    dataset_parser.add_argument("--text_separation", action="store", default=None, required=False)
    dataset_parser.add_argument("--max_text_separations", action="store", default=None, required=False, type=int)

    def __init__(self, tokenizer: str, max_length: Optional[int] = None, text_separation: Optional[str] = None,
                 max_text_separations: Optional[int] = None):
        super().__init__()

        self.several_input_tensors_per_sample = text_separation is not None
        logger.trace("Several_input_tensors_per_sample: {}", self.several_input_tensors_per_sample)

        self.max_length = max_length
        self.text_separation = text_separation

        if self.several_input_tensors_per_sample:
            if max_text_separations is not None:
                logger.warning("You enabled the \"max_text_separations\"={}, so it's possible to miss text parts. "
                               "This fasten your runs and might lead to less OutOfMemory-Errors.", max_text_separations)
            self.max_text_separations = max_text_separations
        else:
            if max_text_separations is not None:
                logger.debug("We ignore your param \"max_text_separations\"={} since we don't seperate/ "
                             "split our input texts", max_text_separations)
            self.max_text_separations = None

        self.tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        logger.info("Loaded following tokenizer: {} ({} vocabs)",
                    self.tokenizer.name_or_path, self.tokenizer.vocab_size)

    def init_data(self, df: DataFrame) -> None:
        def split_function(column: Dict) -> List[str]:
            if self.text_separation == "sent":
                ret = sent_tokenize(
                    column["text"],
                    language=LANGUAGE_ABBREVIATION_TO_FULL.get(column["language"], "english")
                )
            else:
                ret = column["text"].split(self.text_separation)

            logger.debug("Found {} splits with the text-seperator {}", len(ret), self.text_separation)

            return ret

        for _id, columns in df.iterrows():
            try:
                if self.several_input_tensors_per_sample:
                    split_text = split_function(column=columns)
                    if self.max_text_separations is not None and len(split_text) > self.max_text_separations:
                        logger.info("We have to truncate the text. {} text parts -> {} text parts",
                                    len(split_text), self.max_text_separations)
                        split_text = split_text[:self.max_text_separations]
                    dicts = \
                        [
                            self.tokenizer(
                                text=text.strip(),
                                max_length=self.max_length or self.tokenizer.model_max_length,
                                truncation=True,
                                is_split_into_words=False,
                                return_attention_mask=False,
                                return_token_type_ids=False,
                                return_offsets_mapping=False,
                                return_overflowing_tokens=False,
                                return_special_tokens_mask=False,
                                return_length=True
                            ) for text in split_text
                        ]
                    logger.debug("Found {} text parts in \"{}\"", len(dicts), columns["text"])
                    final_length = sum([sum(l) if isinstance(l := d.pop("length"), Iterable) else l for d in dicts])
                    conv_row = dict()
                    for d in dicts:
                        for k, v in d.items():
                            if k in conv_row:
                                conv_row[k] += [v]
                            else:
                                conv_row[k] = [v]
                    conv_row = self.tokenizer.pad(
                        encoded_inputs=conv_row, padding=True, verbose=False, return_tensors="pt"
                    ).data
                    # conv_row["length"] = torch.tensor(data=final_length, device="cpu", dtype=torch.long)
                    logger.trace("Successfully converted the text ({}) - final length: {}",
                                 self.text_separation, final_length)
                else:
                    conv_row = {k: torch.squeeze(v) for k, v in self.tokenizer(
                        text=columns["text"],
                        max_length=self.max_length,
                        is_split_into_words=False,
                        return_tensors="pt",
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_offsets_mapping=False,
                        return_overflowing_tokens=False,
                        return_special_tokens_mask=False,
                        return_length=False
                    ).items() if isinstance(v, torch.Tensor)}

                if "frames" in columns:
                    conv_row["labels"] = self.convert_frame_text(frame_text=columns["frames"])

                if "language" in columns:
                    conv_row["language"] = columns["language"]
                else:
                    logger.warning("No langauge information available in row {}", _id)
                    conv_row["language"] = "n/a"

                self.conv_data.append(conv_row)
            except KeyError:
                logger.opt(exception=True).error("Failed to read {}", _id)
        logger.success("Converted all {} data points!", len(df))