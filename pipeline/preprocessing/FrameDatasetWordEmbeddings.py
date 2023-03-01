from pathlib import Path
from typing import Dict
from argparse import ArgumentParser

from urllib.request import urlretrieve
from urllib3.util.url import parse_url
from zipfile import ZipFile

import torch
from loguru import logger
from nltk import word_tokenize, sent_tokenize
from pandas import DataFrame

from config_languages import LANGUAGE_ABBREVIATION_TO_FULL, get_pure_language_abbreviation
from pipeline.preprocessing.FrameDatasetInterface import FrameDataset


class FrameDatasetForPlainNeuralNets(FrameDataset):
    w2v: Dict[str, Dict[str, torch.Tensor]] = dict()

    dataset_parser = ArgumentParser(prog="Dataset-parser", add_help=False, allow_abbrev=True)
    dataset_parser.add_argument("--w2v_path_cache", action="store", type=Path, default=Path(".out/data/w2v/"),
                                required=False)
    # see https://github.com/facebookresearch/MUSE for the default
    # We provide multilingual embeddings and ground-truth bilingual dictionaries.
    # These embeddings are fastText embeddings that have been aligned in a common space.
    dataset_parser.add_argument("--w2v_url", action="store", type=str,
                                default="https://dl.fbaipublicfiles.com/arrival/vectors/")
    dataset_parser.add_argument("--w2v_file_lang_pattern", action="store", type=str, default="wiki.multi.{}.vec",
                                help="langauge-insensitive if {} is missing in the string")
    dataset_parser.add_argument("--unzip_downloaded_files", action="store_true", default=False, required=False)
    dataset_parser.add_argument("--max_length", action="store", default=9999, type=int, required=False)
    dataset_parser.add_argument("--split_into_sentences", action="store_true", default=False, required=False)

    def __init__(self, w2v_path_cache: Path, w2v_url: str, w2v_file_lang_pattern: str,
                 max_length: int, split_into_sentences: bool = False, unzip_downloaded_files: bool = False):
        super().__init__()

        self.w2v_path_cache_root = w2v_path_cache
        self.w2v_url = w2v_url
        self.w2v_host = parse_url(self.w2v_url).host
        self.w2v_path_cache = self.w2v_path_cache_root.joinpath(self.w2v_host)
        if not self.w2v_path_cache.exists():
            logger.warning("Until yet, there are no cached word embeddings at \"{}\"", self.w2v_path_cache)
            self.w2v_path_cache.mkdir(exist_ok=False, parents=True)
        self.w2v_file_lang_pattern = w2v_file_lang_pattern
        self.unzip_downloaded_files = unzip_downloaded_files
        self.max_length = max_length
        self.split_into_sentences = split_into_sentences

    def init_data(self, df: DataFrame) -> None:
        logger.debug("OK, let's convert the {} data points into a neural-net-readable format", len(df))

        for _id, columns in df.iterrows():
            logger.trace("Let's process {}", _id)

            language = get_pure_language_abbreviation(language_tag=columns.get("language", "en"))
            w2v_key = "{}-{}".format(self.w2v_host, language)
            if w2v_key in FrameDatasetForPlainNeuralNets.w2v:
                w2v = FrameDatasetForPlainNeuralNets.w2v[w2v_key]
                logger.trace("Use the {}-w2v having {} entries", LANGUAGE_ABBREVIATION_TO_FULL[language], len(w2v))
            else:
                logger.info("We have to load the {} word embeddings first", LANGUAGE_ABBREVIATION_TO_FULL[language])
                w2v_file = self.w2v_path_cache.joinpath(
                    self.w2v_file_lang_pattern.format(language)
                    if "{}" in self.w2v_file_lang_pattern else self.w2v_file_lang_pattern
                )
                logger.debug("Check the existence of \"{}\"", w2v_file.name)
                if not w2v_file.exists():
                    if self.unzip_downloaded_files and w2v_file.suffix != ".zip":
                        download_file_name = "".join((w2v_file.stem, ".zip"))
                    else:
                        download_file_name = w2v_file.name

                    if self.w2v_url.startswith("https://dl.fbaipublicfiles.com/arrival/vectors"):
                        logger.debug("In \"{}\", same naming conventions are not conventional... "
                                     "let's prevent downloading bugs!", self.w2v_url)
                        download_file_name = download_file_name.replace(".ge.", ".de.").replace(".gr.", ".el.")

                    logger.warning("We have to download \"{}\" first!", download_file_name)
                    url = "{}{}{}".format(
                        self.w2v_url,
                        "" if self.w2v_url.endswith("/") else "/",
                        download_file_name
                    )
                    dest, http_resp = urlretrieve(
                        url=url,
                        filename=w2v_file.parent.joinpath(download_file_name)
                        if self.unzip_downloaded_files else w2v_file
                    )
                    logger.success("Successfully downloaded the word embeddings to: {} ({})", dest, http_resp)

                    if self.unzip_downloaded_files:
                        with ZipFile(file=dest, mode="r") as zip_ref:
                            filename_list = zip_ref.namelist()
                            logger.debug("Found {} files in the archive: {}",
                                         len(filename_list), ", ".join(filename_list))
                            if len(filename_list) == 0:
                                raise AttributeError(
                                    "Archive is empty, please check your \"w2v_url\"-param: {}".format(self.w2v_url)
                                )
                            elif len(filename_list) == 1:
                                member = filename_list[0]
                            else:
                                member = [m for m in filename_list if "300" in m]
                                member = member[0] if len(member) >= 1 else filename_list[-1]
                                logger.warning("File to extract from \"{}\" is ambiguous! Select: {}", dest, member)
                            unzipped_file = zip_ref.extract(member=member, path=w2v_file.parent)
                        logger.success("Successfully unzipped \"{}", unzipped_file)
                        renamed_file = Path(unzipped_file).rename(w2v_file)
                        logger.info("Renaming done: \"{}\"->\"{}\"", unzipped_file, renamed_file)

                try:
                    lines = w2v_file.read_text(encoding="utf-8", errors="ignore").split(sep="\n")
                    logger.info("Read {} lines from \"{}\"", len(lines), w2v_file.name)

                    w2v = {sep_line[0][2:] if sep_line[0].startswith("b'") else sep_line[0]:
                               torch.tensor(data=[float(v) for v in sep_line[1:]], dtype=torch.float, device="cpu")
                           for line in lines if len(sep_line := line.strip(" \t\r'").split(sep=" ")) >= 100}
                    logger.success("Have {} embeddings now :) ({}%)",
                                   len(w2v), round(100 * len(w2v) / len(lines), 1))
                    FrameDatasetForPlainNeuralNets.w2v[w2v_key] = w2v
                except IOError:
                    logger.opt(exception=True).error("Can't load \"{}\" -- 0 word embeddings", w2v_file.absolute())
                    w2v = dict()
                except ValueError:
                    logger.opt(exception=True).error("Can't load \"{}\" -- 0 word embeddings", w2v_file.absolute())
                    w2v = dict()

            try:
                if self.split_into_sentences:
                    conv_data_point = \
                        {
                            "input_tensors": torch.nn.utils.rnn.pad_sequence(
                                sequences=[
                                    torch.stack(
                                        tensors=(
                                                    t := [
                                                        w2v.get(
                                                            word, torch.zeros((300,), dtype=torch.float, device="cpu")
                                                        )
                                                        for word in word_tokenize(
                                                            text=sent,
                                                            language=LANGUAGE_ABBREVIATION_TO_FULL.get(
                                                                columns["language"], "english"
                                                            )
                                                        )]
                                                )[:min(self.max_length, len(t))],
                                        dim=0
                                    )
                                    for sent in sent_tokenize(
                                        text=columns["text"],
                                        language=LANGUAGE_ABBREVIATION_TO_FULL.get(columns["language"], "english")
                                    )
                                ],
                                batch_first=True,
                                padding_value=-1.
                            )
                        }
                else:
                    conv_data_point = \
                        {
                            "input_tensors": torch.stack(
                                tensors=(
                                            t := [
                                                w2v.get(word, torch.zeros((300,), dtype=torch.float, device="cpu"))
                                                for word in word_tokenize(
                                                    text=columns["text"],
                                                    language=LANGUAGE_ABBREVIATION_TO_FULL.get(columns["language"],
                                                                                               "english")
                                                )]
                                        )[:min(self.max_length, len(t))],
                                dim=0
                            )
                        }

                if "frames" in columns:
                    conv_data_point["labels"] = self.convert_frame_text(frame_text=columns["frames"])
                if "language" in columns:
                    conv_data_point["language"] = columns["language"]
                else:
                    logger.warning("No langauge information available in row {}", _id)
                    conv_data_point["language"] = "n/a"

                self.conv_data.append(conv_data_point)
                logger.trace("Appended following datapoint: {}", conv_data_point)
            except KeyError:
                logger.opt(exception=True).error("Failed to read {}", _id)
            except TypeError:
                logger.opt(exception=True).warning("Empty text sample: {}", _id)
            except RuntimeError:
                logger.opt(exception=True).error("Tensor-conversion-issue in: {}", _id)
        logger.success("Converted all {} data points!", len(df))