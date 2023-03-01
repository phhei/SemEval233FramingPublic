import copy
import math
import json
import sys
from typing import Optional, List, Union, Tuple, Dict

import numpy
import torch
from tqdm import tqdm

import pipeline.encoder.TransformerEncoder
import pipeline.output.Losses
from baselines.st2 import make_dataframe
from pipeline.Pipeline import Pipeline
from pipeline.aggregator.AggregatorInterface import Aggregator
from pipeline.encoder.EncoderInterface import Encoder
from pipeline.preprocessing.FrameDatasetInterface import FrameDataset
from pipeline.output import PredictionUtils
from scorers.scorer_subtask_2 import read_frame_list_from_file, evaluate
from config_languages import LANGUAGE_FULL_TO_ABBREVIATION
from const import ARG_STR_TO_MODULE
from utils import MLJsonSerializer

from argparse import ArgumentParser
from loguru import logger
from pathlib import Path

import pandas


def merge_database(old: Optional[pandas.DataFrame], new: pandas.DataFrame) -> pandas.DataFrame:
    if old is None:
        return new

    return pandas.concat(objs=[old, new], axis="index", ignore_index=False, verify_integrity=True, sort=False)


def convert_arg_string_to_module(arg_sep: str, module_kind: str):
    if module_kind not in ARG_STR_TO_MODULE:
        raise AttributeError("Wrong module-kind declaration")

    selected_module = ARG_STR_TO_MODULE[module_kind][arg_sep[0]][0]

    if selected_module is None:
        logger.warning("Your string \"{}\" maps to None!", arg_sep)
        return None

    arg_sep_parser: ArgumentParser = ARG_STR_TO_MODULE[module_kind][arg_sep[0]][1]

    if len(arg_sep) == 1:
        logger.info("Call the {} {} without any specific args!", module_kind, arg_sep[0])
        try:
            return selected_module(**vars(arg_sep_parser.parse_args([])))
        except TypeError:
            logger.opt(exception=True).critical("{} was wrongly called", module_kind)
            raise AttributeError("Please check your script parameters ({}) at -m".format(sys.argv))

    logger.debug("Having a module {} with {} arguments to parse",
                 selected_module, len(arg_sep)-1)

    return selected_module(**vars(arg_sep_parser.parse_args(arg_sep[1:])))


def convert_arg_string_to_aggregator(arg_crossed: str):
    return convert_arg_string_to_module(arg_sep=arg_crossed.split("#"), module_kind="aggregator")


def convert_arg_string_to_encoder(arg_crossed: str, predict_amount_of_frames: bool = False):
    if predict_amount_of_frames and "--predict_number_of_frames" not in arg_crossed:
        arg_crossed_ect = arg_crossed + "#--predict_number_of_frames"
    else:
        arg_crossed_ect = arg_crossed
    return convert_arg_string_to_module(arg_sep=arg_crossed_ect.split("#"), module_kind="encoder")


def convert_arg_string_to_dataset(arg_crossed: str):
    return convert_arg_string_to_module(arg_sep=arg_crossed.split("#"), module_kind="preprocessing")


if __name__ == "__main__":
    arg_parser = ArgumentParser(prog="SemEval23-Task3-Subtask2-FramePredictor", allow_abbrev=True)
    arg_parser.add_argument(
        "-l", "--languages", action="store", nargs="+", default=["en"],
        help="List of languages which should be used for training, dev and testing. "
             "Please use the langauge abbreviations, e.g. \"en\" for english"
    )
    arg_parser.add_argument(
        "--new_train_dev", action="store", nargs="*", default=None,
        help="Merge the train and dev split to a new split. If you don't give further specifications, "
             "we randomly split 80-20. However, you can define your own splits in the following format: "
             "<lang>:<percent_in_train>:<percent_in_dev> (for each language that you want to have in train/dev)"
    )
    arg_parser.add_argument(
        "--skip_inference", action="store_true", default=False,
        help="Skips the inference (testing)"
    )
    arg_parser.add_argument(
        "--skip_training", action="store_true", default=False,
        help="Skips the training (model parameter adaption)"
    )
    arg_parser.add_argument(
        "-m", "--modules", action="append", default=[], nargs="+", required=True,
        help="Modules that should be used. For each module, use a new \"--modules\"-command. "
             "Each \"--modules\" must be specified by 2-3 arguments: the Preprocessor, the encoder "
             "and optionally the aggregator. Please have a look at const.ARG_STR_TO_MODULE to get to know how "
             "each module part can be specified. "
             "ATTENTION! While the different module-parts (Preprocessor, Encoder, Aggregator) should be seperated "
             "by whitespaces (standard), additional arguments for each module-part should be seperated by \"#\". "
             "For example, if you want to define further arguments for the aggregator \"PoolingSmoothMax\", "
             "you have to write \"PoolingSmoothMax#--exponent#3.#...\""
    )
    arg_parser.add_argument(
        "-agg", "--final_aggregator", action="store", default="Average", type=str,
        help="The final aggregator aggregating the single predictions from each module. "
             "Please have a look at const.ARG_STR_TO_MODULE to get to know to define a specific aggregator. "
             "Further specifications are again \"#\"-seperated"
    )
    arg_parser.add_argument(
        "--predict_amount_of_frames", action="store_true", default=False,
        help="If you want the modules predict the number of frames for each sample sample, set this flag to True"
    )
    arg_parser.add_argument(
        "-lr", "--learning_rate", action="store", type=float, default=5e-4,
        help="TRAINING] The (maximum) learning rate"
    )
    arg_parser.add_argument(
        "-bs", "--batch_size", action="store", type=int, default=8, choices=list(range(1, 130)),
        help="TRAINING] The batch size"
    )
    arg_parser.add_argument(
        "--compute_loss_for_each_module", action="store_true", default=False,
        help="TRAINING] Applies a loss for each module, "
             "i.e., the parameters of a module is adapted directly on its predictions(, too)"
    )
    arg_parser.add_argument(
        "--compute_final_loss", action="store_true", default=False,
        help="TRAINING] Applies a loss on the final aggregated prediction. If you don't set this param, we have just a "
             "standard ensemble approach. If you set this param, you might have a huge RAM-consumption."
    )
    arg_parser.add_argument(
        "--early_stopping", action="store_true", default=False,
        help="TRAINING] If you want to use early stopping, set this flag to True. "
             "(stopping the training when the performance drops on the dev-split and restores the best params)"
    )
    arg_parser.add_argument(
        "--max_epochs", action="store", default=8, type=int, choices=list(range(1, 16)),
        help="TRAINING] The number of epochs. If you set \"--early_stopping\", the training process might stop before"
    )
    arg_parser.add_argument(
        "-v", "--verbose", action="count", default=False,
        help="Enables more debugging/ dumping more fine-grained information/ files"
    )
    arg_parser.add_argument(
        "--train_on_dev", action="store_true", default=False,
        help="INFERENCE] uses the dev set for training at the end, too"
    )
    arg_parser.add_argument(
        "--min_frame_selection", action="store", default=1, type=int,
        help="INFERENCE] The minimum number of frames that should be assigned (predict) to each sample"
    )
    arg_parser.add_argument(
        "--max_frame_selection", action="store", default=8, type=int,
        # there is one sample with 10 frames und 5 with 9 frames across all languages (dev),
        # so I guess max. 8 frames sounds reasonable :)
        help="INFERENCE] The maximum number of frames that should be assign (predict) to each sample"
    )
    arg_parser.add_argument(
        "--root_save_path", action="store",
        default=Path(".out"),
        type=Path,
        help="The path where the results should be saved. Will be extended by a try-subfolder (incremental)"
    )

    logger.info("Let's start the journey!")

    args = arg_parser.parse_args()

    args.root_save_path = args.root_save_path.joinpath("try-{}".format(sum(1 for _ in args.root_save_path.glob(pattern="try-*"))))

    args.root_save_path.mkdir(parents=True, exist_ok=True)

    if args.verbose >= 2:
        logger.add(sink=args.root_save_path.joinpath("log.txt"),
                   level="INFO",
                   colorize=False,
                   diagnose=True,
                   delay=True,
                   encoding="utf-8")
    elif not args.verbose:
        logger.remove()
        logger.add(sink=sys.stdout, level="INFO", colorize=True)

    try:
        args.root_save_path.joinpath("args.txt").write_text("\n".join(sys.argv), encoding="utf-8")
    except IOError:
        logger.opt(exception=True).warning("Failed to write the params ({})", len(sys.argv)-1)

    base_data_path = Path("data")
    data_folder = {
        "train": "train-articles-subtask-2",
        "dev": "dev-articles-subtask-2",
        "test": "test-articles-subtask-2"
    }

    labels_file_paths = {
        "train": "train-labels-subtask-2.txt",
        "dev": "dev-labels-subtask-2.txt",
        "test": "test-labels-subtask-2.txt"
    }

    collected_data = {
        "train": None, "dev": None, "test": None
    }
    for language in args.languages:
        logger.info("Receiving the data for {}", language)
        if len(language) > 2 and not language.endswith("2en"):
            try:
                language = LANGUAGE_FULL_TO_ABBREVIATION[language]
                logger.debug("The abbreviation of your language is \"{}\"", language)
            except KeyError:
                logger.opt(exception=False).error("You input an unknown langauge \"{}\", please use the following: {}",
                                                  language, " AND/OR ".join(LANGUAGE_FULL_TO_ABBREVIATION.keys()))
                continue

        for split in (("train", "dev") if args.skip_inference else ("train", "dev", "test")):
            input_folder = base_data_path.joinpath(language, data_folder[split])
            labels_file = base_data_path.joinpath(language, labels_file_paths[split])
            if not input_folder.exists():
                logger.warning("\"{}\" not exist!", input_folder, labels_file)
                continue

            df = make_dataframe(
                input_folder=input_folder,
                labels_folder=str(labels_file.absolute()) if labels_file.exists() else None
            )
            df["language"] = language

            logger.info("Load {} samples from \"{}\"", len(df), input_folder)

            if not labels_file.exists():
                logger.warning("We found no labels ({}->{}) - so this is test data!", split, labels_file.absolute())
                split = "test"

            try:
                collected_data[split] = merge_database(old=collected_data[split], new=df)
                logger.debug("Merged the loaded dataframe into the split {}", split)
            except ValueError:
                logger.opt(exception=True).error(
                    "Failed to merge the new data into the {}-split. "
                    "We consider only the data of {} ({}) for this split!",
                    split, input_folder.name, language
                )
                collected_data[split] = df

            logger.trace("Close split {} for language \"{}\"", split, language)
        logger.debug("Language \"{}\" is complete", language)

    logger.info("Loaded all {} languages -- now, lets check the splits!", len(args.languages))

    if not args.skip_inference and collected_data["test"] is None:
        logger.warning("You don't have any test data! We have to change this...")
        if collected_data["dev"] is not None and collected_data["train"] is not None:
            logger.warning("... by dividing the dev-data ({} samples)", len(collected_data["train"]))
            train_df_split: pandas.DataFrame = \
                collected_data["train"].sample(frac=.15, replace=False, ignore_index=False, random_state=42)
            collected_data["train"] = \
                collected_data["train"].drop(index=train_df_split.index, inplace=False)
            collected_data["test"] = collected_data["dev"]
            collected_data["dev"] = train_df_split
            logger.debug("train: {}/ dev: {}/ test: {}",
                            len(collected_data["train"]), len(collected_data["dev"]), len(collected_data["test"]))
        else:
            logger.error("You have no data at all/ necessary splits are missing!")
            exit(-1)
    elif not args.skip_inference:
        logger.success("We collected {} test data", len(collected_data["test"]))

    if collected_data["dev"] is None:
        logger.warning("You don't have any dev data!")
        if collected_data["train"] is not None:
            dev_df: pandas.DataFrame = \
                collected_data["train"].sample(frac=.2, replace=False, ignore_index=False, random_state=43)
            collected_data["dev"] = dev_df
            collected_data["train"] = collected_data["train"].drop(index=dev_df.index, inplace=False)
            if not args.skip_inference:
                logger.debug("train: {}/ dev: {}/ test: {}",
                             len(collected_data["train"]), len(collected_data["dev"]), len(collected_data["test"]))
        else:
            logger.error("You have no (training) data at all!")
            exit(-1)

    if collected_data["train"] is None:
        logger.warning("You don't have any training data!")
    else:
        logger.success("We collected {} training data", len(collected_data["train"]))

    if args.new_train_dev is not None:
        logger.info("OK, now you want to have another train({})-dev({})-split!",
                    len(collected_data["train"]), len(collected_data["dev"]))
        merged_train_dev = merge_database(old=collected_data["train"], new=collected_data["dev"])

        if len(args.new_train_dev) == 0:
            logger.warning("No further specifications on \"--new_train_dev\", split randomly into {}-{} samples",
                           int(len(merged_train_dev)*.8), len(merged_train_dev)-int(len(merged_train_dev)*.8))
            collected_data["train"] = merged_train_dev.sample(frac=.8, ignore_index=False)
            collected_data["dev"] = merged_train_dev.drop(index=collected_data["train"].index, inplace=False)
        else:
            logger.debug("OK, having following wishes: {}", " and ".join(args.new_train_dev))
            collected_data["train"] = None
            collected_data["dev"] = None
            for config in args.new_train_dev:
                config_splits = config.strip().split(sep=":")
                if len(config_splits) != 3:
                    logger.warning("Invalid format in \"--new_train_dev\" -- ignore: \"{}\"", config)
                else:
                    config_lang, percent_train, percent_dev = config_splits
                    merged_train_dev_lang = merged_train_dev[merged_train_dev["language"] == config_lang]
                    logger.info("Found {} samples of {}", len(merged_train_dev_lang), config_lang)
                    if len(merged_train_dev_lang) >= 1:
                        try:
                            collected_data["train"] = merge_database(
                                old=collected_data["train"],
                                new=merged_train_dev_lang[:int((float(percent_train)/100)*len(merged_train_dev_lang))]
                            )
                            collected_data["dev"] = merge_database(
                                old=collected_data["dev"],
                                new=merged_train_dev_lang[
                                    int((float(percent_train)/100)*len(merged_train_dev_lang)):
                                    int((float(percent_train+percent_dev)/100)*len(merged_train_dev_lang))
                                    ]
                            )
                        except ValueError:
                            logger.opt(exception=True).error("Skip \"{}\" - not parsable", config)

    logger.debug("Now, let's have a look on the given modules")

    if len(args.modules) == 0:
        logger.info("You don't defined any modules")
        exit(-1)

    datasets_and_modules: List[Union[Tuple[Tuple[FrameDataset, FrameDataset, FrameDataset], Encoder],
                                     Tuple[Tuple[FrameDataset, FrameDataset, FrameDataset], Encoder, Aggregator]]] = []

    for module_args in args.modules:
        logger.debug("Create a module receiving {} args: {}", len(module_args), "+".join(module_args))
        if len(module_args) not in (2, 3):
            logger.error("One module description requires 2 or 3 part descriptions, but you gave {}", len(module_args))

        dataset_train: FrameDataset = convert_arg_string_to_dataset(module_args[0])
        dataset_dev: FrameDataset = copy.deepcopy(dataset_train)
        dataset_test: FrameDataset = copy.deepcopy(dataset_dev)
        dataset_train.init_data(collected_data["train"])
        dataset_dev.init_data(collected_data["dev"])
        if not args.skip_inference:
            dataset_test.init_data(collected_data["test"])
        encoder: Encoder = convert_arg_string_to_encoder(arg_crossed=module_args[1],
                                                         predict_amount_of_frames=args.predict_amount_of_frames)
        aggregator: Optional[Aggregator] =\
            convert_arg_string_to_aggregator(module_args[2]) if len(module_args) >= 3 else None

        if aggregator is None:
            logger.info("Add a module without an aggregator: {}->{} ({} params)",
                        dataset_train, (end_str := str(encoder))[:min(len(end_str), 100)],
                        sum(p.numel() for p in encoder.parameters()))
            datasets_and_modules.append(
                (
                    (
                        dataset_train, dataset_dev, dataset_test
                    ),
                    encoder
                )
            )
        else:
            logger.info("Add an module including an own aggregator: {}->*{} ({} params)->{}",
                        dataset_train, encoder, sum(p.numel() for p in encoder.parameters()), aggregator)
            datasets_and_modules.append(
                (
                    (
                        dataset_train, dataset_dev, dataset_test
                    ),
                    encoder,
                    aggregator
                )
            )
        logger.debug(datasets_and_modules[-1])

    having_transformer = any(map(lambda dm: isinstance(dm[1], pipeline.encoder.TransformerEncoder.Huggingface),
                                 datasets_and_modules))

    logger.success("Finished the initialization of all {} module pipes! ({} one or more LLMs)",
                   len(datasets_and_modules), "including" if having_transformer else "excluding")
    final_aggregator: Aggregator = convert_arg_string_to_aggregator(arg_crossed=args.final_aggregator)

    if args.verbose >= 1 and args.predict_amount_of_frames:
        if having_transformer:
            module_loss = pipeline.output.Losses.verbose_loss_reduced_count_frames
            final_loss = pipeline.output.Losses.verbose_loss_count_frames
        else:
            module_loss = pipeline.output.Losses.verbose_loss_count_frames
            final_loss = pipeline.output.Losses.verbose_loss_count_frames
    elif args.verbose >= 1 and not args.predict_amount_of_frames:
        if having_transformer:
            module_loss = pipeline.output.Losses.verbose_loss_reduced
            final_loss = pipeline.output.Losses.verbose_loss
        else:
            module_loss = pipeline.output.Losses.verbose_loss
            final_loss = pipeline.output.Losses.verbose_loss
    elif args.verbose == 0 and args.predict_amount_of_frames:
        if having_transformer:
            module_loss = pipeline.output.Losses.loss_reduced_count_frames
            final_loss = pipeline.output.Losses.loss_count_frames
        else:
            module_loss = pipeline.output.Losses.loss_count_frames
            final_loss = pipeline.output.Losses.loss_count_frames
    else:
        if having_transformer:
            module_loss = pipeline.output.Losses.loss_reduced
            final_loss = pipeline.output.Losses.loss
        else:
            module_loss = pipeline.output.Losses.loss
            final_loss = pipeline.output.Losses.loss

    framework = Pipeline(
        datasets_and_modules=datasets_and_modules,
        final_aggregator=final_aggregator,
        loss_for_each_module=module_loss if args.compute_loss_for_each_module else None,
        loss_for_final_aggregation=final_loss if args.compute_final_loss or not args.compute_loss_for_each_module else None,
        verbose=args.verbose >= 1
    )

    logger.success("Successfully initialized the pipeline: {}", framework)

    optimizer = torch.optim.Adam(params=framework.parameters(), lr=args.learning_rate)
    logger.debug("Have following optimizer now: {} ({})",
                 optimizer, "/".join(map(lambda pg: "-".join(pg.keys()), optimizer.param_groups)))


    def loop(
            f_epoch: int,
            f_split="train",
            with_param_adaption: Optional[bool] = None,
            threshold: Union[bool, float, list[float]] = False
    ) -> Dict[str, Union[float, Path, List[float], List[str]]]:
        total_size = len(collected_data[f_split])
        total_steps = math.ceil(total_size / args.batch_size)
        logger.debug("Start {} loop in the {}. epoch ({} samples)", f_split, f_epoch + 1, total_size)

        all_pred = []
        all_truth = []
        loss_history = []

        for i, start_index in tqdm(iterable=enumerate(range(0, total_size, args.batch_size)),
                                   total=total_steps,
                                   unit="batch"):
            end_index = min(total_size, start_index + args.batch_size)
            logger.trace("{}. batch: {}-{}", i, start_index, end_index)
            pred, truth, loss = framework(
                split=f_split,
                enable_backpropagation=with_param_adaption,
                module_loss_enabled=(
                    (.1 if f_epoch == 1 else 1 - i / total_steps) if args.compute_final_loss else True
                ) if args.compute_loss_for_each_module and f_epoch <= 1 else False,
                final_aggregation_loss_enabled=(
                    i / total_steps if args.compute_loss_for_each_module and f_epoch == 0 else
                    (0.5 + 0.5 * i / total_steps if f_epoch == 0 else True)
                ) if args.compute_final_loss else False,
                index=(start_index, end_index)
            )

            if with_param_adaption:
                if loss is None:
                    logger.error("{}-> {}. epoch: You want to adapt your parameters (train), "
                                 "but we don't have any loss!", f_split, f_epoch+1)
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.trace("Backpropagation successful!")
                    loss_history.append(loss.cpu().item())
            else:
                logger.trace("No backpropagation")

            all_pred.append(pred.detach().cpu())
            all_truth.append(None if truth is None else truth.detach().cpu())

        ret: Dict[str, Path, Optional[Union[str, float, List[float]]]] = {
            "{}_pred_history".format(f_split): [p.tolist() for p in all_pred],
            "{}_loss_history".format(f_split): loss_history,
            "{}_loss_max".format(f_split): max(loss_history) if len(loss_history) >= 1 else 0,
            "{}_loss_min".format(f_split): min(loss_history) if len(loss_history) >= 1 else 0,
            "{}_loss_avg".format(f_split): sum(loss_history) / len(loss_history) if len(loss_history) >= 1 else 0,
            "{}_weight_final_agg".format(f_split): "n/a" if final_aggregator.module_weight_vector is None else
            final_aggregator.module_weight_vector.cpu().tolist(),
            "{}_path".format(f_split): args.root_save_path.joinpath("epoch-{}".format(f_epoch), f_split)
        }
        ret["{}_path".format(f_split)].mkdir(parents=True, exist_ok=True)

        if args.early_stopping and with_param_adaption:
            logger.info("we have to store the model-pipeline for the purpose of early stopping ({})",
                        ret["{}_path".format(f_split)])
            framework.save_modules(path=ret["{}_path".format(f_split)])

        try:
            numpy_pred: Optional[numpy.ndarray] = torch.concat(tensors=all_pred, dim=0).numpy()
            numpy_truth: Optional[numpy.ndarray] = torch.concat(tensors=all_truth, dim=0).numpy() \
                if all(map(lambda t: t is not None, all_truth)) else None
        except RuntimeError:
            logger.opt(exception=True).error(
                "{}->{}. epoch: Something doesn't fit together! (received following shapes: {})",
                f_split, f_epoch+1,
                "/".join(map(lambda p: str(p.shape) if isinstance(p, torch.Tensor) else str(type(p)), all_pred))
            )
            if len(all_pred) == 0:
                logger.critical("You don't have a single prediction! (frame probability)")
            numpy_pred: Optional[numpy.ndarray] = None
            numpy_truth: Optional[numpy.ndarray] = None

        if isinstance(threshold, bool) and threshold:
            if numpy_truth is None:
                logger.warning("{}->{}. epoch: You want to determine the optimal threshold without knowing the "
                               "ground truth!", f_split, f_epoch+1)
            elif numpy_pred is None:
                logger.warning("{}->{}. epoch: You want to determine the optimal threshold without having a "
                               "single sample!", f_split, f_epoch+1)
            else:
                ret["{}_optimal_threshold".format(f_split)] = [
                    PredictionUtils.compute_optimal_threshold(
                        predicted=numpy_pred[:, i+int(args.predict_amount_of_frames)],
                        reference=numpy_truth[:, i]
                    ) for i in range(numpy_truth.shape[-1])
                ]
                ret["{}_used_threshold".format(f_split)] = ret["{}_optimal_threshold".format(f_split)]
        elif isinstance(threshold, bool):
            ret["{}_used_threshold".format(f_split)] = .5
        else:
            ret["{}_used_threshold".format(f_split)] = threshold

        try:
            class_offset = numpy.fromiter(iter=ret["{}_used_threshold".format(f_split)], dtype=float)
        except TypeError:
            logger.opt(exception=True).debug("\"threshold\"-param ({}) was not converted in an array -- "
                                             "we don't define any class offset", threshold)
            class_offset = None
        binary_pred = [PredictionUtils.map_count_prediction_to_binary_prediction(
            logits=single_pred,
            class_offset=class_offset,
            sigmoid_logits=False
        ) if args.predict_amount_of_frames else PredictionUtils.map_logits_to_binary_prediction(
            logits=single_pred,
            threshold=ret["{}_used_threshold".format(f_split)],
            min_selection=args.min_frame_selection,
            max_selection=args.max_frame_selection,
            sigmoid_logits=False
        ) for single_pred in numpy_pred]
        ret["{}_pred_text".format(f_split)] = \
            PredictionUtils.map_binary_tensors_to_text_prediction(logits=binary_pred)

        logger.info(
            "Got following predictions ({}x):\n{} (for the first 10 samples)",
            len(ret["{}_pred_text".format(f_split)]),
            "\n".join(ret["{}_pred_text".format(f_split)][:min(10, len(ret["{}_pred_text".format(f_split)]))])
        )

        if numpy_pred is None or numpy_truth is None:
            logger.info("We don't have a ground truth, no metrics available!")
        else:
            logger.debug("Let's calculate the metrics having {} samples", len(binary_pred))

            ret["{}_ground_truth_text".format(f_split)] = \
                PredictionUtils.map_binary_tensors_to_text_prediction(logits=numpy_truth)

            df_f = collected_data[f_split]
            pred_labels = {(df_f.iloc[i].get("language", "unknown"), i): single_bin_pred.split(sep=",")
                           for i, single_bin_pred in enumerate(ret["{}_pred_text".format(f_split)])}
            gold_labels = {(df_f.iloc[i].get("language", "unknown"), i): single_bin_pred.split(sep=",")
                           for i, single_bin_pred in enumerate(ret["{}_ground_truth_text".format(f_split)])}
            frame_classes = read_frame_list_from_file(file_full_name="scorers/frames_subtask2.txt")

            macro_f1, micro_f1 = evaluate(pred_labels=pred_labels, gold_labels=gold_labels, CLASSES=frame_classes)
            logger.success("Successfully calculated the F1-scores for {}->{} (using thresholds {}): {}% (F1-macro)",
                           f_split, f_epoch, ret["{}_used_threshold".format(f_split)], str(round(100 * macro_f1, 1)))

            ret["{}_f1_macro".format(f_split)] = macro_f1
            ret["{}_f1_micro".format(f_split)] = micro_f1

            languages = set(df_f["language"])
            logger.debug("OK, evaluating {} languages separately now!")
            for lang_eval in languages:
                macro_f1, micro_f1 = evaluate(
                    pred_labels={k[1]:v for k, v in pred_labels.items() if k[0] == lang_eval},
                    gold_labels={k[1]:v for k, v in gold_labels.items() if k[0] == lang_eval},
                    CLASSES=frame_classes
                )
                logger.info("For the language \"{}\", we got {}% (F1-macro)", lang_eval, str(round(100 * macro_f1, 1)))
                ret["{}_{}_f1_macro".format(f_split, lang_eval)] = macro_f1
                ret["{}_{}_f1_micro".format(f_split, lang_eval)] = micro_f1

        logger.success(
            "Finished the {}. epoch of {} ({})",
            f_epoch+1, f_split,
            "-" if with_param_adaption is None else ("Parameters updated" if with_param_adaption else "inference")
        )

        try:
            with ret["{}_path".format(f_split)].joinpath("stats.json").open(mode="w",
                                                                            encoding="utf-8",
                                                                            errors="ignore") as f_stats_stream:
                json.dump(obj=ret, fp=f_stats_stream, indent="\t", skipkeys=True, sort_keys=True, cls=MLJsonSerializer)
        except IOError:
            logger.opt(exception=True).warning("Failed to write the stats in {}",
                                               ret["{}_path".format(f_split)].absolute())
        except TypeError:
            logger.opt(exception=True).warning("Failed to json-write the dictionary: {}", ret)

        if args.verbose >= 1:
            try:
                f_df = collected_data[f_split]
                f_df = f_df.drop(
                    columns=set(f_df.columns).intersection({"text", "frames", "language"}),
                    inplace=False,
                    errors="ignore"
                )
                f_df["prediction"] = ret["{}_pred_text".format(f_split)]
                f_df.to_csv(path_or_buf=ret["{}_path".format(f_split)].joinpath("predictions.txt"),
                            sep="\t",
                            index=True,
                            index_label="article_id",
                            encoding="utf-8",
                            header=False)
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to write the predictions: {}",
                    ret.get("{}_pred_text".format(f_split), "Reason: no predictions available")
                )

        return ret

    final_dict: Dict[int, Dict[str, Union[float, List[float], List[str]]]] = dict()
    optimal_epoch = -1

    for epoch in range(args.max_epochs):
        logger.info("Start the {}. epoch", epoch+1)
        if args.skip_training:
            logger.warning("Skip the training!")
        else:
            train_stats = loop(
                f_epoch=epoch,
                f_split="train",
                with_param_adaption=True,
                threshold=final_dict[epoch-1].get("dev_optimal_threshold", final_dict[epoch-1]["dev_used_threshold"]) if epoch >= 1 else True
            )

            logger.debug("New train stuff for the final dict: {} keys", len(list(train_stats.keys())))
            final_dict[epoch] = train_stats

        dev_stats = loop(
            f_epoch=epoch,
            f_split="dev",
            with_param_adaption=False,
            threshold=True
        )

        logger.debug("New dev stuff for the final dict: {} keys", len(list(dev_stats.keys())))
        if epoch in final_dict:
            final_dict[epoch].update(dev_stats)
        else:
            final_dict[epoch] = dev_stats

        if args.early_stopping and not args.skip_training and epoch >= 1:
            try:
                logger.debug("we have to check whether we loosed performance!")
                if final_dict[epoch-1]["dev_f1_macro"]*final_dict[epoch-1]["dev_f1_micro"] > \
                        final_dict[epoch]["dev_f1_macro"]*final_dict[epoch]["dev_f1_micro"]:
                    checkpoint_path = final_dict[epoch-1]["train_path"]
                    logger.warning("Yes, epoch {} generalizes better then epoch {}! Load \"{}\"-weights",
                                   epoch, epoch+1, checkpoint_path)
                    framework.load_modules(checkpoint_path)
                    optimal_epoch = epoch-1
                    break
            except KeyError:
                logger.opt(exception=True).warning("Can't determine the stats for early stopping - "
                                                   "skipping this feature!")
                optimal_epoch = epoch
                continue
        else:
            optimal_epoch = epoch

        logger.success("Successfully finished the {}. epoch", epoch+1)

    if args.skip_inference:
        logger.info("Inference is skipped, nothing left to do :)")
    else:
        if args.train_on_dev:
            dev_stats = loop(
                f_epoch=999,
                f_split="dev",
                with_param_adaption=True,
                threshold=True
            )
            logger.debug("Gathered {} keys", len(list(dev_stats.keys())))
            final_dict[999] = dev_stats

            optimal_threshold = dev_stats.get("dev_optimal_threshold", .5)
            logger.info("Determined the optimal thresholds for the test split based on the dev-retrain: {}",
                        optimal_threshold)
        elif optimal_epoch >= 0:
            optimal_threshold = final_dict[optimal_epoch].get(
                "dev_used_threshold",
                final_dict[optimal_epoch].get("train_used_threshold", .5)
            )
            logger.info("Determined the optimal thresholds for the test split: {}", optimal_threshold)
        else:
            optimal_threshold = False

        test_stats = loop(
            f_epoch=999,
            f_split="test",
            with_param_adaption=False,
            threshold=optimal_threshold
        )

        logger.debug("Received {} test keys...", len(list(test_stats.keys())))

        df: pandas.DataFrame = collected_data["test"]
        try:
            df["prediction"] = test_stats["test_pred_text"]
            df_without_info_columns = df.drop(columns=set(df.columns).intersection({"text", "frames"}))

            if "language" in df_without_info_columns.columns:
                set_languages = set(df["language"])
                logger.info("Found {} languages in the test data: {}",
                            len(set_languages), ",".join(map(lambda l: "\"{}\"".format(l), set_languages)))
                for lang in set_languages:
                    lang_test_pred_file: Path = args.root_save_path.joinpath("test_pred-{}.txt".format(lang))
                    df_without_info_columns[df_without_info_columns.language == lang].drop(
                        columns=["language"], inplace=False
                    ).to_csv(path_or_buf=lang_test_pred_file, sep="\t", index=True, index_label="article_ID",
                             header=False,  encoding="utf-8")
                    logger.debug("Saved the {} predictions into \"{}\"", lang, lang_test_pred_file.name)
            else:
                df_without_info_columns.to_csv(path_or_buf=args.root_save_path.joinpath("test_pred.txt"), sep="\t",
                                               encoding="utf-8", index=True, index_label="article_ID", header=False)
                logger.debug("Saved all predictions into {}", args.root_save_path.joinpath("test_pred.txt"))

            with args.root_save_path.joinpath("test_pred.csv").open(mode="w", encoding="utf-8") as csv_stream:
                df.to_csv(path_or_buf=csv_stream, index=True, index_label="article_ID")
        except KeyError:
            logger.opt(exception=True).error("Can't write the test stats, malformed test keys ({})",
                                             "/".join(test_stats.keys()))
        except IOError:
            logger.opt(exception=True).error("Can't write {}",
                                             df.to_csv(path_or_buf=None, index=True, index_label="article_id"))

        logger.success("Finished inference!")

    with args.root_save_path.joinpath("stats.json").open(mode="w", encoding="utf-8", errors="ignore") as stats_stream:
        json.dump(obj=final_dict, fp=stats_stream, indent="\t", skipkeys=True, sort_keys=True, cls=MLJsonSerializer)
