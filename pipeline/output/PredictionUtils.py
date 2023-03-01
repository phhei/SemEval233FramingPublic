from typing import Union, List, Optional

from sklearn.metrics import roc_curve

from scorers.scorer_subtask_2 import read_frame_list_from_file

from loguru import logger

import numpy
import torch


def numpy_sigmoid(x: numpy.ndarray) -> numpy.ndarray:
    return 1/(1 + numpy.exp(-x))


def map_logits_to_binary_prediction(
        logits: Union[numpy.ndarray, torch.Tensor],
        threshold: Union[float, List[float], numpy.ndarray,  torch.Tensor] = .5,
        min_selection: int = 1, max_selection: int = 5,
        sigmoid_logits: bool = True
) -> Union[numpy.ndarray, torch.LongTensor]:
    if isinstance(threshold, List):
        threshold = torch.tensor(data=threshold, dtype=torch.float, device=logits.device) \
            if isinstance(logits, torch.Tensor) else numpy.array(threshold, dtype=numpy.float32)

    if isinstance(logits, torch.Tensor):
        logits_sigmoid = torch.sigmoid(logits) if sigmoid_logits else logits
        binary_logits = torch.where(logits_sigmoid > threshold, 1, 0).type(torch.LongTensor)
        selected_sum = torch.sum(binary_logits).item()
    else:
        logits_sigmoid = numpy_sigmoid(logits) if sigmoid_logits else logits
        binary_logits = numpy.where(logits_sigmoid > threshold, 1, 0).astype(numpy.int_)
        selected_sum = numpy.sum(binary_logits)

    if min_selection <= selected_sum <= max_selection:
        return binary_logits

    if isinstance(logits, torch.Tensor):
        indices = torch.argsort(input=logits_sigmoid-threshold, descending=True)
        if selected_sum < min_selection:
            return torch.sum(torch.nn.functional.one_hot(indices[:min_selection], num_classes=logits_sigmoid.shape[-1]), dim=0)

        return torch.sum(torch.nn.functional.one_hot(indices[:max_selection], num_classes=logits_sigmoid.shape[-1]), dim=0)
    else:
        indices = numpy.flip(numpy.argsort(logits_sigmoid - threshold))
        if selected_sum < min_selection:
            return numpy.array([int(i in indices[:min_selection]) for i in range(logits_sigmoid.shape[0])], dtype=numpy.int_)

        return numpy.array([int(i in indices[:max_selection]) for i in range(logits_sigmoid.shape[0])], dtype=numpy.int_)


def map_count_prediction_to_binary_prediction(
        logits: Union[numpy.ndarray, torch.Tensor],
        class_offset: Optional[Union[numpy.ndarray, torch.Tensor]] = None,
        sigmoid_logits: bool = True
) -> Union[numpy.ndarray, torch.LongTensor]:
    if class_offset is None:
        class_offset = 0

    if isinstance(logits, torch.Tensor):
        estimated_count, _logits = logits[0].item(), torch.sigmoid(logits[1:]) if sigmoid_logits else logits[1:]
    else:
        estimated_count, _logits = logits[0], numpy_sigmoid(logits[1:]) if sigmoid_logits else logits[1:]
    estimated_count = max(1, int(round(estimated_count)))

    if isinstance(logits, torch.Tensor):
        indices = torch.argsort(input=_logits - class_offset, descending=True)

        return torch.sum(torch.nn.functional.one_hot(indices[:estimated_count], num_classes=_logits.shape[-1]), dim=0)
    else:
        indices = numpy.flip(numpy.argsort(_logits - class_offset))
        return numpy.array([int(i in indices[:estimated_count]) for i in range(_logits.shape[0])], dtype=numpy.int_)


def map_binary_tensors_to_text_prediction(
        logits: Union[numpy.ndarray, torch.Tensor,  List[numpy.ndarray], List[torch.Tensor]],
        file_to_class_labels: str = "scorers/frames_subtask2.txt"
) -> List[str]:
    if isinstance(logits, List) and len(logits) == 0:
        return []

    class_labels = {i: s for i, s in enumerate(read_frame_list_from_file(file_to_class_labels))}

    if isinstance(logits, torch.Tensor):
        logits = logits.numpy(force=True)

    if isinstance(logits, numpy.ndarray):
        if len(logits.shape) == 1:
            logits = numpy.expand_dims(logits, axis=0)

    ret = []
    for prediction in logits:
        if isinstance(prediction, torch.Tensor):
            prediction_numpy: numpy.ndarray = prediction.numpy(force=True)
        else:
            prediction_numpy: numpy.ndarray = prediction

        ret.append(",".join([class_labels[index] for index in numpy.ravel(numpy.argwhere(prediction_numpy == 1))]))

    return ret


def compute_optimal_threshold(predicted: numpy.ndarray, reference: numpy.ndarray) -> float:
    logger.trace("OK, having {} entries...", len(predicted))
    try:
        fpr, tpr, thresholds = roc_curve(y_score=predicted, y_true=reference, pos_label=1)
        true_false_rate = tpr - fpr
        ix = numpy.argmax(true_false_rate)
        logger.trace("Found following true-positive-rates: {}, false-positive-rates: {} "
                     "under following thresholds: {}", tpr, fpr, thresholds)
        threshold = thresholds[ix]
        logger.info("Found the optimal threshold: {}", round(threshold, 5))
    except ValueError:
        logger.opt(exception=True).critical("Something went wrong in calculating the optimal threshold "
                                            "(fall back to .5). "
                                            "The values for predicted_arg_kp_matches: {}. "
                                            "The values for ground_truth_arg_kp_matches: {}",
                                            predicted.tolist(),
                                            reference.tolist())
        threshold = .5
    return threshold
