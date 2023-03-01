import torch

from loguru import logger


def base_loss(labels: torch.Tensor, pred: torch.Tensor,
              scale: float = 1, verbose: bool = False,
              how_many_frames_prediction_in_front: bool = False) -> torch.Tensor:
    if verbose:
        logger.trace("Compute loss again")
    try:
        logger.trace("Calculated following output: {}", pred)

        should_frame_count = torch.sum(labels, dim=-1)
        is_frame_count = pred[:, 0] if how_many_frames_prediction_in_front else should_frame_count

        pred_frame_probabilities = pred[:, 1:] if how_many_frames_prediction_in_front else pred
        if verbose:
            logger.debug("Following frame probabilities are predicted: {}", pred_frame_probabilities.cpu().tolist())

        loss_frame_count = torch.mean((is_frame_count - should_frame_count) ** 2)
        loss_frames = -torch.mean(labels * torch.log(pred_frame_probabilities + 1e-7) +
                                  (1 - labels) * torch.log(1 + 1e-7 - pred_frame_probabilities))

        f_loss = loss_frames + loss_frame_count
        if verbose:
            logger.info("For this batch, the loss regarding how many frames are in the text is {} and the loss for the "
                        "frame probabilities is {}, resulting in a final loss of {}",
                        loss_frame_count.cpu().item(), loss_frames.cpu().item(), f_loss.cpu().item())
    except RuntimeError:
        if verbose:
            logger.opt(exception=True).critical("Fatal error in loss computation! (verbose mode)")
        else:
            logger.opt(exception=True).error("Fatal error in loss computation!")
        return torch.tensor(data=0, dtype=torch.float)

    return scale * f_loss


def loss(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=1., verbose=False, how_many_frames_prediction_in_front=False)


def loss_reduced(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=1/10, verbose=False, how_many_frames_prediction_in_front=False)


def loss_strong(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=2., verbose=False, how_many_frames_prediction_in_front=False)


def loss_count_frames(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=1., verbose=False, how_many_frames_prediction_in_front=True)


def loss_reduced_count_frames(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=1/10, verbose=False, how_many_frames_prediction_in_front=True)


def verbose_loss_strong_count_frames(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=2., verbose=True, how_many_frames_prediction_in_front=True)


def verbose_loss(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=1., verbose=True, how_many_frames_prediction_in_front=False)


def verbose_loss_reduced(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=1/10, verbose=True, how_many_frames_prediction_in_front=False)


def verbose_loss_strong(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=2., verbose=True, how_many_frames_prediction_in_front=False)


def verbose_loss_count_frames(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=1., verbose=True, how_many_frames_prediction_in_front=True)


def verbose_loss_reduced_count_frames(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=1/10, verbose=True, how_many_frames_prediction_in_front=True)


def verbose_loss_strong_count_frames(labels: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return base_loss(labels=labels, pred=pred, scale=2., verbose=True, how_many_frames_prediction_in_front=True)
