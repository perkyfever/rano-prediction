def default_process(batch, device):
    """
    Extracts needed input from the batch.
    :param batch: monai batch
    :param device: device to move input to
    :return: (baseline_data, followup_data data, labels)
    """
    baseline_data = {k: v.to(device) for k, v in batch["baseline"].items()}
    followup_data = {k: v.to(device) for k, v in batch["followup"].items()}
    labels = batch["label"].to(device)
    return baseline_data, followup_data, labels
