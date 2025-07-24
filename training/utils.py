import torch


@torch.inference_mode()
def test_model_correctness(model, loader, process_fn, device) -> str:
    """
    Tests model correctness (just output dimension for now).
    :param model: model to test
    :param loader: any loader capable of producing one batch
    :param process_fn: function to process batch
    :param device: GPU device
    """
    batch = next(iter(loader))
    baseline_data, followup_data, _ = process_fn(batch, device)
    model = model.to(device)
    outputs = model(baseline_data, followup_data).cpu().numpy()
    assert (outputs.shape[-1] == 4), f"Wrong last output dimension. Expected {4}; Got {outputs.shape[-1]}"
    return "MODEL SEEMS TO BE FINE!"


def calculate_params(model: torch.nn.Module) -> str:
    """
    Calculate number of parameters for the model.
    :param model: target model
    :return: number of parameters in millions.
    """
    return f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M"
