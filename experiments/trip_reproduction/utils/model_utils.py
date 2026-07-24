import copy
import torch


def copy_model(model):
    """
    Creates a deep copy of a PyTorch model.

    In DFL every client owns its own model, so we need
    independent copies rather than references to the same model.
    """
    return copy.deepcopy(model)



def average_models(models, weights=None):
    """
    Weighted average of multiple PyTorch models.

    This is the basic aggregation operation used by DFL.

    Args:
        models:
            List of PyTorch models.

        weights:
            List of aggregation weights.
            If None, uniform averaging is used.

    Returns:
        A new model containing the averaged parameters.
    """

    if len(models) == 0:
        raise ValueError("Cannot average an empty model list")

    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    if len(weights) != len(models):
        raise ValueError(
            "Number of weights must match number of models"
        )

    # Create independent output model
    averaged_model = copy.deepcopy(models[0])

    # Disable gradients while modifying parameters
    with torch.no_grad():

        # Reset parameters
        for param in averaged_model.parameters():
            param.zero_()

        # Weighted sum
        for model, weight in zip(models, weights):

            for avg_param, param in zip(
                averaged_model.parameters(),
                model.parameters()
            ):
                avg_param += weight * param

    return averaged_model



def evaluate_model(model, dataloader, device="cpu"):
    """
    Evaluate model accuracy.

    Args:
        model:
            PyTorch model.

        dataloader:
            Test/validation DataLoader.

        device:
            cpu or cuda.

    Returns:
        Accuracy between 0 and 1.
    """

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    if total == 0:
        return 0.0

    return correct / total



def model_difference(model_a, model_b):
    """
    Computes the parameter difference:

        model_a - model_b

    This will later be useful for TRIP-Shapley because
    we need to reason about model updates.
    """

    differences = []

    for param_a, param_b in zip(
        model_a.parameters(),
        model_b.parameters()
    ):
        differences.append(
            param_a.data - param_b.data
        )

    return differences