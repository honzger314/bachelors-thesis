import copy
import torch

from trip.shapley import compute_shapley
from utils.model_utils import average_models, evaluate_model



def build_mixed_model(
    coalition,
    messages,
    device="cpu"
):
    """
    Implements Equation (5):

        θ_i^(t+1)(S)

    Clients inside coalition:
        use post-training model θ(t+1/2)

    Clients outside coalition:
        use pre-training model θ(t)

    """

    models = []


    for msg in messages:

        client_id = msg["client_id"]


        if client_id in coalition:

            models.append(
                msg["post_model"]
            )

        else:

            models.append(
                msg["pre_model"]
            )


    weights = [
        1.0 / len(models)
    ] * len(models)


    mixed_model = average_models(
        models,
        weights
    )


    return mixed_model.to(device)



def compute_utility(
    coalition,
    messages,
    test_loader,
    device="cpu"
):

    print(
        f"      Evaluating coalition {coalition}"
    )


    model = build_mixed_model(
        coalition,
        messages,
        device
    )


    accuracy = evaluate_model(
        model,
        test_loader,
        device
    )


    print(
        f"      Coalition {coalition} accuracy={accuracy:.4f}"
    )


    return accuracy


def compute_lcv(
    messages,
    test_loader,
    device="cpu"
):
    """
    Computes the Local Contribution Vector ψ(i,t).

    Input:

        messages:

        [
            {
              client_id,
              pre_model,
              post_model
            }
        ]

    Output:

        {
            client_id: contribution
        }


    This implements Equation (4).
    """


    players = [
        msg["client_id"]
        for msg in messages
    ]


    def utility(coalition):

        return compute_utility(
            coalition,
            messages,
            test_loader,
            device
        )


    shapley_values = compute_shapley(
        players,
        utility
    )


    return shapley_values