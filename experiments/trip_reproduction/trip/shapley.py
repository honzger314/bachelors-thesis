from itertools import combinations
from math import factorial


def powerset(players):
    """
    Generate all subsets of the given players.

    Example:
        [0,1]

    yields

        ()
        (0,)
        (1,)
        (0,1)
    """

    players = list(players)

    for r in range(len(players) + 1):
        for subset in combinations(players, r):
            yield subset


def shapley_weight(n, subset_size):
    """
    Shapley coefficient

        |S|! (n-|S|-1)! / n!

    """

    return (
        factorial(subset_size)
        * factorial(n - subset_size - 1)
        / factorial(n)
    )


def compute_shapley(players, utility_function):
    """
    Computes the exact Shapley value.

    Parameters
    ----------
    players : iterable

        Example:
            [2,4,7]

    utility_function : callable

        Receives a tuple/list representing a coalition.

        Example:

            utility((2,7))

        Returns

            float

    Returns
    -------
    dict

        {
            player : shapley_value
        }

    """

    players = list(players)

    n = len(players)

    shapley = {
        p: 0.0
        for p in players
    }

    for player in players:

        others = [
            p
            for p in players
            if p != player
        ]

        for subset in powerset(others):

            subset = tuple(subset)

            with_player = tuple(
                sorted(subset + (player,))
            )

            marginal = (
                utility_function(with_player)
                - utility_function(subset)
            )

            weight = shapley_weight(
                n,
                len(subset)
            )

            shapley[player] += (
                weight * marginal
            )

    return shapley


def normalize(values):
    """
    Optional helper.

    Converts Shapley values into percentages
    summing to one.

    Useful for visualization.
    """

    total = sum(values.values())

    if abs(total) < 1e-12:
        return values.copy()

    return {
        k: v / total
        for k, v in values.items()
    }


if __name__ == "__main__":

    #
    # Small sanity check
    #

    players = [0, 1, 2]

    def utility(coalition):
        """
        Simple additive game.

        value(S)=sum(player_id+1)
        """

        score = 0

        for p in coalition:
            score += p + 1

        return score

    phi = compute_shapley(
        players,
        utility
    )

    print("Shapley values:")

    for k, v in phi.items():
        print(f"{k}: {v:.3f}")