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

    players = list(players)

    n = len(players)

    shapley = {
        p: 0.0
        for p in players
    }


    cache = {}


    def cached_utility(coalition):

        coalition = tuple(sorted(coalition))

        if coalition not in cache:
            cache[coalition] = utility_function(coalition)

        return cache[coalition]


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
                cached_utility(with_player)
                -
                cached_utility(subset)
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