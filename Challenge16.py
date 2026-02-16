import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
from ast import literal_eval

def get_input_data():
    with open('data/coins.txt') as f:
        data = f.read()
    return data


def calculate_min_coins(coin_data: list):
    coins = np.array(coin_data[0])
    coin_counts = np.array(coin_data[1])
    target = coin_data[2]

    c = np.ones(len(coins)).tolist()
    integrality = 1
    constraints = [LinearConstraint(coins, target, target)]
    bounds = Bounds(0, coin_counts)

    res = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    return -1 if not res.success else int(sum(res.x))


def test_solution():
    assert(calculate_min_coins([[1, 5, 10], [10, 2, 1], 23]) == 6)
    assert(calculate_min_coins([[1,5],[3,2],14]) == -1)

if __name__ == '__main__':
    test_solution()
    solution = [int(calculate_min_coins(i)) for i in literal_eval(get_input_data())]
    print(solution)


