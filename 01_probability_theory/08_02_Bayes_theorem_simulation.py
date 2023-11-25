'''Simulation of Bayesian Inference'''

import random
from scipy.special import comb
from typing import Union


def experiment(n: int = 8, x: int = 3) -> Union[None, str]:
    '''Bayesian experiment'''
    # Bob이 이길 확률 (랜덤값)
    prob_bob_win = random.random()

    # Alice 5점, Bob 3점 상태에 이를 확률(likelihood)
    prob_current_status =  comb(n, x) * prob_bob_win**x * (1-prob_bob_win)**(n-x)

    # 실제로 Alice 5점, Bob 3점이 발생했는지 확인
    test_result = None # 결괏값
    prob_test = random.random() # 현재 상황이 발생하였는지를 체크하기 위해 랜덤값 생성
    # 랜덤 확률이 현재 상황보다 작다면 -> 실제로 발생한 경우로 처리
    if prob_test < prob_current_status:
        # Bob 3번 연속으로 이길 확률
        prob_bob_three_in_a_row = prob_bob_win ** x
        # Bob이 실제로 이겼는지 확인
        if random.random() < prob_bob_three_in_a_row:
            test_result = 'Bob'
        else:
            test_result = 'Alice'

    return test_result


def run_simulation(trials: int = 100000):
    '''Conduct simulation for Bayesianist'''
    result = [experiment() for _ in range(trials)]
    bob_wins = result.count('Bob')
    alice_wins = result.count('Alice')
    print(f'# Experiment: {trials}')
    print(f'Bayesian prob   : {bob_wins/(bob_wins + alice_wins)*100:.2f}%')
    print(f'Frequentist prob: 5.3%')


if __name__=='__main__':
    run_simulation()
