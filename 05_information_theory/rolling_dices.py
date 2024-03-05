'''Entropy excercise
Random variable X: n개의 주사위를 던졌을 경우 나온 값의 합
'''

from itertools import product # 중복 순열 구하기
from functools import reduce
from math import log2
import matplotlib.pyplot as plt

OUTCOMES = [1, 2, 3, 4, 5, 6]


def get_permutations(num_dices: int = 2) -> list:
    '''모든 가능한 결과의 순열'''
    result = [x for x in product(OUTCOMES, repeat=num_dices)]
    return result


def get_probability(target_value: int = None, target_prob: float = None) -> dict:
    '''각 주사위 값 나올 확률'''
    if not target_value:
        equal_prob = 1/len(OUTCOMES)
        prob_dic = {value: equal_prob for value in OUTCOMES}
        return prob_dic
    elif not target_prob:
        '''주사위 값은 지정하고, 그 확률을 지정하지 않은 경우'''
        print(f'지정한 주사위 값 "{target_value}"이 나올 확률(target_prob)을 지정하지 않았습니다.')
        print(f'모든 확률(target_prob)을 동일하게 지정하겠습니다.')
        equal_prob = 1/len(OUTCOMES)
        prob_dic = {value: equal_prob for value in OUTCOMES}
        return prob_dic
    else:
        probability = (1.0 - target_prob)/(len(OUTCOMES)-1)
        prob_dic = {value: probability for value in OUTCOMES if value != target_value}
        prob_dic[target_value] = target_prob
        return prob_dic

def get_outcome_prob(outcome_set: set, prob_dic: dict) -> float:
    '''해당 outcome이 나올 확률'''
    prob_list = []
    for x in outcome_set:
        prob_list.append(prob_dic[x])
    prob_of_outcome = reduce(lambda x, y: x*y, prob_list)
    return prob_of_outcome

def compute_entropy(num_dices, target_value, target_prob) -> float:
    '''결괏값에 따른 엔트로피 계산'''
    outcome_set = get_permutations(num_dices)
    prob_dic = get_probability(target_value, target_prob)
    
    # Random variable 구하기
    rand_var_list = [sum(outcome) for outcome in outcome_set]
    
    # Random variable 확률값 초기화
    rand_var_dic = {x: 0.0 for x in rand_var_list}
    
    # Random variable 확률값 계산
    for outcome in outcome_set:
        p = get_outcome_prob(outcome, prob_dic)
        random_var = sum(outcome)
        rand_var_dic[random_var] += p
        
    # Random variable 확률분포의 엔트로피 계산
    entropy = 0.0
    for p in rand_var_dic.values():
        temp = -1.0 * p * log2(p)
        entropy += temp
    
    # Random variable 확률분포 시각화
    x = rand_var_dic.keys()
    y = rand_var_dic.values()
    plt.bar(x, y)
    plt.xlabel('Random variable (sum of values)')
    plt.ylabel('Probability')
    target_prob = 'Equal' if not target_prob else target_prob
    plt.title(f'#Dices: {num_dices}, Tgt_val: {target_value}, Tgt_prob: {target_prob}, Entropy: {entropy: .2f}')
    plt.show()


if __name__=='__main__':
    num_dices = input('실험에 사용할 주사위 개수: ')
    if not num_dices:
        num_dices = 2
    else:
        num_dices = int(num_dices)
    
    target_value = input('확률을 지정할 주사위 값: ')
    if not target_value:
        target_value = None
    else:
        target_value = int(target_value)
    
    target_prob = input('주사위 값이 나올 확률: ')
    if not target_prob:
        target_prob = None
    else:
        target_prob = float(target_prob)
    
    # 엔트로피 계산
    compute_entropy(num_dices, target_value, target_prob)