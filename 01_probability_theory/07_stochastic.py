def rolling_die_1():
    '''return 1에서 6사이 하나의 정수값'''


# def rolling_die_2():
#     '''return 1에서 6사이 정수 값 중
#        랜던하게 선택된 숫자
#     '''

import random
import math

def rolling_die_2():
    '''return 1에서 6사이 정수 값 중
       랜던하게 선택된 숫자
    '''
    return random.choice([1, 2, 3, 4, 5, 6])


def run_rolling(test_num=10):
    '''주사위 던지기 실행'''
    result = []
    for i in range(test_num):
        result.append(rolling_die_2())
    print(result)


def die_prob_estimator(target: str, trials:int):
    '''원하는 주사위 값 나올 확률 계산'''
    hit_count = 0
    for _ in range(trials):
        result = ''
        for _ in range(len(target)):
            result += str(rolling_die_2())
        if target == result:
            hit_count += 1
    prob_estimation = round(hit_count/trials, 6)
    print(f'Estimated Probability of {target}: {prob_estimation}')


def same_birthday_prob(
    num_people: int,
    num_same: int
    ) -> bool:
    '''num_same 보다 많은 생일이 있는지 판별'''
    possible_date = range(365) # 2월 29일 포함
    birthdays = [0] * 365
    for _ in range(num_people):
        birth_date = random.choice(possible_date)
        birthdays[birth_date] += 1
    return max(birthdays) >= num_same # return boolean


def birthday_estimator(
    num_people: int,
    num_same: int,
    trials: int) -> float:
    '''num_people 중에 생일이 같은 사람이 있을 확률 계산'''
    hit_count = 0
    for _ in range(trials):
        if same_birthday_prob(num_people, num_same):
            hit_count += 1
    return hit_count/trials


def run_birthday_simulation(
    num_same: int,
    peoples: list) -> None:
    '''같은 생일이 있는지 시뮬레이션 수행'''
    for people in peoples:
        print('---'*10)
        estimated_probability = round(birthday_estimator(people, num_same, 1000), 3)
        numerator = math.factorial(365)
        denominator = (365**people)*math.factorial(365-people)
        theoretical_prob = round(1 - numerator/denominator, 3)
        print(f'{people}명 중 같은 생일이 있을 확률: \n\
시뮬레이션: {estimated_probability}\t이론값: {theoretical_prob}')



if __name__=='__main__':
    # die_prob_estimator('11111', 1000000)
    run_birthday_simulation(2, [10, 20, 40, 100])
    # run_birthday_simulation(3, [10, 20, 40, 100])
    # run_birthday_simulation(4, [10, 20, 40, 100])