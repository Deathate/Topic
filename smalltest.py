
import numpy as np
import collections
import matplotlib.pyplot as plt
import math

# 固定隨機種子
np.random.seed(42)
# Q table
Q = collections.defaultdict(dict, {})


def output():
    # 印出Q table
    print(f"  Q  state")
    for i in Q.keys():
        print(f"[{i:>2}]", end=" ")
        for j in Q[i].keys():
            print(f"({j}: {round(Q[i][j],2):<4})", end=" ")
        print()


# 第幾期
ROUND = 0
# epsilon greedy 的 epsilon
epsilon = 0

# customer 基礎價格
customers = [10, 20, 50, 60]
# customer 更新率
actions = [.9, 1, 2.1]
# 客戶數量
N = len(customers)


def epsilon_greedy(state, deterministic=False):
    # epsilon greedy function
    global epsilon

    if ep_mode == 0:
        epsilon = max(epsilon_min, epsilon*decay)
    elif ep_mode == 1:
        epsilon = max(epsilon-decay, epsilon_min)
    elif ep_mode == 2:
        epsilon = 1/ROUND
    elif ep_mode == 3:
        epsilon = 1

    def random_select():
        return np.random.choice(actions)
    rv = np.random.uniform(0, 1)
    if deterministic:
        rv = 1
    if rv < epsilon:
        return random_select()
    else:
        # greedy method
        lst = [Q[state].get(x, 0) for x in actions]
        if lst.count(lst[0]) == len(lst):
            return random_select()
        else:
            return max(Q[state], key=Q[state].get)


c = []


def Train_v1():
    global ROUND
    oldfn = 0
    while True:
        f1n = 0
        cum_value = 0
        ROUND = ROUND + 1

        def train():
            nonlocal f1n, cum_value
            f1n = 0
            cum_value = 0
            for j in range(N):
                an = epsilon_greedy(f1n)
                rho = .5
                if RANDOMNESS:
                    rho = np.random.binomial(SAMPLESIZE, .5)/SAMPLESIZE
                    rho = round(rho, 1)
                cum_value += customers[j] * an * rho
                value = cum_value

                maxq = Q[value][max(Q[value],
                                    key=Q[value].get)] if Q[value] else 0
                result = 0.6 * Q[f1n].get(an, 0) + 0.4 * (value-f1n + 1 * maxq)
                # if ROUND % (1000 if not PROGRESSIVE else 1) == 0:
                #     output()
                #     print()
                #     print(
                #         f"Q[{f1n},{an}] = 0.6 * {Q[f1n].get(an, 0)} + 0.4 * (({value} - {f1n}) + 0.6 * max({maxq})) = {result}")
                #     print(f"next state: Q[{value}]")
                #     print(f"epsilon: {epsilon}")
                #     print("-----------------------------")
                #     print(ROUND)
                #     if PROGRESSIVE:
                #         input()

                Q[f1n][an] = result
                f1n = value
            return f1n
        train()
        if abs(f1n - oldfn) <= CONVBOUND:
            isConv = isConv + 1
        else:
            # print(f1n-oldfn)
            isConv = 0
        if isConv == 20:
            output()
            print("Converge")
            print(ROUND, train())
            break
        oldfn = f1n
        c.append(f1n)

        if ROUND == EPOCH:
            print("reach limit")
            print(ROUND, train())
            break


def Train_v2():
    global ROUND
    oldfn = 0
    while True:
        f1n = 0
        cum_value = 0
        ROUND = ROUND + 1

        def train():
            nonlocal f1n, cum_value
            f1n = sum([customers[r] for r in range(0, N)])
            cum_value = 0
            for j in range(N):
                an = epsilon_greedy(f1n)
                rho = .5
                if RANDOMNESS:
                    rho = np.random.binomial(SAMPLESIZE, .5)/SAMPLESIZE
                    rho = round(rho, 1)
                cum_value += customers[j] * an * rho
                value = cum_value + sum([customers[r]
                                         for r in range(j+1, N)])

                maxq = Q[value][max(Q[value],
                                    key=Q[value].get)] if Q[value] else 0
                result = 0.6 * Q[f1n].get(an, 0) + 0.4 * (value-f1n + 1 * maxq)
                # if ROUND % (1000 if not PROGRESSIVE else 1) == 0:
                #     output()
                #     print()
                #     print(
                #         f"Q[{f1n},{an}] = 0.6 * {Q[f1n].get(an, 0)} + 0.4 * (({value} - {f1n}) + 0.6 * max({maxq})) = {result}")
                #     print(f"next state: Q[{value}]")
                #     print(f"epsilon: {epsilon}")
                #     print("-----------------------------")
                #     print(ROUND)
                #     if PROGRESSIVE:
                #         input()

                Q[f1n][an] = result
                f1n = value
            return f1n
        train()
        if abs(f1n - oldfn) <= CONVBOUND:
            isConv = isConv + 1
        else:
            # print(f1n-oldfn)
            isConv = 0
        if isConv == 20:
            output()
            print("Converge")
            print(ROUND, train())
            break
        oldfn = f1n
        c.append(f1n)

        if ROUND == EPOCH:
            print("reach limit")
            print(ROUND, train())
            break


# 0 為 decayed-episilon-greedy, power linear 1/n, 1 為完全隨機
ep_mode = 1
epsilon_max = 1
epsilon_min = 0.001
decayed_step = 500
if ep_mode == 0:
    decay = math.exp(math.log(epsilon_min/epsilon_max) /
                     (decayed_step * N))
    epsilon = epsilon_max / decay
elif ep_mode == 1:
    decay = (epsilon_max - epsilon_min) / (decayed_step * N)
    epsilon = epsilon_max

EPOCH = 1000
SAMPLESIZE = 2000
PROGRESSIVE = False
RANDOMNESS = True
isConv = 0
CONVBOUND = 0
Train_v2()
plt.plot(c)
plt.show()
