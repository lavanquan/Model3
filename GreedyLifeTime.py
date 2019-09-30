import math
import random
from operator import itemgetter
import copy
import pandas as pd
from ast import literal_eval
from sklearn import linear_model
from pulp import *

node_pos = [(10,10),(30,30),(50,50),(70,70),(90,90),
           (10,30),(30,10),(30,50),(50,30),(50,70)]
charge_pos = [(10, 50), (90, 50)]
time_move = [1.019803902718557, 1.6, 2.0591260281974]
E = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
e = [0.2, 0.3, 0.3, 0.5, 0.6, 0.2, 0.6, 0.4, 0.6, 0.3]
numNode = len(node_pos)
numCharge = len(charge_pos)
E_mc = 5 # nang luong khoi tao cua MC
e_mc = 1 #cong suat sac moi giay
E_max = 10.0 #nang luong toi da
e_move = 0.1 #nang luong tieu thu moi giay cho viec di chuyen
E_move = [e_move * time_move_i for time_move_i in time_move] # nang luong tieu thu de di chuyen toi moi charge position
chargeRange = 10**10
alpha = 600
beta = 30
charge = []
delta = [[0 for u, _ in enumerate(charge_pos)] for j, _ in enumerate(node_pos)]

def getData(file_name="data.csv", index=0):
    global node_pos
    global numNode
    global E
    global e
    global charge_pos
    global numCharge
    global time_move
    global E_mc
    global e_mc
    global E_max
    global e_move
    global E_move
    global alpha
    global beta

    df = pd.read_csv(file_name)
    node_pos = list(literal_eval(df.node_pos[index]))
    numNode = len(node_pos)
    E = [df.energy[index] for _ in node_pos]
    e = map(float, df.e[index].split(","))
    charge_pos = list(literal_eval(df.charge_pos[index]))
    numCharge = len(charge_pos)
    velocity = df.velocity[index]
    E_mc = df.E_mc[index]
    E_max = df.E_max[index]
    e_mc = df.e_mc[index]
    e_move = df.e_move[index]
    alpha = df.alpha[index]
    beta = df.beta[index]

    charge_extend = []
    charge_extend.extend(charge_pos)
    charge_extend.append((0,0))
    time_move = [distance(charge_extend[i], charge_extend[i+1]) / velocity for i, _ in enumerate(charge_pos)]
    E_move = [e_move * item for item in time_move]

def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) * (node1[0] - node2[0])
                     + (node1[1] - node2[1]) * (node1[1] - node2[1]))

def charge(node, charge):
    d = distance(node, charge)
    if d > chargeRange:
        return 0
    else:
        return alpha / ((d + beta)**2)

def getWeightLinearRegression():
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(delta, e)
    w = regr.coef_
    if sum(w):
        x = [item / sum(w) for item in w]
    else:
        x = 0
    return w

def getWeightLinearPrograming1():
    w = 0 # bien return
    model = LpProblem("Charge", LpMinimize)
    x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
    t = LpVariable.matrix("t", list(range(numNode)), 0, None, LpContinuous)
    for j, _ in enumerate(node_pos):
        model += lpSum([x[u] * delta[j][u] for u, _ in enumerate(charge_pos)]) - e[j] <= t[j]
        model += lpSum([x[u] * delta[j][u] for u, _ in enumerate(charge_pos)]) - e[j] >= -t[j]
    model += lpSum(t)
    status = model.solve()
    if status == 1:
        valueX = [value(item) for item in x]
        if sum(valueX):
            w = [item / sum(valueX) for item in valueX]
        else:
            print "khong tim duoc lo trinh"
    else:
        print "khong giai duoc bai toan LP"
    return w

def getWeightLinearPrograming2(E_now):
    w = 0 # bien return
    model = LpProblem("Charge", LpMinimize)
    x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
    t = LpVariable.matrix("t", list(range(numNode)), 0, None, LpContinuous)
    for j, _ in enumerate(node_pos):
        model += lpSum([x[u] * delta[j][u] for u, _ in enumerate(charge_pos)]) - e[j] / E_now[j] <= t[j]
        model += lpSum([x[u] * delta[j][u] for u, _ in enumerate(charge_pos)]) - e[j] / E_now[j] >= -t[j]
    model += lpSum(t)
    status = model.solve()
    if status == 1:
        valueX = [value(item) for item in x]
        if sum(valueX):
            w = [item / sum(valueX) for item in valueX]
        else:
            print "khong tim duoc lo trinh"
    else:
        print "khong giai duoc bai toan LP"
    return w

def getWeightLinearPrograming3(E_now, gamma):
    w = 0 # bien return
    model = LpProblem("Charge", LpMinimize)
    x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
    t = LpVariable.matrix("t", list(range(numNode)), 0, None, LpContinuous)
    for j, _ in enumerate(node_pos):
        model += lpSum([x[u] * delta[j][u] for u, _ in enumerate(charge_pos)]) - gamma * (e[j] / sum(e)) + (1 - gamma) * (E_now[j] / sum(E_now)) <= t[j]
        model += lpSum([x[u] * delta[j][u] for u, _ in enumerate(charge_pos)]) - gamma * (e[j] / sum(e)) + (1 - gamma) * (E_now[j] / sum(E_now)) >= -t[j]
    model += lpSum(t)
    status = model.solve()
    if status == 1:
        valueX = [value(item) for item in x]
        if sum(valueX):
            w = [item / sum(valueX) for item in valueX]
        else:
            print "khong tim duoc lo trinh"
    else:
        print "khong giai duoc bai toan LP"
    return w

def getWeightLinearPrograming4(E_now):
    w = 0 # bien return
    model = LpProblem("Charge", LpMinimize)
    x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
    t = LpVariable.matrix("t", list(range(numNode)), 0, None, LpContinuous)
    for j, _ in enumerate(node_pos):
        model += lpSum([x[u] * delta[j][u] for u, _ in enumerate(charge_pos)]) - (e[j] / sum(e)) / (E_now[j] / sum(E_now)) <= t[j]
        model += lpSum([x[u] * delta[j][u] for u, _ in enumerate(charge_pos)]) - (e[j] / sum(e)) / (E_now[j] / sum(E_now)) >= -t[j]
    model += lpSum(t)
    status = model.solve()
    if status == 1:
        valueX = [value(item) for item in x]
        if sum(valueX):
            w = [item / sum(valueX) for item in valueX]
        else:
            print "khong tim duoc lo trinh"
    else:
        print "khong giai duoc bai toan LP"
    return w

def getCharge(E_mc_now, w):
    t = E_mc_now / sum([w[u] * sum([charge[j][u] for j, _ in enumerate(node_pos)]) for u, _ in enumerate(charge_pos)])
    x = [t * w[u] for u, _ in enumerate(charge_pos)]
    return x

def getRound():
    E_now = E
    E_mc_now = E_mc

    gen = []
    T = []
    remain = 0
    life_time = 0

    isStop = False
    index = 0
    t = 0
    while True:
        print "circle = ", t, "energy = ", min(E_now), max(E_now)
        t += 1
        temp_T = (E_max - E_mc_now) / e_mc
        temp_E = [E_now[j] - temp_T * e[j] for j, _ in enumerate(node_pos)]
        eNode = [temp_E[j] - time_move[0] * e[j] for j, _ in enumerate(node_pos)]
        if min(eNode) < 0:
            remain = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
            break
        else:
            T.append(temp_T)
            E_mc_now = E_mc_now + temp_T * e_mc
            E_now = temp_E
            w = getWeightLinearPrograming3(E_now, 0.5)
            tmp = getCharge(E_mc_now - sum(E_move), w)
            x = [0 for u, _ in enumerate(charge_pos)]
            for u, _ in enumerate(charge_pos):
                p = [min(charge[j][u] * tmp[u], E[j] - E_now[j] + time_move[u] * e[j]) for j, node in enumerate(node_pos)]
                temp_E_mc = E_mc_now - sum(p) - E_move[u]
                temp_E = [E_now[j] + p[j] - (time_move[u] + tmp[u]) * e[j] for j, _ in enumerate(node_pos)]
                if min(temp_E) < 0:
                    isStop = True
                    index = u
                    break
                else:
                    x[u] = tmp[u]
                    E_mc_now = temp_E_mc
                    E_now = temp_E
            gen.append(x)
            if not isStop:
                E_mc_now = E_mc_now - E_move[-1]
                E_now = [E_now[j] - time_move[-1] * e[j] for j, _ in enumerate(node_pos)]
            else:
                break
    remain = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
    nbRound = len(T)
    life_time = sum(T[:nbRound]) + sum([sum(gen[nbGen]) for nbGen in range(nbRound - 1)]) + (nbRound - 1) * sum(time_move)
    life_time += T[-1] + sum(gen[-1][:index]) + sum(time_move[:index]) + remain
    return life_time, T, gen, remain

def get_life_time(T, gen, remain):
    life_time = 0.0
    for index in range(len(T)):
        life_time += T[index]
        for id, current in enumerate(gen[index]):
            if id == 0:
                time = time_move[-1][current[0]]
            else:
                last = gen[index][id - 1]
                time = time_move[last[0]][current[0]]
            life_time += time + current[1]
        if index != len(T) - 1:
            u_last, _ = gen[index][-1]
            life_time += time_move[-1][u_last]
    life_time += remain
    return life_time

# main task
getData(file_name="data.csv", index=0)

charge = [[charge(node, pos) for u, pos in enumerate(charge_pos)] for j, node in enumerate(node_pos)]
# do chenh lech nang luong cua moi sensor j khi MC dung sac tai vi tri u
delta = [[charge[j][u] - e[j] for u, _ in enumerate(charge_pos)] for j, _ in enumerate(node_pos)]
life_time, T, gen, remain = getRound()
print life_time
print T[0], T[-1]
print remain
print gen[0]
print gen[-1]
















