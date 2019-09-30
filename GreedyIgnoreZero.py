import math
import random
from operator import itemgetter
import copy
import pandas as pd
from ast import literal_eval
from sklearn import linear_model
from pulp import *
import csv

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
velocity = 0.0
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
    global velocity

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
    time_move = [[distance(pos1, pos2) / velocity for pos2 in charge_extend] for pos1 in charge_extend]

    tmp = [time_move[i][i+1] * e_move for i in range(len(time_move) - 1)]
    E_move = [time_move[-1][0] * e_move]
    E_move.extend(tmp)

def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) * (node1[0] - node2[0])
                     + (node1[1] - node2[1]) * (node1[1] - node2[1]))

def charging(node, charge):
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
            w = [1.0 / len(charge_pos) for _ in charge_pos]
    else:
        print "khong giai duoc bai toan LP"
    print w
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
            w = [1.0 / len(charge_pos) for _ in charge_pos]
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
            w = [1.0 / len(charge_pos) for _ in charge_pos]
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
            w = [1.0 / len(charge_pos) for _ in charge_pos]
    else:
        print "khong giai duoc bai toan LP"
   # print w
    return w

def getCharge(E_mc_now, w):
    t = (E_mc_now - sum(E_move)) / sum([w[u] * sum([charge[j][u] for j, _ in enumerate(node_pos)]) for u, _ in enumerate(charge_pos)])
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
        #print "circle = ", t, "energy = ", min(E_now), max(E_now)
        t += 1
        # temp_T, temp_E: cac gia tri nang luong cua sensor va MC trong truong hop T thoa man
        temp_T = (E_max - E_mc_now) / e_mc
        temp_E = [E_now[j] - temp_T * e[j] for j, _ in enumerate(node_pos)]
        #w = getWeightLinearPrograming1()
        #w = getWeightLinearPrograming2(E_now)
        w = getWeightLinearPrograming3(E_now, 0.5)
        #w = getWeightLinearPrograming4(E_now)
        #print w
        tmp = getCharge(E_mc_now + temp_T * e_mc - sum(E_move), w)
        x_not_zero = [(index, item) for index, item in enumerate(tmp) if item > 0]
        u_first, _ = x_not_zero[0]
        eNode = min([temp_E[j] - time_move[-1][u_first] * e[j] for j, _ in enumerate(node_pos)])
        print x_not_zero
        if eNode < 0:
            break
        else:
            T.append(temp_T)
            E_mc_now = E_mc_now + temp_T * e_mc
            E_now = temp_E
            x = []
            for index, current in enumerate(x_not_zero):
                # time: thoi gian di chuyen tu vi tri sac truoc do den vi tri hien tai
                # current[0]: id cua vi tri sac
                # current[1]: thoi gian dung sac
                u, xu = current
                if index == 0:
                    time = time_move[-1][u]
                else:
                    last = x_not_zero[index-1]
                    time = time_move[last[0]][u]
                #print min(E_now)
                eNode = min([E_now[j] - time * e[j] for j, _ in enumerate(node_pos)])
                if eNode < 0:
                    isStop = True
                    break
                #print eNode
                p = [min(charge[j][u] * xu, E[j] - E_now[j] + (time + xu) * e[j]) for j, node in enumerate(node_pos)]
                temp_E_mc = E_mc_now - sum(p) - time * e_move
                temp_E = [E_now[j] + p[j] - (time + xu) * e[j] for j, _ in enumerate(node_pos)]
                #print max(p), len(p)
                #print min(temp_E)
                if min(temp_E) < 0:
                    isStop = True
                    break
                else:
                    x.append(current)
                    E_mc_now = temp_E_mc
                    E_now = temp_E
            gen.append(x)
            if not isStop:
                u_last, _ = x[-1]
                E_mc_now = E_mc_now - time_move[-1][u_last] * e_move
                E_now = [E_now[j] - time_move[-1][u_last] * e[j] for j, _ in enumerate(node_pos)]
            else:
                break
    remain = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
    life_time = get_life_time(T, gen, remain)
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
index = 0

f = open("Greedy_LP3_lifetime.csv", mode="w")
header = ["Bo Du Lieu", "Co Sac", "Khong Sac"]
writer = csv.DictWriter(f, fieldnames=header)
writer.writeheader()
while index < 25:
    if index == 18:
        index = index + 1
        continue
    print "Data Set ", index
    getData(file_name="data.csv", index=index)
    charge = [[charging(node, pos) for u, pos in enumerate(charge_pos)] for j, node in enumerate(node_pos)]
    # do chenh lech nang luong cua moi sensor j khi MC dung sac tai vi tri u
    delta = [[charge[j][u] - e[j] for u, _ in enumerate(charge_pos)] for j, _ in enumerate(node_pos)]
    life_time, T, gen, remain = getRound()

    row = {}
    row["Bo Du Lieu"] = "No." + str(index)
    row["Co Sac"] = life_time
    row["Khong Sac"] = min([E[j] / e[j] for j, _ in enumerate(node_pos)])
    writer.writerow(row)
    print "Done Data Set ", index
    index = index + 1

f.close()
print "Done All"

"""print life_time
for item in T:
    print item,
print
for item in gen:
    print item
print remain"""
















