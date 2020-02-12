import math
import random
from operator import itemgetter
import copy
import pandas as pd
from ast import literal_eval
from multiprocessing import cpu_count, Process, Pipe
from pulp import *
import csv
import time
from scipy.stats import sem, t
from scipy import mean

node_pos = [(10, 10), (30, 30), (50, 50), (70, 70), (90, 90),
            (10, 30), (30, 10), (30, 50), (50, 30), (50, 70)]
charge_pos = [(10, 50), (90, 50)]
time_move = [1.019803902718557, 1.6, 2.0591260281974]
E = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
e = [0.2, 0.3, 0.3, 0.5, 0.6, 0.2, 0.6, 0.4, 0.6, 0.3]
numNode = len(node_pos)
numCharge = len(charge_pos)
E_mc = 5  # nang luong khoi tao cua MC
e_mc = 1  # cong suat sac moi giay
E_max = 10.0  # nang luong toi da
e_move = 0.1  # nang luong tieu thu moi giay cho viec di chuyen
E_move = [e_move * time_move_i for time_move_i in time_move]  # nang luong tieu thu de di chuyen toi moi charge position
chargeRange = 10 ** 10
velocity = 0.0
alpha = 600
beta = 30
charge = []
delta = [[0 for u, _ in enumerate(charge_pos)] for j, _ in enumerate(node_pos)]
U = 0.0
depot = (0.0, 0.0)
E_sensor_max = 10.0


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
    global U

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
    charge_extend.append((0, 0))
    time_move = [[distance(pos1, pos2) / velocity for pos2 in charge_extend] for pos1 in charge_extend]

    tmp = [time_move[i][i + 1] * e_move for i in range(len(time_move) - 1)]
    E_move = [time_move[-1][0] * e_move]
    E_move.extend(tmp)
    U = alpha / beta ** 2


def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) * (node1[0] - node2[0])
                     + (node1[1] - node2[1]) * (node1[1] - node2[1]))


def charging(node, charge):
    d = distance(node, charge)
    if d > chargeRange:
        return 0
    else:
        return alpha / ((d + beta) ** 2)


def get_T():
    Time_a_cycle = 0.0
    T_sensor = [E_sensor_max / e[j] + E_sensor_max / (U - e[j]) for j, _ in enumerate(e) if e[j] > 0]
    for j, _ in enumerate(T_sensor):
        if Time_a_cycle <= T_sensor[j]:
            Time_a_cycle = T_sensor[j]
    return Time_a_cycle


def fitness(indi, T):
    travel = indi["travel"]
    d = 0.0
    for i in range(len(travel) - 1):
        d = d + distance(node_pos[travel[i]], node_pos[travel[i + 1]])
    d = d + distance(depot, node_pos[travel[0]]) + distance(depot, node_pos[travel[-1]])
    tau_tsp = d / velocity
    #  E_tsp = tau_tsp * e_move
    tau_vac = (U - sum(e[i] for i in travel)) * T / U - tau_tsp
    return tau_vac / T


def individual(T):
    indi = {"travel": [], "velocity": [], "pbest": [], "fitness": 0.0}
    a = [i for i in range(numNode)]
    random.shuffle(a)
    # n = random.randint(1, numNode-1)
    n = numNode
    indi["travel"] = a[:n]
    indi["velocity"] = [0 for _ in range(n)]
    indi["pbest"] = a[:n]
    indi["fitness"] = fitness(indi, T)
    return indi


def selectionBest(popu):
    new_list = sorted(popu, key=itemgetter("fitness"), reverse=True)
    return new_list[:population_size]


def crossover(father, mother, T):
    mother_child = {"travel": [], "velocity": [], "pbest": [], "fitness": 0.0}
    father_child = {"travel": [], "velocity": [], "pbest": [], "fitness": 0.0}
    n = len(father["travel"])
    m = len(mother["travel"])
    p = min(m, n)
    if p == 1 or p == 2:
        return -1
    cutA = random.randint(1, p)
    cutB = random.randint(1, p)
    while cutB == cutA:
        cutB = random.randint(1, p)
    start = min(cutA, cutB)
    end = max(cutA, cutB)
    # print start, end
    father_temp = father["travel"][start:end]
    mother_temp = mother["travel"][start:end]
    # print temp
    index = 0
    while index < start:
        for i, item in enumerate(mother["travel"]):
            if item not in father_temp and item not in mother_child["travel"]:
                mother_child["travel"].append(item)
                mother_child["velocity"].append(mother["velocity"][i])
                index = index + 1
                break
    while index < end:
        mother_child["travel"].append(father["travel"][index])
        mother_child["velocity"].append((father["velocity"][index]))
        index = index + 1
    for i, item in enumerate(mother["travel"]):
        if item not in mother_child["travel"]:
            mother_child["travel"].append(item)
            mother_child["velocity"].append(mother["velocity"][i])

    index = 0
    while index < start:
        for i, item in enumerate(father["travel"]):
            if item not in mother_temp and item not in father_child["travel"]:
                father_child["travel"].append(item)
                father_child["velocity"].append(father["velocity"][i])
                index = index + 1
                break
    while index < end:
        father_child["travel"].append(mother["travel"][index])
        father_child["velocity"].append((mother["velocity"][index]))
        index = index + 1
    for i, item in enumerate(father["travel"]):
        if item not in father_child["travel"]:
            father_child["travel"].append(item)
            father_child["velocity"].append(father["velocity"][i])
    # print "temp1 =", gen1, "temp2 =", gen2, "Off =", off
    father_child["fitness"] = fitness(father_child, T)
    mother_child["fitness"] = fitness(mother_child, T)
    father_child["pbest"] = father_child["travel"]
    mother_child["pbest"] = mother_child["travel"]
    return father_child, mother_child


def add_operate(v2, v1):
    v = [0 for _ in v2]
    if len(v1) < len(v2):
        v1.extend(v2[len(v1):len(v2)])
    for i, _ in enumerate(v2):
        if v2[i] == 0:
            v[i] = v1[i]
        else:
            v[i] = v2[i]
    return v


def mul_operate(v1, c):
    v = [0 for _ in v1]
    r = [random.random() for _ in v1]
    for i, _ in enumerate(v):
        if r[i] >= c:
            v[i] = 0
        else:
            v[i] = v1[i]
    return v


def sub_operate(x2, x1):
    if len(x1) < len(x2):
        x1.extend(x2[len(x1):len(x2)])
    v = [0 for _ in x2]
    print len(x2), len(x1)
    for i, _ in enumerate(x2):
        if x2[i] == x1[i]:
            v[i] = 0
        else:
            v[i] = x2[i]
    return v


def mutation(indi, c1, c2, T, gbest):
    child = {"travel": [], "velocity": [], "pbest": [], "fitness": 0.0}
    sub_pbest = sub_operate(indi["pbest"], indi["travel"])
    mul_pbest = mul_operate(sub_pbest, c1)
    sub_gbest = sub_operate(gbest, indi["travel"])
    mul_gbest = mul_operate(sub_gbest, c2)
    child["velocity"] = add_operate(mul_pbest, mul_gbest)
    child["travel"] = add_operate(indi["travel"], child["velocity"])
    child["fitness"] = fitness(child, T)
    if child["fitness"] > indi["fitness"]:
        child["pbest"] = child["travel"]
    else:
        child["pbest"] = indi["pbest"]
    return child


def evolution(maxIterator, k1, k2, c1, c2, T):
    global population
    population = selectionBest(population)
    print population[0]["fitness"]
    bestFitness = 0.0
    nbIte = 0
    t = 0
    while t < maxIterator and nbIte < 200:
        fitness_mean = sum([population[i]["fitness"] for i, _ in enumerate(population)]) / len(population)
        fitness_max = max([population[i]["fitness"] for i, _ in enumerate(population)])
        print "t =", t
        i = 0
        while i < population_size:
            # print "i =", i
            j = random.randint(0, population_size - 1)
            while j == i:
                j = random.randint(0, population_size - 1)
            fitness_bar = max(population[i]["fitness"], population[j]["fitness"])
            if fitness_bar > fitness_mean:
                pc = k1 - k2 * (fitness_max - fitness_bar) / (fitness_max - fitness_mean)
            else:
                pc = k1
            rc = random.random()
            if rc < pc:
                child = crossover(population[i], population[j], T)
                if child != -1:
                    population.extend(child)
            i = i + 1
        # new_population = []
        # print "pop0 = ", population[0]["travel"]
        # for i, _ in enumerate(population):
        #     # print i
        #     new_population.append(mutation(population[i], c1, c2, T, population[0]["travel"]))
        population = selectionBest(population)
        print population[0]["fitness"]
        if population[0]["fitness"] - bestFitness >= 10**-3:
            bestFitness = population[0]["fitness"]
            nbIte = 0
        else:
            nbIte = nbIte + 1
        t = t + 1
    return population[0]


def getLifetime(travel):
    total = 0.0
    isDead = False
    while True:
        if isDead:
            break
        E_mc_now = E_mc
        E_now = E
        T = (E_max - E_mc_now) / e_mc
        temp_E_mc = E_max
        temp_E = [E_now[j] - T * e[j] for j, _ in enumerate(node_pos)]
        total = total + T
        if min(temp_E) <= 0 or temp_E_mc <= 0:
            isDead = True
        E_mc_now = temp_E_mc
        E_now = temp_E
        for i, item in enumerate(travel):
            if i != 0:
                time_to_move = distance(node_pos[travel[i - 1]], node_pos[travel[i]]) / velocity
            else:
                time_to_move = distance(depot, node_pos[travel[i]]) / velocity
            total = total + time_to_move
            temp_E_mc = E_mc_now - time_to_move * e_move
            temp_E = [E_now[j] - e[j] * time_to_move for j, _ in enumerate(node_pos)]
            if min(temp_E) <= 0 or temp_E_mc <= 0:
                isDead = True
                break
            E_mc_now = temp_E_mc
            E_now = temp_E
            time_to_charge = (E_sensor_max - E[item]) / (U - e[item])
            total = total + time_to_charge
            temp_E = [E_now[j] - e[j] * time_to_charge if j != item else E_sensor_max for j, _ in enumerate(node_pos)]
            temp_E_mc = E_mc_now - U * time_to_charge
            if min(temp_E) <= 0 or temp_E_mc <= 0:
                isDead = True
                break
        if not isDead:
            time_to_move = distance(depot, node_pos[travel[-1]]) / velocity
            E_mc_now = E_mc_now - time_to_move * e_move
            E_now = [E_now[j] - time_to_move * e[j] for j, _ in enumerate(node_pos)]
            total = total + time_to_move
            if min(E_now) <= 0 or E_mc_now <= 0:
                isDead = True
    return total


#  main task
index = 0

f = open("temp.csv", mode="w")
header = ["Bo Du Lieu", "Co Sac", "Khong Sac"]
writer = csv.DictWriter(f, fieldnames=header)
writer.writeheader()
nbRun = 5
while index < 5:
    print "Data Set ", index

    file_name = "HPSOGA/DataSet" + str(index) + ".csv"
    f = open(file_name, mode="w")
    header = ["Lan Chay", "Time", "Co Sac", "Khong Sac"]
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

    sum_lifetime = 0.0
    sum_time = 0.0
    confidence_interval = []
    for idRun in range(nbRun):
        start_time = time.time()
        random.seed(idRun)
        getData(file_name="thaydoisonode.csv", index=index)
        population_size = 10 * cpu_count()
        charge = [[charging(node, pos) for u, pos in enumerate(charge_pos)] for j, node in enumerate(node_pos)]
        delta = [[charge[j][u] - e[j] for u, _ in enumerate(charge_pos)] for j, _ in enumerate(node_pos)]
        T = get_T()
        population = [individual(T) for _ in range(population_size)]
        indi = evolution(maxIterator=5000, k1=0.9, k2=0.3, c1=0.5, c2=0.5, T=T)
        # print len(indi["travel"])
        life_time = getLifetime(indi["travel"])
        print life_time

        end_time = time.time()
        sum_lifetime = sum_lifetime + life_time
        sum_time = sum_time + end_time - start_time
        # write to file
        row = {"Lan Chay": "No." + str(idRun), "Time": end_time - start_time, "Co Sac": indi["fitness"],
               "Khong Sac": min([E[j] / e[j] for j, _ in enumerate(node_pos)])}
        writer.writerow(row)
        confidence_interval.append(life_time)
        idRun = idRun + 1

    row = {"Lan Chay": "Average", "Time": sum_time / nbRun, "Co Sac": sum_lifetime / nbRun,
           "Khong Sac": min([E[j] / e[j] for j, _ in enumerate(node_pos)])}
    writer.writerow(row)

    confidence = 0.95
    n = len(confidence_interval)
    m = mean(confidence_interval)
    std_err = sem(confidence_interval)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    row = {"Co Sac": h}
    writer.writerow(row)
    f.close()
    f.close()
    print "Done Data Set ", index
    index = index + 1

print "Done All"
