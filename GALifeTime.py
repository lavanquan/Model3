import math
import random
from operator import itemgetter
import copy
import pandas as pd
from ast import literal_eval
from multiprocessing import cpu_count, Process, Pipe

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

# cac tham so thuat toan
population = []

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
    charge_extend.append((0, 0))
    time_move = [[distance(pos1, pos2) / velocity for pos2 in charge_extend] for pos1 in charge_extend]

    tmp = [time_move[i][i + 1] * e_move for i in range(len(time_move) - 1)]
    E_move = [time_move[-1][0] * e_move]
    E_move.extend(tmp)

def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) * (node1[0] - node2[0])
                     + (node1[1] - node2[1]) * (node1[1] - node2[1]))

def charge(node, charge):
    d = distance(node, charge)
    if d > chargeRange:
        return 0
    else:
        return alpha / ((d + beta)**2)

def fitness(indi):
    total = 0
    for index in range(indi["num_gen"]-1):
        total += indi["T"][index] + sum(indi["gen"][index]) + sum(time_move)
    row = indi["num_gen"]-1
    index = numCharge + 1
    for i, item in reversed(list(enumerate(indi["gen"][row]))):
        if item != 0:
            index = i
            break
    total += indi["T"][row] + sum(indi["gen"][row][:index+1]) + sum(time_move[:index+1])
    return total + indi["remain"]


def getRound(E_mc_now, E_now):
    x = [0 for u, _ in enumerate(charge_pos)]
    T = 0
    # mang chua nang luong cua cac sensor khi thuc hien chu ki moi
    eNode = [E_now[j] / e[j] for j, _ in enumerate(node_pos)]
    T_max = min(min(eNode), (E_max - E_mc_now) / e_mc)
    T_min = max(0, (sum(E_move) - E_mc_now) / e_mc)

    if T_max >= T_min:
        # gia tri cua T
        T = T_max - 0.2 * abs(T_max - T_min) * random.random()
        # cac thong so nang luong cua MC va cua sensor ngay truoc vi tri sac u
        E_mc_new = E_mc_now + T * e_mc - sum(E_move)

        a = [sum([charge[j][u] for j, _ in enumerate(node_pos)]) for u, _ in enumerate(charge_pos)]
        p = [u for u, _ in enumerate(charge_pos)]
        random.shuffle(p)
        for u in p:
            x[u] = random.random() * E_mc_new / a[u]
            E_mc_new -= a[u] * x[u]
        return [T, x]
    else:
        return -1

def genRound(E_mc_now, E_now):
    # sinh mot gia tri T thoa man rang buoc
    # E_mc_now: nang luong cua MC tai vi tri hien tai
    # E_now: nang luong cua cac sensor tai thoi diem hien tai
    T = 0
    x = [0 for pos in charge_pos]
    remain = 0

    # mang chua nang luong cua cac sensor khi thuc hien chu ki moi
    eNode = [E_now[j] / e[j] - time_move[0] for j, _ in enumerate(node_pos)]
    T_max = min(min(eNode), (E_max - E_mc_now) / e_mc)
    T_min = max(0, (sum(E_move) - E_mc_now) / e_mc)

    if T_max >= T_min:
        # gia tri cua T
        T = random.random() * (T_min - T_max) + T_max
        # cac thong so nang luong cua MC va cua sensor ngay truoc vi tri sac u
        E_mc_new = E_mc_now + T * e_mc
        E_new = [E_now[j] - T * e[j] for j, node in enumerate(node_pos)]

        for u, pos in enumerate(charge_pos):
            # max_charge: khoang thoi gian lon nhat de nang luong sac khong vuot qua nang luong cho phep
            max_charge = (E_mc_new - sum(E_move[u:])) / sum([charge[j][u] for j, node in enumerate(node_pos)])

            # E_remain: nang luong con lai cua cac sensor truoc khi MC den u
            E_remain = [E_new[j] - time_move[u] * e[j] for j, _ in enumerate(node_pos)]
            if min(E_remain) < 0:
                remain = min([E_new[j] / e[j] for j, _ in enumerate(node_pos)])
                break
            else:
                # low, upp: thoi gian sac lon nhat va nho nhat de cac sensor van con song sau khi sac tai u
                low = [(time_move[u] * e[j] - E_new[j]) / (charge[j][u] - e[j]) for j, _ in enumerate(node_pos) if
                       charge[j][u] - e[j] > 0]
                upp = [(time_move[u] * e[j] - E_new[j]) / (charge[j][u] - e[j]) for j, _ in enumerate(node_pos) if
                       charge[j][u] - e[j] < 0]
                mid = [time_move[u] * e[j] - E_new[j] for j, _ in enumerate(node_pos) if charge[j][u] == 0]

                # tinh toan can tren va can duoi cua x[u]
                if upp:
                    x_max = min(min(upp), max_charge)
                else:
                    x_max = max_charge
                if low:
                    x_min = max(max(low), 0)
                else:
                    x_min = 0

            # tinh gia tri cua x[u]: thoi gian dung tai vi tri sac thu u
            if mid and max(mid) > 0:
                break
            elif x_max < x_min:
                break
            else:
                x[u] = random.random() * (x_min - x_max) + x_max

                p = [min(charge[j][u] * x[u], E[j] - E_new[j] + time_move[u] * e[j]) for j, _ in enumerate(node_pos)]
                # x[u] = max([p[j] / charge[j][u] for j, _ in enumerate(node_pos)])

                # tinh lai gia tri nang luong cua MC va cacs cam bien sau khi sac tai vi tri u
                E_mc_new = E_mc_new - E_move[u] - sum(p)
                E_new = [E_new[j] + p[j] - (time_move[u] + x[u]) * e[j] for j, _ in enumerate(node_pos)]

        remain = min([E_new[j] / e[j] for j, _ in enumerate(node_pos)])
        return T, x, remain
    else:
        return -1

def individual():
    indi = {"T": [], "gen": [], "fitness": 0.0, "num_gen": 1}  #so luong chu ki sac
    #so luong chu ki sac
    E_mc_now = E_mc
    E_now = E
    while E_mc_now > 0 and E_now > 0:
        tmp = getRound(E_mc_now, E_now)
        if tmp == -1:
            break
        T, x = tmp
        indi["T"].append(T)
        indi["gen"].append(x)
        E_mc_now = E_mc_now + T * e_mc
        E_now = [E_now[j] - T * e[j] for j, _ in enumerate(node_pos)]
        for u, _ in enumerate(charge_pos):
            p = [min(charge[j][u] * x[u], E[j] - E_now[j] + (time_move[u] + x[u]) * e[j]) for j, node in enumerate(node_pos)]
            E_mc_now = E_mc_now - sum(p) - time_move[u]
            E_now = [E_now[j] + p[j] - (time_move[u] + x[u]) * e[j] for j, _ in enumerate(charge_pos)]
            if E_mc_now <= 0 or min(E_now) <= 0:
                break
    indi = injust(indi)
    return indi

def selection(new_population):
    new_list = sorted(new_population, key = itemgetter("fitness"), reverse = True)
    return new_list[:population_size]

def BLX(gen1, gen2):
    temp = []
    for x, y in zip(gen1, gen2):
        low = max(min(x, y) - abs(x - y) / 2.0, 0.0)
        upp = max(x, y) + abs(x - y) / 2.0
        temp.append(random.random() * (upp - low) + low)
    return temp


def mutation(indi):
    if indi["gen"][-1][-1] == 0:
        return -1

    off = {}
    off["num_gen"] = indi["num_gen"]
    off["T"] = copy.copy(indi["T"])
    off["gen"] = copy.copy(indi["gen"])

    E_mc_now = E_mc
    E_now = [E[j] for j, _ in enumerate(node_pos)]
    energy_add = [0 for k, _ in enumerate(node_pos)]
    for k, _ in enumerate(off["gen"]):
        route = indi["gen"][k]
        E_mc_now = E_mc_now + off["T"][k] * e_mc
        E_now = [E_now[j] - off["T"][k] * e[j] for j, _ in enumerate(node_pos)]
        for u, pos in enumerate(charge_pos):
            p = [min(charge[j][u] * route[u], E[j] - E_now[j] + (time_move[u] + route[u]) * e[j]) for j, node in enumerate(node_pos)]
            E_mc_now = E_mc_now - sum(p) - E_move[u]
            E_now = [E_now[j] + p[j] - (time_move[u] + route[u]) * e[j] for j, _ in enumerate(node_pos)]
        E_mc_now -= E_move[-1]
        E_now = [E_now[j] - time_move[-1] * e[j] for j, _ in enumerate(node_pos)]

    if min(E_now) < 0 or E_mc_now < 0:
        # mang khong du nang luong de sinh round moi
        return -1
    else:
        tmp = getRound(E_mc_now, E_now)
        if tmp != -1:
            off["T"].append(tmp[0])
            off["gen"].append(tmp[1])
            off = injust(off)
            return off
        else:
            # mang khong du nang luong de sinh round moi
            return -1


def injust(indi):
    E_mc_now = E_mc
    E_now = [item for item in E]

    off = {}
    off["T"] = []
    off["gen"] = []
    off["remain"] = -1

    isStop = False
    idRound = 0

    for index, gen in enumerate(indi["gen"]):
        T = min(indi["T"][index], (E_max - E_mc_now) / e_mc)
        temp_E = [E_now[j] - T * e[j] for j, _ in enumerate(node_pos)]
        temp_E_mc = E_mc_now + T * e_mc
        eNode = min([temp_E[j] - time_move[0] * e[j] for j, _ in enumerate(node_pos)])

        if eNode < 0 or temp_E_mc < sum(E_move):
            if index == 0:
                off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
            else:
                off["remain"] = min([E_now[j] / e[j] + time_move[-1] for j, _ in enumerate(node_pos)])
            break
        else:
            E_mc_now = temp_E_mc
            E_now = temp_E
            off["T"].append(T)
            x = [0 for u, _ in enumerate(charge_pos)]
            for u, pos in enumerate(charge_pos):
                p = [min(charge[j][u] * gen[u], E[j] - E_now[j] + time_move[u] * e[j]) for j, node in enumerate(node_pos)]
                temp_E_mc = E_mc_now - sum(p) - E_move[u]
                temp_E = [E_now[j] + p[j] - (time_move[u] + gen[u]) * e[j] for j, _ in enumerate(node_pos)]

                if min(temp_E) < 0 or temp_E_mc < sum(E_move[u + 1:]):
                    isStop = True
                    break
                else:
                    x[u] = gen[u]
                    E_mc_now = temp_E_mc
                    E_now = temp_E
            off["gen"].append(x)

            if not isStop:
                E_mc_now = E_mc_now - E_move[-1]
                E_now = [E_now[j] - time_move[-1] * e[j] for j, _ in enumerate(node_pos)]
            else:
                break

    off["num_gen"] = len(off["gen"])
    if off["remain"] == -1:
        if isStop:
            off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
        else:
            off["remain"] = min([E_now[j] / e[j] + time_move[-1] for j, _ in enumerate(node_pos)])
    off["fitness"] = fitness(off)
    return off


def crossover(father, mother):
    off = {}
    f = father["num_gen"]
    m = mother["num_gen"]

    if f == m:
        off["num_gen"] = f
        off["T"] = BLX(father["T"], mother["T"])
        off["gen"] = [BLX(father["gen"][i], mother["gen"][i]) for i, _ in enumerate(father["gen"])]
    elif f > m:
        off["num_gen"] = f

        tempT = [mother["T"][i] if i < m else 0 for i, _ in enumerate(father["T"])]
        off["T"] = BLX(father["T"], tempT)

        zeroGen = [0 for _ in charge_pos]
        tempGen = [mother["gen"][i] if i < m else zeroGen for i, _ in enumerate(father["gen"])]
        off["gen"] = [BLX(father["gen"][i], tempGen[i]) for i, _ in enumerate(father["gen"])]
    else:
        off["num_gen"] = m

        tempT = [father["T"][i] if i < f else 0 for i, _ in enumerate(mother["T"])]
        off["T"] = BLX(mother["T"], tempT)

        zeroGen = [0 for _ in charge_pos]
        tempGen = [father["gen"][i] if i < f else zeroGen for i, _ in enumerate(mother["gen"])]
        off["gen"] = [BLX(mother["gen"][i], tempGen[i]) for i, _ in enumerate(mother["gen"])]
    off = injust(off)
    off["fitness"] = fitness(off)
    return off

def evol(start, end, pc, pm, connection):
    global population
    sub_pop = []
    count = 0
    i = start
    while i < end:
        rc = random.random()
        rm = random.random()
        if rc <= pc:
            j = random.randint(0, population_size - 1)
            while j == i:
                j = random.randint(0, population_size - 1)
            child = crossover(population[i], population[j])
            if rm <= pm:

                mutated_child = mutation(child)
                if mutated_child != -1:
                  #  print True
                    count += 1
                    sub_pop.append(mutated_child)
            else:
                sub_pop.append(child)
        i += 1
    connection.send([count, sub_pop])
    connection.close()

"""def evolution(maxIterator, p_c, p_m):
    global population

    count = 0
    t = 0
    while t < maxIterator:
        for i in range(population_size):
            r_c = random.random()
            r_m = random.random()
            if r_c <= p_c:
                j = random.randint(0, population_size - 1)
                while j == i:
                    j = random.randint(0, population_size - 1)
                child = crossover(population[i], population[j])
                population.append(child)
            if r_m <= p_m:
                child = mutation(population[i])
                if child != -1:
                    count += 1
                    population.append(child)
        population = selection(population)
        print t, count, round(population[0]["fitness"], 1), population[0]["num_gen"]
        t += 1
    return population[0]"""

def evolution(maxIterator, pc, pm):
    global population
    t = 0
    while t < maxIterator:
        count = 0
        nproc = cpu_count()
        process = []
        connection = []
        for pid in range(nproc):
            connection.append(Pipe())
        for pid in range(nproc):
            pro = Process(target=evol, args=(5 * pid, 5 * (pid + 1), pc, pm, connection[pid][1]))
            process.append(pro)
            pro.start()
        for pid in range(nproc):
            nbMutation, sub_pop = connection[pid][0].recv()
            count += nbMutation
            population.extend(sub_pop)
            process[pid].join()
        try:
            population = selection(population)
        except:
            print population
            break
        max_gen = population[0]["num_gen"]
        population = selection(population)
        if t % 50 == 0:
            print t, count, round(population[0]["fitness"], 1), max_gen
        t += 1
    population = selection(population)
    return population[0]

def injustNewModel(indi):
    E_mc_now = E_mc
    E_now = [item for item in E]

    off = {}
    off["T"] = []
    off["gen"] = []
    off["remain"] = -1

    isStop = False
    for index, gen in enumerate(indi["gen"]):
        T_max = (E_max - E_mc_now) / e_mc
        T = min(T_max, indi["T"][index])
        temp_E = [E_now[j] - T * e[j] for j, _ in enumerate(node_pos)]
        temp_E_mc = E_mc_now + T * e_mc
        # row chua vi tri va thoi gian sac cua nhung diem sac co thoi gian sac > 0
        row = [(u, xu) for u, xu in enumerate(gen) if xu > 0]
        # neu tat ca cac xu = 0 thi bo qua chu ki nay va tinh toan den chu ki tiep theo
        if not row:
            isStop = True
            continue
        u_first, _ = row[0]
        eNode = min([temp_E[j] - time_move[-1][u_first] * e[j] for j, _ in enumerate(node_pos)])

        if eNode < 0 or temp_E_mc < sum(E_move):
            off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
            """if index == 0:
                off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
            else:
                pre_row = [(u, xu) for u, xu in enumerate(indi["gen"][index-1]) if xu > 0]
                pre_u, _ = pre_row[-1]
                off["remain"] = min([E_now[j] / e[j] + time_move[-1][pre_u] for j, _ in enumerate(node_pos)])"""
            break
        else:
            E_mc_now = temp_E_mc
            E_now = temp_E
            off["T"].append(T)
            x = [0 for u, _ in enumerate(charge_pos)]
            for id, current in enumerate(row):
                u, xu = current
                if id == 0:
                    time = time_move[-1][u]
                else:
                    pre = row[id - 1]
                    pre_u, pre_xu = pre
                    time = time_move[pre_u][u]
                p = [min(charge[j][u] * xu, E[j] - E_now[j] + (time + xu) * e[j]) for j, node in enumerate(node_pos)]
                temp_E_mc = E_mc_now - sum(p) - time * e_mc
                temp_E = [E_now[j] + p[j] - (time + xu) * e[j] for j, _ in enumerate(node_pos)]

                if min(temp_E) < 0 or temp_E_mc < sum(E_move[u + 1:]):
                    isStop = True
                    break
                else:
                    x[u] = xu
                    E_mc_now = temp_E_mc
                    E_now = temp_E
            off["gen"].append(x)

            if not isStop:
                u_last, _ = row[-1]
                E_mc_now = E_mc_now - time_move[-1][u_last] * e_mc
                E_now = [E_now[j] - time_move[-1][u_last] * e[j] for j, _ in enumerate(node_pos)]
            else:
                break

    """idRound = len(off["gen"]) - 1
    #    print len(off["gen"]), idRound, indi["num_gen"]
    while idRound < indi["num_gen"] - 1:
        off["T"].append(0.0)
        off["gen"].append([0.0 for _ in charge_pos])
        idRound += 1"""

    off["num_gen"] = len(off["gen"])
    if off["remain"] == -1:
        off["remain"] = min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
    off["fitness"] = fitnessNewModel(off)
    #   print off["num_gen"]
    return off

def fitnessNewModel(indi):
    total = 0.0
    for index in range(indi["num_gen"]):
        total += indi["T"][index]
        tmp = indi["gen"][index]
        row = [(u, xu) for u, xu in enumerate(tmp) if xu > 0]
        if not row:
            continue
        for id, current in enumerate(row):
            u, xu = current
            if id == 0:
                time = time_move[-1][u]
            else:
                pre = row[id - 1]
                pre_u, pre_xu = pre
                time = time_move[pre_u][u]
            total += time + xu
        if index != indi["num_gen"] - 1:
            last_u, _ = row[-1]
            total += time_move[-1][last_u]
    total += indi["remain"]
    return total

# main task
getData(file_name="data.csv", index=0)

population_size = 60
charge = [[charge(node, pos) for u, pos in enumerate(charge_pos)] for j, node in enumerate(node_pos)]
population = [individual() for _ in range(population_size)]
indi = evolution(500, 0.8, 0.5)
best = injustNewModel(indi)
for i, gen in enumerate(indi["gen"]):
    print indi["T"][i],
    for item in gen:
        print round(item, 1),
print
for i, gen in enumerate(best["gen"]):
    print best["T"][i],
    for item in gen:
        print round(item, 1),














