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
E_max = 10.0 #nang luong toi da cua MC
E_min = 0.0 # nang luong toi thieu cua MC
e_move = 0.1 #nang luong tieu thu moi giay cho viec di chuyen
E_move = [e_move * time_move_i for time_move_i in time_move] # nang luong tieu thu de di chuyen toi moi charge position
chargeRange = 10**10
velocity = 0.0
alpha = 600
beta = 30
charge = []
delta = [[0 for u, _ in enumerate(charge_pos)] for j, _ in enumerate(node_pos)]
E_thred = 0.0
near_charge = []

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
    global E_min
    global e_move
    global E_move
    global alpha
    global beta
    global velocity
    global E_thred
    global near_charge

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
    E_thred = 0.4 * E[0]

    charge_extend = []
    charge_extend.extend(charge_pos)
    charge_extend.append((0,0))
    time_move = [[distance(pos1, pos2) / velocity for pos2 in charge_extend] for pos1 in charge_extend]

    tmp = [time_move[i][i+1] * e_move for i in range(len(time_move) - 1)]
    E_move = [time_move[-1][0] * e_move]
    E_move.extend(tmp)

    near_charge = getNear()
    E_min = getEmin()

def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) * (node1[0] - node2[0])
                     + (node1[1] - node2[1]) * (node1[1] - node2[1]))

def charging(node, charge):
    d = distance(node, charge)
    if d > chargeRange:
        return 0
    else:
        return alpha / ((d + beta)**2)


# tra ve vi tri sac gan nhat tuong ung voi moi node
def getNear():
    near = [0 for j, _ in enumerate(node_pos)]
    for j, node in enumerate(node_pos):
        min_dis = 10**10
        index = -1
        for u, char in enumerate(charge_pos):
            d = charging(node, char)
            if d < min_dis:
                min_dis = d
                index = u
        near[j] = index
    return near

def getEmin():
    min_E = 0.0
    for i in range(len(charge_pos) + 1):
        for j in range(i+1):
            d = time_move[i][j] * e_move
            if d > min_E:
                min_E = d
    return min_E

def getProb(queue, current, t, lamda):
    # queue la hang doi cac node can duoc sac
    # current la vi tri hien tai cua MC
    # t la thoi gian hien tai
    # lamda la tham thuat toan
    # moi phan tu cua p se co dang (j, Ej, tj, pj) bao gom id cua node, nang luong cua node, thoi gian node request, va xac suat cua node do
    temp = [0.0 for _ in queue]
    for k, item in enumerate(queue):
        j, Ej, tj = item
        next_u = near_charge[j]  # diem sac gan nhat cua node j
        temp[k] = (j, Ej, tj, 1 - lamda * Ej / E_thred - (1 - lamda) * distance(current, charge_pos[next_u]) / (math.sqrt(2) * 1000))
    p = sorted(temp, key = itemgetter(1), reverse = True)
    return p

def life_time(lamda):
    total  = 0.0
    queue = [] # queue se luu vi tri va nang luong con lai cua node, moi phan tu se co cau truc la (j, E[j])
    E_now = copy.copy(E)
    E_mc_now = E_mc
    current = (0.0, 0.0) # vi tri hien tai cua MC, khoi tao tai vi tri co toa do (0, 0)
    t = 0.0

    while min(E_now) > 0.01:
        print "t = ", t, min(E_now), E_mc_now
        queue = []
        for j, node in enumerate(node_pos):
            if E_now[j] <= E_thred:
                queue.append((j, E_now[j], t))

        if not queue: # neu queue null thi nhay toi thoi gian gan nhat lam cho queue khac null
            delta_t = min([(E_now[j]-E_thred)/e[j] for j, _ in enumerate(node_pos)])
            t = t+delta_t
            E_now = [E_now[j] - delta_t * e[j] for j, _ in enumerate(node_pos)]
            continue
        if E_mc_now <= E_min: # neu nang luong cua MC khong du de di chuyen
            #print "Throw 1"
            #break
            E_mc_now = E_mc_now - time_move[-1][u]
            time_charge = (E_max - E_mc_now) / e_mc
            delta_t = time_move[-1][u] + time_charge
            temp_E_now = [E_now[j] - delta_t * e[j] for j, _ in enumerate(node_pos)]
            if min(temp_E_now) < 0.01:
                t = t + min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
                break
            else:
                t = t + delta_t
                E_now = temp_E_now
                E_mc_now = E_max
                current = (0.0, 0.0)
                continue
        else: # neu nang luong cua MC du de di chuyen
            #print "Throw 2"
            queue = getProb(queue=queue, current=current, t=t, lamda=lamda)
            j_now, Ej, tj, pj = queue[0]
            u_now = near_charge[j_now]
            delta_t = distance(current, charge_pos[u_now]) / velocity # thoi gian di chuyen den vi tri sac
            E_mc_now = E_mc_now - delta_t * e_move
            temp_E_now = [E_now[j] - delta_t * e[j] for j, _ in enumerate(node_pos)]
            if min(temp_E_now) < 0.01: #  neu mang chet trong qua trinh MC di chuyen
                t = t + min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
                break
            else: # neu mang van du nang luong de MC di chuyen toi vi tri j
                E_now = temp_E_now # nang luong moi cua cac node
                current = charge_pos[u_now] # vi tri moi cua MC
                t = t + delta_t
            # kiem tra xem nang luong cua Mc co du de sac khong
            if E_mc_now <= E_min:
                #print "throw 3"
                E_mc_now = E_mc_now - time_move[-1][u_now]
                time_charge = (E_max - E_mc_now) / e_mc
                delta_t = time_move[-1][u] + time_charge
                temp_E_now = [E_now[j] - delta_t * e[j] for j, _ in enumerate(node_pos)]
                if min(temp_E_now) < 0.01:
                    t = t + min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
                    break
                else:
                    t = t + delta_t
                    E_now = temp_E_now
                    E_mc_now = E_max
                    current = (0.0, 0.0)
                    continue
            # neu tat ca cac dieu kien ve nang nuong cua sensor va MC deu pass qua
            if charge[j_now][u_now] - e[j_now] <= 0:
                max_t = (E_mc_now - E_min) / charge[j_now][u_now]
            else:
                print (E[j_now] - E_now[j_now])/(charge[j_now][u_now] - e[j_now]), (E_mc_now - E_min) / charge[j_now][u_now]
                max_t = min((E[j_now] - E_now[j_now])/(charge[j_now][u_now] - e[j_now]), (E_mc_now - E_min) / charge[j_now][u_now])
            # max_t la thoi gian toi da Mc se dung sac cho node j
            temp_E_now = [E_now[j] - max_t * e[j] if j != j_now else E_now[j] + max_t * (charge[j][u_now] - e[j]) for j, _ in enumerate(node_pos)]
            if min(temp_E_now) < 0.01:
                print "throw 2.4.1", min(temp_E_now), max_t
                t = t + min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
                break
            else:
                print max_t
                t = t + max_t
                E_now = temp_E_now
                E_mc_now = E_mc_now - max_t * (charge[j_now][u_now] - e[j_now])
                print "throw 2.4.2", t, min(E_now), E_mc_now
    return t

def life_time_nonfix(lamda):
    total  = 0.0
    queue = [] # queue se luu vi tri va nang luong con lai cua node, moi phan tu se co cau truc la (j, E[j])
    E_now = copy.copy(E)
    E_mc_now = E_mc
    current = (0.0, 0.0) # vi tri hien tai cua MC, khoi tao tai vi tri co toa do (0, 0)
    t = 0.0

    while min(E_now) > 0.01:
        print "t = ", t, min(E_now), E_mc_now
        queue = []
        for j, node in enumerate(node_pos):
            if E_now[j] <= E_thred:
                queue.append((j, E_now[j], t))

        if not queue: # neu queue null thi nhay toi thoi gian gan nhat lam cho queue khac null
            delta_t = min([(E_now[j]-E_thred)/e[j] for j, _ in enumerate(node_pos)])
            t = t+delta_t
            E_now = [E_now[j] - delta_t * e[j] for j, _ in enumerate(node_pos)]
            continue
        if E_mc_now <= E_min: # neu nang luong cua MC khong du de di chuyen
            #print "Throw 1"
            #break
            E_mc_now = E_mc_now - time_move[-1][u]
            time_charge = (E_max - E_mc_now) / e_mc
            delta_t = time_move[-1][u] + time_charge
            temp_E_now = [E_now[j] - delta_t * e[j] for j, _ in enumerate(node_pos)]
            if min(temp_E_now) < 0.01:
                t = t + min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
                break
            else:
                t = t + delta_t
                E_now = temp_E_now
                E_mc_now = E_max
                current = (0.0, 0.0)
                continue
        else: # neu nang luong cua MC du de di chuyen
            #print "Throw 2"
            queue = getProb(queue=queue, current=current, t=t, lamda=lamda)
            j_now, Ej, tj, pj = queue[0]
            u_now = near_charge[j_now]
            delta_t = distance(current, charge_pos[u_now]) / velocity # thoi gian di chuyen den vi tri sac
            E_mc_now = E_mc_now - delta_t * e_move
            temp_E_now = [E_now[j] - delta_t * e[j] for j, _ in enumerate(node_pos)]
            if min(temp_E_now) < 0.01: #  neu mang chet trong qua trinh MC di chuyen
                t = t + min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
                break
            else: # neu mang van du nang luong de MC di chuyen toi vi tri j
                E_now = temp_E_now # nang luong moi cua cac node
                current = charge_pos[u_now] # vi tri moi cua MC
                t = t + delta_t
            # kiem tra xem nang luong cua Mc co du de sac khong
            if E_mc_now <= E_min:
                #print "throw 3"
                E_mc_now = E_mc_now - time_move[-1][u_now]
                time_charge = (E_max - E_mc_now) / e_mc
                delta_t = time_move[-1][u] + time_charge
                temp_E_now = [E_now[j] - delta_t * e[j] for j, _ in enumerate(node_pos)]
                if min(temp_E_now) < 0.01:
                    t = t + min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
                    break
                else:
                    t = t + delta_t
                    E_now = temp_E_now
                    E_mc_now = E_max
                    current = (0.0, 0.0)
                    continue
            # neu tat ca cac dieu kien ve nang nuong cua sensor va MC deu pass qua
            if charge[j_now][u_now] - e[j_now] <= 0:
                max_t = (E_mc_now - E_min) / charge[j_now][u_now]
            else:
                print (E[j_now] - E_now[j_now])/(charge[j_now][u_now] - e[j_now]), (E_mc_now - E_min) / charge[j_now][u_now]
                max_t = min((E[j_now] - E_now[j_now])/(charge[j_now][u_now] - e[j_now]), (E_mc_now - E_min) / charge[j_now][u_now])
            # max_t la thoi gian toi da Mc se dung sac cho node j
            temp_E_now = [E_now[j] - max_t * e[j] if j != j_now else E_now[j] + max_t * (charge[j][u_now] - e[j]) for j, _ in enumerate(node_pos)]
            if min(temp_E_now) < 0.01:
                print "throw 2.4.1", min(temp_E_now), max_t
                t = t + min([E_now[j] / e[j] for j, _ in enumerate(node_pos)])
                break
            else:
                print max_t
                t = t + max_t
                E_now = temp_E_now
                E_mc_now = E_mc_now - max_t * (charge[j_now][u_now] - e[j_now])
                print "throw 2.4.2", t, min(E_now), E_mc_now
    return t

# main task
index = 0

f = open("Compare_fixCharge_lifetime.csv", mode="w")
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
    delta = [[charge[j][u] - e[j] for u, _ in enumerate(charge_pos)] for j, _ in enumerate(node_pos)]
    print min(E[j] / e[j] for j, _ in enumerate(node_pos))
    t = life_time(0.5)

    row = {}
    row["Bo Du Lieu"] = "No." + str(index)
    row["Co Sac"] = t
    row["Khong Sac"] = min([E[j] / e[j] for j, _ in enumerate(node_pos)])
    writer.writerow(row)
    print "Done Data Set ", index
    index = index + 1

f.close()
print "Done All"

"""for item in indi["T"]:
    print item,
print
for item in indi["gen"]:
    for tmp in item:
        if tmp > 0:
            print round(tmp, 1),
    print"""





































