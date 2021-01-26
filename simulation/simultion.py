import os, sys
import random
import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import numpy as np
from model.LSTMEncoder import LSTM

sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "/home/sepehr/PycharmProjects/DAD/simulation/simulations/simulation.xml", "--fcd-output",
           "osma.anomaly.output", "--seed", '1']

traci.start(sumoCmd)
step = 0
f = open('../dataset/v0.4/simulation2.out', 'w')
anm = ''
d = [[], [], [], [], []]
while step < 100000:
    traci.simulationStep()
    if len(traci.vehicle.getIDList()) == 5:
        l = ''
        # flag = False
        # for i in traci.vehicle.getIDList():
        #     if i == anm:
        #         flag = True
        for i in traci.vehicle.getIDList():
            l += str(traci.vehicle.getSpeed(i)) + ' '
        # l += str(1 if flag else 0)
        l += '\n'
        f.write(l)
        anm = ''
        for i in range(0, 5):
            d[i].append(traci.vehicle.getSpeed(traci.vehicle.getIDList()[i]))

    #
    if step % 100 == 0:
        f.write('\n')
        # mean, std = np.mean(np.array(d)), np.std(np.array(d))
        # d -= mean
        # d /= std
        # plt.plot(range(0, len(d[0])), d[0])
        # plt.plot(range(0, len(d[0])), d[1])
        # plt.plot(range(0, len(d[0])), d[2])
        # plt.plot(range(0, len(d[0])), d[3])
        # plt.plot(range(0, len(d[0])), d[4])
        #
        # plt.grid(True)
        # plt.xlabel('time')
        # plt.ylabel('Speed')
        # plt.show()

        d = [[], [], [], [], []]
    if step % 100 == 20 and len(traci.vehicle.getIDList()) == 5:
        ids = traci.vehicle.getIDList()
        id = 4
        traci.vehicle.highlight(ids[id])
        traci.vehicle.setSpeedMode(ids[id], 0)
        traci.vehicle.setSpeedFactor(ids[id], 1.1)
        # traci.vehicle.setSpeed(ids[id], traci.vehicle.getSpeed(ids[id]) + 3)
        anm = ids[id]

    step += 1
