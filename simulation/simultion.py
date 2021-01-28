import os, sys
import random
import matplotlib.pyplot as plt
from torch import tensor, device

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from model.LSTMEncoderLSTM import LSTMEncoderLSTM
import traci
import numpy as np
from model.LSTMEncoder import LSTM

sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "/home/sepehr/PycharmProjects/DAD/simulation/simulations/simulation.xml", "--fcd-output",
           "osma.anomaly.output", "--seed", '1']
#
# model = LSTMEncoderLSTM.load_from_checkpoint(
#     "/home/sepehr/PycharmProjects/DAD/lightning_logs/version_54/checkpoints/LSTMEncoderLSTM--v_num=00-epoch=99-validation_loss=0.00-train_loss=0.00.ckpt"
# ).cuda()
traci.start(sumoCmd)
step = 0
f = open('../dataset/v0.4/abnormal2.output', 'w')
anm = ''
d = [[], [], [], [], []]
id = 0
max_speed = 0
while step < 100000:
    traci.simulationStep()
    if len(traci.vehicle.getIDList()) == 5:
        l = ''
        for i in traci.vehicle.getIDList():
            l += str(traci.vehicle.getSpeed(i)) + ' '
        l += str(max_speed) + '\n'
        f.write(l)
        anm = ''
        for i in range(0, 5):
            d[i].append(traci.vehicle.getSpeed(traci.vehicle.getIDList()[i]))

    #
    if step % 200 == 0:
        f.write('\n')
        max_speed = 13.89 * traci.vehicle.getSpeedFactor(traci.vehicle.getIDList()[0])
        # # mean, std = np.mean(np.array(d)), np.std(np.array(d))
        # # d -= mean
        # # d /= std
        p = []
        # plt.plot(range(0, len(d[id])), d[id])
        # for i in range(0, len(d[id]) - 5):
        #     t = [[d[l][j] / 20 for l in range(0, 5)] for j in range(i, i + 5)]
        #     t = tensor(t, device=device('cuda'))
        #     t = model(t)
        #     p.append(t[0][0][id].item() * 20)
        # plt.plot(range(5, len(p) + 5), p)

        # plt.plot(range(0, len(d[0])), d[0], alpha=0.3)
        # plt.plot(range(0, len(d[0])), d[1], alpha=0.3)
        # plt.plot(range(0, len(d[0])), d[2], alpha=0.3)
        # plt.plot(range(0, len(d[0])), d[3], alpha=0.3)
        # plt.plot(range(0, len(d[0])), d[4], alpha=0.3)
        # # # #
        # id = random.randint(0, 4)
        # plt.grid(True)
        # plt.xlabel('time')
        # plt.ylabel('Speed')
        # plt.show()

        d = [[], [], [], [], []]
    if step % 100 == 20 and len(traci.vehicle.getIDList()) == 5:
        ids = traci.vehicle.getIDList()
        traci.vehicle.highlight(ids[id])
        traci.vehicle.setSpeedMode(ids[id], 0)
        traci.vehicle.setSpeedFactor(ids[id], traci.vehicle.getSpeedFactor(ids[id]) - 0.1)
        # traci.vehicle.setSpeed(ids[id], traci.vehicle.getSpeed(ids[id]) + 3)

    step += 1
