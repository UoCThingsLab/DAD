import os, sys
import random
from torch import nn
import math
from model.Siamese import Siamese
from torch import tensor, device

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
import matplotlib.pyplot as plt


class Simulator:

    def __init__(self, address='../dataset/v0.1/abnormal.output', w=True):
        self.step = 0
        self.end = 100000
        sumoBinary = "sumo-gui"
        sumoCmd = [sumoBinary, "-c", "downtown/osm.sumocfg", "--seed", '1']
        traci.start(sumoCmd)

        self.id = 0

        self.w = w
        if self.w:
            self.file = open(address, 'w')
            self.s = []
            self.x = []
            self.y = []
        self.model = Siamese.load_from_checkpoint(
            "../checkpoint/LSTMEncoderLSTM--v_num=00-epoch=00-validation_loss=0.53419-train_loss=0.00000.ckpt"
        ).cuda()
        self.similarity = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.run()

    def init_variables(self):
        if self.step % 200 == 0:
            self.id = random.randint(0, 4)
            self.s = []
            self.x = []
            self.y = []
            if self.w:
                self.file.write('\n')

    def write(self):
        if len(traci.vehicle.getIDList()) == 5:
            max_speed = 100
            if traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger1':
                max_speed = 13.89 * 1.4
            elif traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger2':
                max_speed = 13.89 * 1
            elif traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger3':
                max_speed = 13.89 * 0.6
            # elif traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger4':
            #     max_speed = 13.89 * 0.7
            # elif traci.vehicle.getTypeID(traci.vehicle.getIDList()[0]) == 'veh_passenger5':
            #     max_speed = 13.89 * 0.6

            l = ''
            for i in traci.vehicle.getIDList():
                l += str(traci.vehicle.getSpeed(i)) + ',' + str(traci.vehicle.getPosition(i)[0]) + ',' + str(
                    traci.vehicle.getPosition(i)[1]) + ',' + str(traci.vehicle.getLaneIndex(i)) + ' '
            # l += str(id if flag else -1) + '\n'
            l += str(max_speed) + '\n'
            self.file.write(l)
            self.s.append([traci.vehicle.getSpeed(traci.vehicle.getIDList()[i]) for i in range(0, 5)])
            self.x.append(
                [(traci.vehicle.getPosition(traci.vehicle.getIDList()[i])[0] + 41.12) / (965.07 + 41.12) for i in
                 range(0, 5)])
            self.y.append(
                [(traci.vehicle.getPosition(traci.vehicle.getIDList()[i])[1] - 44.20) / (50.60 - 44.20) for i in
                 range(0, 5)])

    def over_speed(self):
        if self.step % 100 == 20 and len(traci.vehicle.getIDList()) == 5:
            ids = traci.vehicle.getIDList()
            traci.vehicle.setColor(ids[self.id], (255, 0, 0))
            traci.vehicle.setSpeedMode(ids[self.id], 0)
            traci.vehicle.setSpeedFactor(ids[self.id], traci.vehicle.getSpeedFactor(ids[self.id]) + 0.2)

    def under_speed(self):
        if self.step % 100 == 20 and len(traci.vehicle.getIDList()) == 5:
            ids = traci.vehicle.getIDList()
            traci.vehicle.setColor(ids[self.id], (255, 0, 0))
            traci.vehicle.setSpeedMode(ids[self.id], 0)
            traci.vehicle.setSpeedFactor(ids[self.id], traci.vehicle.getSpeedFactor(ids[self.id]) - 0.1)

    def lane_anomaly(self):
        if self.step % 100 > 20 and len(traci.vehicle.getIDList()) == 5:
            ids = traci.vehicle.getIDList()
            traci.vehicle.changeLane(ids[self.id],
                                     (traci.vehicle.getLaneIndex(ids[self.id]) + 1 + random.randint(0, 1)) % 3,
                                     2)

    def lane_change(self):
        ple = int(((self.step % 100) - (self.step % 10)) / 10)
        if len(traci.vehicle.getIDList()) == 5 and ple < 5 and self.step % 10 == 0:
            ids = traci.vehicle.getIDList()
            traci.vehicle.changeLane(ids[ple], (traci.vehicle.getLaneIndex(ids[ple]) + 1 + random.randint(0, 1)) % 3, 2)

    def normalize(self, d, max, min):
        return (d - min) / (max - min)

    def detect(self):
        if len(self.x) >= 7 and len(traci.vehicle.getIDList()) == 5:
            i = len(self.x) - 7

            min_x, min_y, min_d_x, min_d_y, min_dd_x, min_dd_y = 1000, 1000, 1000, 1000, 1000, 1000
            max_x, max_y, max_d_x, max_d_y, max_dd_x, max_dd_y = -1000, -1000, -1000, -1000, -1000, -1000
            for j in range(i + 2, i + 7):
                for k in range(0, 5):
                    if max_x < self.x[j][k]:
                        max_x = self.x[j][k]
                    if min_x > self.x[j][k]:
                        min_x = self.x[j][k]
                    if max_d_x < self.x[j][k] - self.x[j - 1][k]:
                        max_d_x = self.x[j][k] - self.x[j - 1][k]
                    if min_d_x > self.x[j][k] - self.x[j - 1][k]:
                        min_d_x = self.x[j][k] - self.x[j - 1][k]
                    if max_dd_x < (self.x[j][k] - self.x[j - 1][k]) - (self.x[j - 1][k] - self.x[j - 2][k]):
                        max_dd_x = (self.x[j][k] - self.x[j - 1][k]) - (self.x[j - 1][k] - self.x[j - 2][k])
                    if min_dd_x > (self.x[j][k] - self.x[j - 1][k]) - (self.x[j - 1][k] - self.x[j - 2][k]):
                        min_dd_x = (self.x[j][k] - self.x[j - 1][k]) - (self.x[j - 1][k] - self.x[j - 2][k])

                    if max_y < self.y[j][k]:
                        max_y = self.y[j][k]
                    if min_y > self.y[j][k]:
                        min_y = self.y[j][k]
                    if max_d_y < self.y[j][k] - self.y[j - 1][k]:
                        max_d_y = self.y[j][k] - self.y[j - 1][k]
                    if min_d_y > self.y[j][k] - self.y[j - 1][k]:
                        min_d_y = self.y[j][k] - self.y[j - 1][k]
                    if max_dd_y < (self.y[j][k] - self.y[j - 1][k]) - (self.y[j - 1][k] - self.y[j - 2][k]):
                        max_dd_y = (self.y[j][k] - self.y[j - 1][k]) - (self.y[j - 1][k] - self.y[j - 2][k])
                    if min_dd_y > (self.y[j][k] - self.y[j - 1][k]) - (self.y[j - 1][k] - self.y[j - 2][k]):
                        min_dd_y = (self.y[j][k] - self.y[j - 1][k]) - (self.y[j - 1][k] - self.y[j - 2][k])
            seq = []
            for j in range(i + 2, i + 7):
                seq.append([
                    [self.normalize((self.x[j][k] - self.x[j - 1][k]) - (self.x[j - 1][k] - self.x[j - 2][k]), max_dd_x,
                                    min_dd_x)
                     for k in
                     range(0, 5)],
                    [self.normalize(self.x[j][k] - self.x[j - 1][k], max_d_x,
                                    min_d_x)
                     for k in
                     range(0, 5)],
                    [self.normalize(self.x[j][k], max_x, min_x)
                     for k in
                     range(0, 5)],
                    [self.normalize((self.y[j][k] - self.y[j - 1][k]) - (self.y[j - 1][k] - self.y[j - 2][k]), max_dd_y,
                                    min_dd_y)
                     for k in
                     range(0, 5)],
                    [self.normalize(self.y[j][k] - self.y[j - 1][k], max_d_y,
                                    min_d_y)
                     for k in
                     range(0, 5)],
                    [self.normalize(self.y[j][k], max_y, min_y)
                     for k in
                     range(0, 5)],
                ])
            seq = tensor(seq, device=device('cuda'))
            decoded = []
            for j in range(0, 5):
                list = [[seq[i][k][j].item() for k in range(0, 6)] for i in
                        range(0, 5)]
                list = tensor(list, device=device('cuda'))
                ـ, d = self.model(list)
                decoded.append(d[0][0])

            for i in range(0, 5):
                loss2 = 0
                for j in range(0, 5):
                    if i != j:
                        l = self.similarity(decoded[i], decoded[j])
                        l = (-1 * l) + 1
                        loss2 += l
                loss2 /= 4
                ids = traci.vehicle.getIDList()
                if loss2 >= 0.0068:
                    traci.vehicle.highlight(ids[i], color=(255, 0, 0))
                else:
                    traci.vehicle.highlight(ids[i], color=(0, 255, 0))

    def run(self):
        while self.step < self.end:
            self.init_variables()
            traci.simulationStep()
            # self.lane_change()
            # self.lane_anomaly()

            if random.randint(0, 2) == 1:
                self.under_speed()
            else:
                self.over_speed()
            self.detect()
            if self.w:
                self.write()
            self.step += 1


Simulator()
