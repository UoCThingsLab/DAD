import os, sys
import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "/home/sepehr/PycharmProjects/DAD/simulation/simulations/simulation.xml", "--fcd-output",
           "osma.anomaly.output", "--collision.action", 'remove']

traci.start(sumoCmd)
step = 0
f = open('simulation2.out', 'w')
while step < 100000:
    traci.simulationStep()
    if len(traci.vehicle.getIDList()) == 5:
        l = ''
        for i in traci.vehicle.getIDList():
            l += str(traci.vehicle.getSpeed(i)) + ' '
        l += '\n'
        f.write(l)
    if step % 100 == 0:
        f.write('\n')
    if step % 130 == 0 and len(traci.vehicle.getIDList()) == 5:
        ids = traci.vehicle.getIDList()
        id = 0
        traci.vehicle.highlight(ids[id])
        traci.vehicle.setSpeedMode(ids[id], 0)
        traci.vehicle.setSpeed(ids[id], 17 + random.uniform(0, 3))

    step += 1
