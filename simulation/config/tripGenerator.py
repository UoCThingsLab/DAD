import random

# streets = ['gneE6', 'gneE16', 'gneE15']
g = [5, 10, 15]
f = open("trip", "w")
k = 0
for i in range(0, 360000, 200):
    c = random.randint(1, 3)
    # s = streets[random.randint(0, 2)]
    for j in range(0, 5):
        f.write(
            f'<trip id=\"veh{k * 10 + j}\" type=\"veh_passenger{c}\" depart=\"{i + 3 * j}.00\"  departLane=\"random\" departSpeed=\"max\" from=\"gneE2\" to=\"gneE3\"/>\n')
    k += 1
