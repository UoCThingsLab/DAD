import sys
import xml.etree.ElementTree as ET
import math


class XMLParser:
    def __init__(self, address):
        self.address = address

    def read_txt(self):
        f = open(self.address, 'r')
        a = []
        d = []
        f.readline()
        for l in f.readlines():
            if l == '\n':
                a.append(d)
                d = []
            elif len(l.split(' ')) < 6:
                pass
            else:
                d.append(([float(i) for i in l.split(' ')[0:5]], float(l.split(' ')[5])))
        return a

    def read_xml(self, start, end):
        print(f"read time step {start} to {end}")
        vehicles = {}
        for _, elem in ET.iterparse(self.address, events=("end",)):
            if elem.tag == 'timestep':
                if float(elem.attrib['time']) < start:
                    continue
                if float(elem.attrib['time']) >= end:
                    break
                for vehicle in elem.iter('vehicle'):
                    attr = vehicle.attrib
                    data = [[
                        self.nm(attr, 'x'), self.nm(attr, 'y'), self.nm(attr, 'angle'), self.nm(attr, 'speed'),
                        self.nm(attr, 'lane', 'gneE0_0'), self.nm(attr, 'lane', 'gneE0_1'),
                        self.nm(attr, 'lane', 'gneE0_2'), attr['acceleration']]]
                    other_veh = []
                    max = []
                    for vehicle2 in elem.iter('vehicle'):
                        attr2 = vehicle2.attrib
                        if attr2['id'] != attr['id']:
                            dist = self.dist(self.nm(attr, 'x'), self.nm(attr, 'y'), self.nm(attr2, 'x'),
                                             self.nm(attr2, 'y'))
                            other_veh.append([[self.nm(attr2, 'x'), self.nm(attr2, 'y'), self.nm(attr2, 'angle'),
                                               self.nm(attr2, 'speed'),
                                               self.nm(attr2, 'lane', 'gneE0_0'), self.nm(attr2, 'lane', 'gneE0_1'),
                                               self.nm(attr2, 'lane', 'gneE0_2')], dist])
                            max.append(dist)
                    for o in other_veh:
                        if o[1] in sorted(max)[0:min(len(max), 4)]:
                            data.append(o[0])
                    for p in range(len(data), 5):
                        data.append([-1, -1, -1, -1, -1, -1, -1])

                    if attr['id'] not in vehicles:
                        vehicles[attr['id']] = []
                    vehicles[attr['id']].append(data)
        return vehicles

    def dist(self, x, y, x1, y1):
        return math.sqrt(((x - x1) ** 2) + ((y - y1) ** 2))

    def nm(self, attr, key, lane=''):
        if key != 'lane':
            data = float(attr[key])
        else:
            data = attr[key]
        if key == 'y':
            return (data + 40) / (51 + 40)
        if key == 'x':
            return (data - 36) / (966 - 36)
        if key == 'angle':
            return data / 360
        if key == 'speed':
            # return data / 20
            return data
        if key == 'lane':
            return 1 if data == lane else 0
