import sys
import xml.etree.ElementTree as ET


class XMLParser:
    def __init__(self, address):
        self.address = address

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
                    lead_car_speed = 1000
                    lead_car_distance = 1000
                    attr = vehicle.attrib
                    for vehicle2 in elem.iter('vehicle'):
                        attr2 = vehicle2.attrib
                        if attr2['id'] != attr['id'] and attr2['lane'] == attr['lane'] and float(attr2['pos']) > float(
                                attr['pos']) and float(attr2['pos']) - float(attr['pos']) < lead_car_distance:
                            lead_car_distance = float(attr2['pos']) - float(attr['pos'])
                            lead_car_speed = float(attr2['speed'])
                    if attr['id'] not in vehicles:
                        vehicles[attr['id']] = []
                    vehicles[attr['id']].append([float(attr['speed']), lead_car_distance, lead_car_speed])
        return vehicles

#
# parser = XMLParser('/home/sepehr/Sumo/downtown/osm.output')
# parser.read_xml(0, 10)
