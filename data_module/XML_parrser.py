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
                    attr = vehicle.attrib
                    if attr['id'] not in vehicles:
                        vehicles[attr['id']] = []
                    vehicles[attr['id']].append([float(attr['x']), float(attr['y']), float(attr['speed'])])
        return vehicles

#
# parser = XMLParser('/home/sepehr/Sumo/downtown/osm.output')
# parser.read_xml(0, 10)
