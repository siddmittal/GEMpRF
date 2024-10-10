import json
import os
import xml.etree.ElementTree as ET

class XMLUtils:
    @classmethod
    def update_xml_node_value(cls, xml_filepath: str, xpath : str, new_value: str):
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        # fMRI datapath: change the measured_data_filepath in the config file
        xml_node = root.find(xpath)   
        xml_node.text = new_value
        
        # Write the modified XML back to the file
        tree.write(xml_filepath)


    @classmethod
    def update_xml_node_attribute(cls, xml_filepath: str, xpath : str, attribute_name: str, new_attribute_value: str):
        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        # fMRI datapath: change the measured_data_filepath in the config file
        xml_node = root.find(xpath)   
        xml_node.set(attribute_name, new_attribute_value)
        
        # Write the modified XML back to the file
        tree.write(xml_filepath)

