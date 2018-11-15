import os
import cv2
from lxml import etree
import xml.etree.cElementTree as ET


def write_xml(folder, img, obj, savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    image = cv2.imread(img.path)
    height, width, depth = image.shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = img.name
    ET.SubElement(annotation, 'segmented').text = '0'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    ob = ET.SubElement(annotation, 'object')
    ET.SubElement(ob, 'name').text = obj
    ET.SubElement(ob, 'pose').text = 'Unspecified'
    ET.SubElement(ob, 'truncated').text = '0'
    ET.SubElement(ob, 'difficult').text = '0'

    #get total width and height
    bbox = ET.SubElement(ob, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = str(0)
    ET.SubElement(bbox, 'ymin').text = str(0)
    ET.SubElement(bbox, 'xmax').text = str(width)
    ET.SubElement(bbox, 'ymax').text = str(height)

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir, img.name.replace('ppm', 'xml')) #rename to xml file
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


if __name__ == '__main__':
    folder = '/home/lee/Project/autocar/detectOject/BelgiumTSC_Training/Training/00022'
    for n, img in enumerate(os.scandir(folder)):
        obj = 'ban-sign'
        savedir = 'ban-sign'
        write_xml(folder, img, obj, savedir)