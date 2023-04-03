from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
import csv
import torch
from torchvision.transforms import Normalize
import numpy as np
from multiprocessing import Pool, Manager, Process
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np

import random
import pickle
from ultradict import UltraDict


# images_path = "../kaist/kaist-cvpr15"
# address_order = dict()
# with open("../kaist/kaist-cvpr15/imageSets/train-all-20.txt", 'r') as ft:
#     for idx,line in enumerate(ft):
#         line = line.strip("\n")
#         address_order[line] = idx
#
# def read_image(line, channel):
#     image_id = line.split("/")
#     if channel == 'rgb':
#         folder = 'visible'
#     else:
#         folder = 'lwir'
#     im_path = os.path.join(images_path, image_id[0], image_id[1], folder, image_id[2] + ".jpg")
#     out = read_image(im_path)
#     return out
#
#
# def get_files(self, file_address):
#     rgb_img = self.read_image(file_address, channel='rgb')
#     thermal_img = self.read_image(file_address, channel='thermal')
#     rgb_img = rgb_img / 255
#     thermal_img = torch.mean(thermal_img.float(), 0)
#     thermal_img = thermal_img.unsqueeze(0)
#     thermal_img = thermal_img / 255
#
#     idx = address_order[file_address]
#     shared_dict = UltraDict(name='shared_dict')
#     shared_dict[idx] = (file_address, rgb_img, thermal_img)
#
# p = Pool(12)
# p.map(get_files, list(address_order.keys()))

global mem_name

class Kaist(Dataset):
    def __init__(self, data_path, set_name="imageSets/train-all-20.txt", only_first=None, normalize=True, test=False):
        self.data_path = data_path
        self.images_path = os.path.join(self.data_path, 'images')
        self.data_dict = dict()
        # self.data_dict = UltraDict(name='shared_dict')
        # self.dummy_dict = UltraDict(name='dummy')
        self.set_path = os.path.join(self.data_path, set_name)
        self.filter = filter
        self.normalize = normalize
        mean_color = torch.tensor([0.3464423, 0.32171702, 0.28396806])
        std_color = torch.tensor([0.23573601, 0.22641623, 0.22726215])
        mean_thermal = torch.tensor(0.16)
        std_thermal = torch.tensor(0.081)
        self.color_normalizer = Normalize(mean_color, std_color)
        self.thermal_normalizer = Normalize(mean_thermal, std_thermal)
        self.address_order = dict()
        with open(self.set_path, 'r') as ft:
            for idx, line in enumerate(ft):
                if only_first is not None:
                    if idx == only_first:
                        break
                line = line.strip("\n")
                self.address_order[line] = idx
        # print(list(self.address_order.keys()))
        if not test:
            for address in self.address_order.keys():
                self.get_files(address)
        print('Loaded :', len(self.data_dict.keys()))


    #     p = Pool(12)
    #     p.map(self.dummy, list(self.address_order.keys()))
    #
    # def dummy(self, address):
    #     print(address)



    def get_files(self, file_address):
        rgb_img = self.read_image(file_address, channel='rgb')
        thermal_img = self.read_image(file_address, channel='thermal')
        if self.normalize:
            rgb_img = rgb_img / 255
            thermal_img = torch.mean(thermal_img.float(), 0)
            thermal_img = thermal_img.unsqueeze(0)
            thermal_img = thermal_img / 255
            rgb_img = self.color_normalizer(rgb_img)
            thermal_img = self.thermal_normalizer(thermal_img)
        idx = self.address_order[file_address]
        self.data_dict[idx] = (file_address, rgb_img, thermal_img)

    def read_image(self, line, channel):
        image_id = line.split("/")
        if channel == 'rgb':
            folder = 'visible'
        else:
            folder = 'lwir'
        im_path = os.path.join(self.images_path, image_id[0], image_id[1], folder, image_id[2]+".jpg")
        out = read_image(im_path)
        return out

    def get_normalization_params(self):
        mean_color = np.mean([torch.mean(value[1].float(), (1, 2)).numpy() for value in list(self.data_dict.values())], axis=0)
        std_color = np.mean([torch.std(value[1].float(), (1, 2)).numpy() for value in list(self.data_dict.values())], axis=0)
        mean_color = mean_color / 255
        std_color = std_color / 255
        mean_thermal = np.mean([torch.mean(value[2].float(), (1, 2)).numpy() for value in list(self.data_dict.values())], axis=0)
        std_thermal = np.mean([torch.std(value[2].float(), (1, 2)).numpy() for value in list(self.data_dict.values())], axis=0)
        mean_thermal = mean_thermal / 255
        std_thermal = std_thermal / 255
        return mean_color, std_color, mean_thermal, std_thermal

    def __len__(self):
        return len(self.data_dict.keys())

    def __getitem__(self, index: int):
        color = self.data_dict[index][1]
        thermal = self.data_dict[index][2]
        # boxes = torch.FloatTensor(self.boxes[frame])
        # labels = torch.LongTensor(self.labels[frame])
        boxes = None
        labels = None
        return color, thermal, boxes, labels

def dummy_func(address):
    global img_path
    global color_normalizer
    global thermal_normalizer
    # global dummy_dict
    image_id = address.split("/")
    rgb_path = os.path.join(img_path, image_id[0], image_id[1], 'visible', image_id[2] + ".jpg")
    rgb_img = read_image(rgb_path)
    thermal_path = os.path.join(img_path, image_id[0], image_id[1], 'lwir', image_id[2] + ".jpg")
    thermal_img = read_image(thermal_path)
    rgb_img = rgb_img / 255
    thermal_img = torch.mean(thermal_img.float(), 0)
    thermal_img = thermal_img.unsqueeze(0)
    thermal_img = thermal_img / 255
    rgb_img = color_normalizer(rgb_img)
    thermal_img = thermal_normalizer(thermal_img)
    # dummy_dict[address] = (rgb_img, thermal_img)
    return None

class ParallelKaist(Dataset):
    def __init__(self, data_path, set_name="train-all-20.txt", filter=None, normalize=True):
        global mem_name
        global img_path
        global color_normalizer
        global thermal_normalizer
        global dummy_dict
        self.data_path = data_path
        self.images_path = os.path.join(self.data_path, 'images')
        self.manager = Manager()
        # self.data_dict =
        # self.data_dict = UltraDict(name='shared_dict')
        # self.dummy_dict = UltraDict(recurse=True, shared_lock=True)
        self.dummy_dict = self.manager.dict()
        dummy_dict = self.dummy_dict
        self.set_path = os.path.join(self.data_path, 'imageSets', set_name)
        self.filter = filter
        self.normalize = normalize
        mean_color = torch.tensor([0.3464423, 0.32171702, 0.28396806])
        std_color = torch.tensor([0.23573601, 0.22641623, 0.22726215])
        mean_thermal = torch.tensor(0.16)
        std_thermal = torch.tensor(0.081)
        self.color_normalizer = Normalize(mean_color, std_color)
        self.thermal_normalizer = Normalize(mean_thermal, std_thermal)
        self.address_order = dict()
        with open(self.set_path, 'r') as ft:
            for idx,line in enumerate(ft):
                line = line.strip("\n")
                self.address_order[line] = idx
        # print(list(self.address_order.keys()))
        # for address in self.address_order.keys():
        #     self.get_files(address)
        color_normalizer = self.color_normalizer
        thermal_normalizer = self.thermal_normalizer
        # self.dummy_dict['order'] = self.address_order
        img_path = self.images_path
        # mem_name = self.dummy_dict.name

        # jobs = []
        # for i in range(5):
        #     p = Process(target=self.dummy, args=(i, self.dummy_dict))
        #     jobs.append(p)
        #     p.start()
        #
        # for proc in jobs:
        #     proc.join()
        p = Pool(12)
        self.returns = p.map(dummy_func, list(self.address_order.keys()), chunksize=100)
        p.close()
        p.join()
        # print(returns)
        # print(self.dummy_dict)

    def dummy(self, address):

        global img_path
        global color_normalizer
        global thermal_normalizer

        img_path = os.path.join('../kaist/kaist-cvpr15', 'images')
        image_id = address.split("/")
        rgb_path = os.path.join(img_path, image_id[0], image_id[1], 'visible', image_id[2] + ".jpg")
        rgb_img = read_image(rgb_path)
        thermal_path = os.path.join(img_path, image_id[0], image_id[1], 'lwir', image_id[2] + ".jpg")
        thermal_img = read_image(thermal_path)
        rgb_img = rgb_img / 255
        thermal_img = torch.mean(thermal_img.float(), 0)
        thermal_img = thermal_img.unsqueeze(0)
        thermal_img = thermal_img / 255
        rgb_img = color_normalizer(rgb_img)
        thermal_img = thermal_normalizer(thermal_img)
        return rgb_img
        # idx = address_order_dict[address]
        # shared_dict[0] = 'test'


    def get_files(self, file_address):
        rgb_img = self.read_image(file_address, channel='rgb')
        thermal_img = self.read_image(file_address, channel='thermal')
        if self.normalize:
            rgb_img = rgb_img / 255
            thermal_img = torch.mean(thermal_img.float(), 0)
            thermal_img = thermal_img.unsqueeze(0)
            thermal_img = thermal_img / 255
            rgb_img = self.color_normalizer(rgb_img)
            thermal_img = self.thermal_normalizer(thermal_img)
        idx = self.address_order[file_address]
        self.data_dict[idx] = (file_address, rgb_img, thermal_img)

    def read_image(self, line, channel):
        image_id = line.split("/")
        if channel == 'rgb':
            folder = 'visible'
        else:
            folder = 'lwir'
        im_path = os.path.join(self.images_path, image_id[0], image_id[1], folder, image_id[2]+".jpg")
        out = read_image(im_path)
        return out

    def get_normalization_params(self):
        mean_color = np.mean([torch.mean(value[1].float(), (1, 2)).numpy() for value in list(self.data_dict.values())], axis=0)
        std_color = np.mean([torch.std(value[1].float(), (1, 2)).numpy() for value in list(self.data_dict.values())], axis=0)
        mean_color = mean_color / 255
        std_color = std_color / 255
        mean_thermal = np.mean([torch.mean(value[2].float(), (1, 2)).numpy() for value in list(self.data_dict.values())], axis=0)
        std_thermal = np.mean([torch.std(value[2].float(), (1, 2)).numpy() for value in list(self.data_dict.values())], axis=0)
        mean_thermal = mean_thermal / 255
        std_thermal = std_thermal / 255
        return mean_color, std_color, mean_thermal, std_thermal

    def __len__(self):
        return len(self.data_dict.keys())

    def __getitem__(self, index: int):
        color = self.data_dict[index][1]
        thermal = self.data_dict[index][2]
        # boxes = torch.FloatTensor(self.boxes[frame])
        # labels = torch.LongTensor(self.labels[frame])
        boxes = None
        labels = None
        return color, thermal, boxes, labels


def get_data(input_path, mode):
    all_imgs = []

    classes_count = {}

    class_mapping = {}

    visualize = False

    # data_paths = input_path #[os.path.join(input_path,s) for s in ['VOC2007', 'VOC2012']]
    print('Parsing annotation files')

    data_path = os.path.join(input_path, 'train-all-01.txt')

    image_dict = dict()


    with open(data_path, 'r') as ft:
        for line in ft:
            line = line.strip("\n")



            annot = line.replace("images", "annotations-xml")
            annot = annot.replace(".jpg", ".xml")
            annot = annot.replace("/visible", "/")






            et = ET.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')
            element_filename = element.find('filename').text
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)

            # if len(element_objs) > 0:
            #     annotation_data = {'filepath': line, 'width': element_width,
            #                        'height': element_height, 'bboxes': []}
            #
            #     '''if element_filename in trainval_files:
            #         annotation_data['imageset'] = 'trainval'
            #     elif element_filename in test_files:
            #         annotation_data['imageset'] = 'test'
            #     else:
            #         annotation_data['imageset'] = 'trainval'''
            #
            #     for element_obj in element_objs:
            #         # class_name = element_obj.find('name').text
            #         class_name = "person"
            #         if class_name not in classes_count:
            #             classes_count[class_name] = 1
            #         else:
            #             classes_count[class_name] += 1
            #         class_mapping[class_name] = 0
            #         '''if class_name not in classes_count:
            #             classes_count[class_name] = 1
            #         else:
            #             classes_count[class_name] += 1
            #
            #         if class_name not in class_mapping:
            #             class_mapping[class_name] = len(class_mapping)'''
            #
            #         obj_bbox = element_obj.find('bndbox')
            #         x1 = int(round(float(obj_bbox.find('xmin').text)))
            #         y1 = int(round(float(obj_bbox.find('ymin').text)))
            #         x2 = int(round(float(obj_bbox.find('xmax').text)))
            #         y2 = int(round(float(obj_bbox.find('ymax').text)))
            #         difficulty = int(element_obj.find('difficult').text) == 1
            #         # print([x1,y1,x2,y2])
            #         annotation_data['bboxes'].append(
            #             {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
            #     all_imgs.append(annotation_data)
            # else:
            #     continue

    return all_imgs, classes_count, class_mapping


'''if __name__ == "__main__":
	input_path = "/home/kishan/Documents/Knowledge_distillation_ped_detection/github_clone/rgbt-ped-detection-master/data/kaist-rgbt/"
	imgs_dicts = get_data(input_path)
	print imgs_dicts[59]
	print len(imgs_dicts)'''


if __name__ == '__main__':
    input_path = "../kaist/kaist-cvpr15"
    imgs_dicts = get_data(input_path)
    len(imgs_dicts)