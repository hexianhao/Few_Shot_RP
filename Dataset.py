import os
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms

class DataLoader(object):

    def __init__(self, file_path, training_labels=64, test_labels=16, img_size=224, img_num_per_class=600):

        self.file_path = file_path
        self.traing_labels = training_labels
        self.test_labels = test_labels
        self.img_size = img_size
        self.img_num_per_class = img_num_per_class
        self.mean = [0.92206, 0.92206, 0.92206]
        self.std = [0.08426, 0.08426, 0.08426]
        self.transform = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        ])
        self.data = self.load_images(file_path)


    def load_images(self, file_path):

        image_name = sorted(os.listdir(file_path))
        data = dict()
        training_data = []
        test_data = []; test_labels =[]

        for i in range(self.traing_labels):

            start = i * self.img_num_per_class
            end = start + self.img_num_per_class

            for j in range(start, end):
                training_data.append(image_name[j])

        for i in range(self.traing_labels, self.traing_labels + self.test_labels):

            start = i * self.img_num_per_class
            end = start + self.img_num_per_class

            for j in range(start, end):
                test_data.append(image_name[j])
                test_labels.append(i)

        training_data = np.asarray(training_data).reshape(self.traing_labels, self.img_num_per_class)
        test_data = np.asarray(test_data).reshape(self.test_labels, self.img_num_per_class)
        test_labels = np.asarray(test_labels).reshape(self.test_labels, self.img_num_per_class)

        data['train'] = training_data
        data['test'] = (test_data, test_labels)

        return data

    def train_batch(self, n_category):

        data = self.data['train']
        category_idx = np.random.permutation(range(self.traing_labels))[:n_category]
        Images = []; Labels = []

        for category in category_idx:

            '''
            for each category, we firstly get 
            n_samples images in n_category class at random
            '''
            indices = np.random.permutation(range(self.img_num_per_class))[:2]

            for idx in indices:
                img = Image.open(self.file_path + '/' + data[category, idx])
                img = self.transform(img)
                Images.append(img.numpy())
                Labels.append(category)

        Images = np.asarray(Images)
        Labels = np.asarray(Labels)

        return Images, Labels
    '''
    def train_batch(self, n_way=5, k_shot=1, test_size=16):
        data = self.data['train']
        category_idx = np.random.permutation(range(self.traing_labels))[:n_way]
        query_data = []
        query_labels = []
        test_data = []
        test_labels = []

        for category in category_idx:
            indices = np.random.permutation(range(self.img_num_per_class))[:k_shot + test_size]
            for idx in indices[:k_shot]:
                img = Image.open(self.file_path + '/' + data[category, idx])
                img = self.transform(img)
                query_data.append(img.numpy())
                query_labels.append(category)
            for idx in indices[k_shot:]:
                img = Image.open(self.file_path + '/' + data[category, idx])
                img = self.transform(img)
                test_data.append(img.numpy())
                test_labels.append(category)
        
        query_data = np.asarray(query_data)
        query_labels = np.asarray(query_labels)
        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

        return query_data, query_labels, test_data, test_labels
    '''
    def test_batch(self, n_way, k_shot, test_size = 16):

        data, labels = self.data['test']
        query_data = []
        query_labels = []
        test_data = []
        test_labels = []
        cls_idx = np.random.permutation(range(self.test_labels))[: n_way]

        for i in cls_idx:
            images = data[i]
            for j in range(k_shot):
                img = Image.open(self.file_path + '/' + images[j])
                img = self.transform(img)
                query_data.append(img.numpy())
                query_labels.append(labels[i, j])

            img_idx = np.random.permutation(range(k_shot, self.img_num_per_class))[: test_size]

            for j in img_idx:
                img = Image.open(self.file_path + '/' + images[j])
                img = self.transform(img)
                test_data.append(img.numpy())
                test_labels.append(labels[i, j])

        query_data = np.asarray(query_data).reshape([-1, 3, self.img_size, self.img_size])
        query_labels = np.asarray(query_labels)
        test_data = np.asarray(test_data).reshape([-1, 3, self.img_size, self.img_size])
        test_labels = np.asarray(test_labels)

        return query_data, query_labels, test_data, test_labels