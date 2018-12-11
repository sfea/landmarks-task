import os
from matplotlib import pyplot as plt
import numpy as np
import re
import shutil
from tqdm import tqdm_notebook, tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.optim import Adam, SGD
from torch.nn.modules import MSELoss
import cv2
from skimage.color import gray2rgb, rgb2gray
import datetime
import sys
import dlib
import glob
from imageio import imread, imsave
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def extract_keypoints(path_to_pts):
    '''
    Function extracts point coordinates from .pts file and returns np.array of point pairs
    
    path_to_pts: string, path to .pts file
    
    return:
    points: np.array, contains pairs of keypoints
    '''
    
    # Getting rid of "\n" symbols
    with open(path_to_pts) as f:
        data = [x.strip() for x in f.readlines()]
        
    # File format assumes that the points are located strictly after '{' and strictly before '}'
    begin = data.index('{') + 1
    end = data.index('}')

    points_raw = data[begin:end]
    points_raw = [x.split(" ") for x in points_raw]

    points = [list([float(point) for point in point_raw]) for point_raw in points_raw]
    points = np.array(points)
    
    return points

def iterate_batches(X, y, batch_size=8, shuffle=False):
    '''Returns iterable batch'''
    
    assert len(X) == len(y), "Number of samples must be equal to number of labels!"
    
    num_samples = len(X)
    permutation = np.random.permutation(num_samples)
    for ndx in range(0, num_samples, batch_size):
        if shuffle:
            indices = permutation[ndx:min(ndx + batch_size, num_samples)]
        else:
            indices = np.arange(ndx, min(ndx + batch_size, num_samples), dtype='int')
        yield np.array(X)[indices], np.array(y)[indices]
            
class Flatten(nn.Module):
    '''Flatten layer'''
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.conv1 = nn.Conv2d(1,   32,  3)
        self.conv2 = nn.Conv2d(32,  64,  3)
        self.conv3 = nn.Conv2d(64,  64, 3)
        self.conv4 = nn.Conv2d(64, 128, 2)
        
        self.mp1 = nn.MaxPool2d(3)
        self.mp2 = nn.MaxPool2d(3)
        self.mp3 = nn.MaxPool2d(3)
        self.mp4 = nn.MaxPool2d(2)
        
        self.flatten = Flatten()
        
        self.fc = nn.Linear(1152, 68 * 2)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.mp1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.mp2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.mp3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = self.mp4(x)
        x = F.relu(x)
        
        x = self.flatten(x)
        
        x = self.fc(x)
        
        return x
    
def img2gray(img):
    '''
    Transforms img to grayscale one with only one color channel
    '''
    if len(img.shape) == 3:
        return rgb2gray(img)
    if len(img.shape) == 2:
        return img / 255
    
def visualize_results(X, y, output, axes, n_imgs):
    '''
    Drawing the results
    '''
    for i in range(n_imgs_to_evaluate):
        axes[i].imshow(X[i][0].numpy(), cmap='gray')
        axes[i].plot(y[i][:, 0], y[i][:, 1], '.')
        axes[i].plot(np.array(output.cpu().detach())[i][::2], np.array(output.cpu().detach())[0][1::2], '.')
        axes[i].legend(["Ground truth", "Predicted"])
    plt.show()
    
# ===============================================================================================================================# 
IMG_PATH = "landmarks_task/Faces/"
ATTR_PATH = "landmarks_task/attributes/"
CROPPED_IMG_PATH = "landmarks_task/Faces_cropped/"

# Раскомментируйте для загрузки данных из файлов
with open("landmarks_task/Faces_cropped/shifts_train.csv", 'r') as f:
    shifts_train = json.load(f)
    
with open("landmarks_task/Faces_cropped/shifts_test.csv", 'r') as f:
    shifts_test = json.load(f)

train_img_paths = [file for file in os.listdir(IMG_PATH + "train/") if re.findall(".jpg", file)]
train_pts = [extract_keypoints(ATTR_PATH + "train/" + file[:-3] + "pts") for file in train_img_paths]

test_img_paths = [file for file in os.listdir(IMG_PATH + "test/") if re.findall(".jpg", file)]
test_pts = [extract_keypoints(ATTR_PATH + "test/" + file[:-3] + "pts") for file in test_img_paths]

train_loss_epoch = []
test_loss_epoch = []

BATCH_SIZE = 256
N_EPOCHS = 50
H = 224
W = 224

model = ONet()
model.cuda()
opt = Adam(model.parameters(), lr=1e-3)
mseloss = MSELoss()

for epoch in tqdm_notebook(range(N_EPOCHS)):

    # Обучение
    if (epoch + 1) % 5 == 0:
        for g in opt.param_groups:
            g['lr'] /= 2
            print("lr decreased twice")

    model.train()
    train_loss = []
    print("Starting epoch #%i!" % epoch)
    for n_batch, (X_batch, y_batch) in tqdm(enumerate(iterate_batches(train_img_paths, train_pts, 
                                                                               BATCH_SIZE, shuffle=True))):
        X_batch_images = []
        y_batch_transformed = np.copy(y_batch)
        for k, img_name in enumerate(X_batch):
            img = imread(CROPPED_IMG_PATH + "train/" + img_name)

            # Трансформируем полученное изображение в одноканальное черно-белое
            img = img2gray(img)

            # Сдвигаем координаты ключевых точек в зависимости от координат прямоугольника
            y_batch_transformed[k] -= shifts_train[CROPPED_IMG_PATH + "train/" + img_name][::-1]

            # Приводим все вырезанные изодражения к одному размеру и соответственно изменяем координаты точек
            X_batch_images.append(torch.Tensor(cv2.resize(img, (H, W))).unsqueeze(0))
            y_scale = img.shape[0] / H  
            x_scale = img.shape[1] / W 
            y_batch_transformed[k] /= [x_scale, y_scale]

        X_batch_images = torch.stack(X_batch_images)
        output = model(X_batch_images.cuda())

        loss = mseloss(output, torch.Tensor(list(map(lambda x: x.flatten(), y_batch_transformed))).cuda())

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss)

    train_loss_epoch.append((sum(train_loss) / len(train_loss)).item())
    print("Epoch #%i: train_loss = %.2f" % (epoch, train_loss_epoch[-1]))

    # Тестирование
    model.train(False)
    test_loss = []
    for n_batch_test, (X_batch_test, y_batch_test) in tqdm(enumerate(iterate_batches(test_img_paths, test_pts, 
                                                                                              BATCH_SIZE))):
        with torch.no_grad():
            X_batch_test_images = []
            y_batch_test_transformed = np.copy(y_batch_test)
            for k, img_name in enumerate(X_batch_test):
                img = imread(CROPPED_IMG_PATH + "test/" + img_name)

                if len(img.shape) == 3:
                    img = rgb2gray(img)
                if len(img.shape) == 2:
                    img = img / 255

                y_batch_test_transformed[k] -= shifts_test[CROPPED_IMG_PATH + "test/" + img_name][::-1]

                X_batch_test_images.append(torch.Tensor(cv2.resize(img, (H, W))).unsqueeze(0))
                y_scale = img.shape[0] / H
                x_scale = img.shape[1] / W
                y_batch_test_transformed[k] /= [img.shape[0] / H, img.shape[1] / W]

            X_batch_test_images = torch.stack(X_batch_test_images)
            output = model(X_batch_test_images.cuda())

            loss = mseloss(output, torch.Tensor(list(map(lambda x: x.flatten(), y_batch_test_transformed))).cuda())
            test_loss.append(loss)

    test_loss_epoch.append((sum(test_loss) / len(test_loss)).item())
    print("Epoch #%i: test_loss = %.2f" % (epoch, test_loss_epoch[-1]))
    
    # Сохранение чекпойнтов
    now = datetime.datetime.now()
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'train_loss': train_loss_epoch[-1],
            'test_loss': test_loss_epoch[-1]
            }, "landmarks_task/checkpoints/" + now.strftime("%Y-%m-%d_%H:%M") + "_" + str(epoch) + "ep.pth")
    
    print("============================================================")