# coding=utf-8
import torch
import torchvision
import numpy as np
import torch.nn as nn
import cv2
import os
import time
import copy
from PIL import Image, ImageEnhance
import nets.hourglass as models
from configs.draw_in_image import *
from nets.hourglass import Bottleneck
from utils.evaluation import final_preds
from configs.remove_small import *

class Fish(object):
    def __init__(self, left_image=None, up_image=None):
        self.t = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.manual_seed(1)
        self.device = device
        self.left_image = left_image
        self.up_image = up_image
        self.left_checkpoint = '../checkpoints/512x428_left/model_best.pth.tar'
        self.left_checkpoint_0 = '../checkpoints/512x256_left/model_best.pth.tar'
        self.left_checkpoint_1 = '../checkpoints/512x512_left/model_best.pth.tar'
        # self.up_checkpoint = './checkpoints/512x428_up/model_best.pth.tar'
        self.up_checkpoint = '../checkpoints/512x256_up/model_best.pth.tar'
        self.left_model = self.load_model(sign="left")
        self.up_model = self.load_model(sign="up")
        self.left_biao = 1/94.2  # 94.2
        self.up_biao = 1/70.4  # 66.934

    def get_image_data(self, left_image, up_image):
        self.left_image = left_image
        self.up_image = up_image

    def load_model(self, sign):
        model = models.HourglassNet(Bottleneck, num_stacks=2, num_blocks=4, num_classes=10 if sign == "left" else 9)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(self.left_checkpoint_1 if sign=="left" else self.up_checkpoint, map_location="cuda:0")
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        return model

    def img_preprocess(self, src_image, model, draw):
        img = copy.deepcopy(src_image)
        img = Image.fromarray(img)
        # img = img.resize((512, 428))
        img = img.resize((512, 512))
        img = self.im_to_torch(img)
        input = img.to(self.device, non_blocking=True)
        output = model(input)
        score_map = output[-1].cpu() if type(output) == list else output.cpu()
        left_preds = final_preds(score_map, res=[128, 128])
        points = []
        for i in left_preds[0]:
            tmp = [int((i.tolist()[0] - 1) * 19.125), int((i.tolist()[1] - 1) * 16)]
            points.append(tuple(tmp))
        src_image, result = draw(src_image, points)
        return src_image, result

    # 处理两张图片得到的体型数据，得到最终的体型数据
    def process_data(self, left_data, up_data):
        fish = {}
        fish["full_length"] = 0.0
        fish["body_length"] = 0.0
        fish["tail_handle_height"] = 0.0
        fish["tail_handle_length"] = 0.0
        fish["height"] = 0.0
        fish["head_length"] = 0.0
        fish["body_thickness"] = 0.0

        fish["height"] = round(distance_with_point(left_data["head_up"], left_data["head_down"]) * self.left_biao, 3)

        # up
        fish["body_thickness"] = round(distance_with_point(up_data["fin_left"], up_data["fin_right"]) * self.up_biao, 3)

        v_fin_middle_to_head = [abs(up_data["head"][0] - up_data["fin_middle"][0]),
                                   abs(up_data["head"][1] - up_data["fin_middle"][1])]

        v_middle_middle_to_fin_middle = [abs(up_data["fin_middle"][0] - up_data["middle_middle"][0]),
                                   abs(up_data["fin_middle"][1] - up_data["middle_middle"][1])]

        v_middle_to_middle_middle = [abs(up_data["middle_middle"][0] - up_data["middle"][0]),
                                   abs(up_data["middle_middle"][1] - up_data["middle"][1])]

        v_small_tail_to_middle = [abs(up_data["middle"][0] - up_data["small_tail"][0]),
                                   abs(up_data["middle"][1] - up_data["small_tail"][1])]

        v_tail_to_small_tail = [abs(up_data["small_tail"][0] - up_data["tail"][0]),
                                abs(up_data["small_tail"][1] - up_data["tail"][1])]

        up_dict = {}
        up_dict["v_fin_middle_to_head"] = [v_fin_middle_to_head[1] / self.distance_vec(v_fin_middle_to_head)]
        up_dict["v_middle_middle_to_fin_middle"] = [v_middle_middle_to_fin_middle[1] / self.distance_vec(v_middle_middle_to_fin_middle),
                                                    v_middle_middle_to_fin_middle[1] / (v_middle_middle_to_fin_middle[1] + v_middle_to_middle_middle[1] + v_small_tail_to_middle[1])]
        up_dict["v_middle_to_middle_middle"] = [v_middle_to_middle_middle[1] / self.distance_vec(v_middle_to_middle_middle),
                                                v_middle_to_middle_middle[1]/(v_middle_middle_to_fin_middle[1] + v_middle_to_middle_middle[1] + v_small_tail_to_middle[1])]
        up_dict["v_small_tail_to_middle"] = [v_small_tail_to_middle[1] / self.distance_vec(v_small_tail_to_middle),
                                             v_small_tail_to_middle[1]/(v_middle_middle_to_fin_middle[1] + v_middle_to_middle_middle[1] + v_small_tail_to_middle[1])]
        up_dict["v_tail_to_small_tail"] = [v_tail_to_small_tail[1] / self.distance_vec(v_tail_to_small_tail)]


        # left
        v_small_middle_to_head_middle = [left_data["head_middle"][0] - left_data["small_middle"][0],
                                         left_data["head_middle"][1] - left_data["small_middle"][1]]

        v_head_middle_to_head = [left_data["head"][0] - left_data["head_middle"][0],
                                 left_data["head"][1] - left_data["head_middle"][1]]

        v_tail_to_small_middle = [left_data["small_middle"][0] - left_data["tail"][0],
                                  left_data["small_middle"][1] - left_data["tail"][1]]

        v_tail_up_to_small_middle = [left_data["small_middle"][0] - left_data["tail_up"][0],
                                     left_data["small_middle"][1] - left_data["tail_up"][1]]

        v_tail_down_to_small_middle = [left_data["small_middle"][0] - left_data["tail_down"][0],
                                       left_data["small_middle"][1] - left_data["tail_down"][1]]

        v_tail_up_to_tail = [left_data["tail"][0] - left_data["tail_up"][0],
                                     left_data["tail"][1] - left_data["tail_up"][1]]

        v_tail_down_to_tail = [left_data["tail"][0] - left_data["tail_down"][0],
                                       left_data["tail"][1] - left_data["tail_down"][1]]

        v_small_middle_to_front_small = [left_data["front_small"][0] - left_data["small_middle"][0],
                                         left_data["front_small"][1] - left_data["small_middle"][1]]

        left_head_length = distance_with_point(left_data["head"], left_data["head_fin"]) * self.left_biao / up_dict["v_fin_middle_to_head"][
            0]

        tmp = (self.distance_vec(v_small_middle_to_head_middle) + \
               self.vec_project(v_tail_to_small_middle, v_small_middle_to_head_middle)) * self.left_biao

        left_body_length = distance_with_point(left_data["head"], left_data["head_middle"]) * self.left_biao + \
                           tmp *(up_dict["v_middle_middle_to_fin_middle"][1]) / up_dict["v_middle_middle_to_fin_middle"][0] + \
                           tmp *(up_dict["v_middle_to_middle_middle"][1]) / up_dict["v_middle_to_middle_middle"][0] + \
                           tmp *(up_dict["v_small_tail_to_middle"][1]) / up_dict["v_small_tail_to_middle"][0]

        left_full_length = distance_with_point(left_data["head"], left_data["head_middle"]) * self.left_biao + \
                           tmp *(up_dict["v_middle_middle_to_fin_middle"][1]) / up_dict["v_middle_middle_to_fin_middle"][0] + \
                           tmp *(up_dict["v_middle_to_middle_middle"][1]) / up_dict["v_middle_to_middle_middle"][0] + \
                           tmp *(up_dict["v_small_tail_to_middle"][1]) / up_dict["v_small_tail_to_middle"][0] + \
                           min(self.vec_project(v_tail_up_to_tail, v_small_middle_to_head_middle),
                               self.vec_project(v_tail_down_to_tail, v_small_middle_to_head_middle)) * self.left_biao / \
                           up_dict["v_tail_to_small_tail"][0]

        left_tail_handle_length = (self.vec_project(v_small_middle_to_front_small, v_small_middle_to_head_middle) + \
                                  self.vec_project(v_tail_to_small_middle, v_small_middle_to_head_middle)) * self.left_biao
                                  # up_dict["v_small_tail_to_middle"][0]

        # -----
        fish["tail_handle_height"] = round(distance_with_point(left_data["tail_up"], left_data["tail_down"]) * self.left_biao*0.55, 3)
        # v_tail_up_tail_down = [left_data["tail_down"][0] - left_data["tail_up"][0],
        #                                  left_data["tail_down"][1] - left_data["tail_up"][1]]
        # vertical_line_to_v_small_middle_to_head_middle = [left_data["small_middle"][0], -v_small_middle_to_head_middle[0]*left_data["small_middle"][0]/v_small_middle_to_head_middle[1]]
        # fish["tail_handle_height"] = round(self.vec_project(v_tail_up_tail_down, vertical_line_to_v_small_middle_to_head_middle)* self.left_biao*0.65, 3)
        # -----

        fish["full_length"] = round(left_full_length, 3)
        fish["body_length"] = round(left_body_length, 3)
        fish["tail_handle_length"] = round(left_tail_handle_length, 3)
        fish["head_length"] = round(left_head_length, 3)
        return fish

    def im_to_torch(self, img):
        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = img[np.newaxis, :]
        img = self.to_torch(img).float()
        if img.max() > 1:
            img /= 255
        return img

    def to_torch(self, ndarray):
        if type(ndarray).__module__ == 'numpy':
            return torch.from_numpy(ndarray)
        elif not torch.is_tensor(ndarray):
            raise ValueError("Cannot convert {} to torch tensor"
                             .format(type(ndarray)))
        return ndarray

    def distance_with_point(self, point1, point0):
        dist = ((point1[1] - point0[1]) ** 2 + (point1[0] - point0[0]) ** 2) ** 0.5
        return dist

    def distance_vec(self, vec):
        return (vec[0] ** 2 + vec[1] ** 2) ** 0.5

    def vec_project(self, pt1, pt2):
        "pt1 length project on pt2"
        return abs((pt1[0] * pt2[0] + pt1[1] * pt2[1]) / self.distance_vec(pt2))

    def img_up_preprocess(self, src_image, model, draw):
        img = copy.deepcopy(src_image)
        img = img[:, 968:1480, :]
        # img = img[200:1848, 978:1458, :]
        # cv2.namedWindow("img", 0)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = cv2.resize(img, (256, 512))
        img = Image.fromarray(img)

        if np.random.random() > 0.5:
            # sharpness
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(factor)
        if np.random.random() > 0.5:
            # color
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)
        if np.random.random() > 0.5:
            # brightness
            factor = 0.5 + np.random.random()
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        img = self.im_to_torch(img)
        input = img.to(self.device, non_blocking=True)
        output = model(input)
        score_map = output[-1].cpu() if type(output) == list else output.cpu()
        left_preds = final_preds(score_map, res=[64, 128])
        points = []
        for i in left_preds[0]:
            tmp = [int((i.tolist()[0] - 1) * 8) + 968, int((i.tolist()[1]-1) * 16)]
            points.append(tuple(tmp))
        src_image, result = draw(src_image, points)
        return src_image, result


    def get_data(self):
        left = copy.deepcopy(self.left_image)
        up = copy.deepcopy(self.up_image)
        left_image, left_data = self.img_preprocess(self.left_image, self.left_model, draw_left)

        # Fish.get_tail_process(left, left_data)
        up_image, up_data = self.img_up_preprocess(self.up_image, self.up_model, draw_up)
        # up_image, up_data = self.img_preprocess(self.up_image, self.up_model, draw_up)
        result = self.process_data(left_data, up_data)
        # result = {}
        # result["full_length"] = 0.0
        # result["body_length"] = 0.0
        # result["tail_handle_height"] = 0.0
        # result["tail_handle_length"] = 0.0
        # result["height"] = 0.0
        # result["head_length"] = 0.0
        # result["body_thickness"] = 0.02
        cv2.imwrite("save/{}_up.bmp".format(self.t), up_image)
        cv2.imwrite("save/{}_left.bmp".format(self.t), left_image)
        self.t += 1
        return left_image, up_image, result

    @staticmethod
    def showImage(image, name):
        cv2.namedWindow(name, 0)
        cv2.imshow(name, image)

    @staticmethod
    def get_tail_process(image, data):

        up = data["small_up"]
        down = data["small_down"]
        roi_width = 120
        roi_height = 60
        roi_left_top = [up[0] - roi_width//2, up[1] - roi_height//2]
        roi_right_down = [down[0] + roi_width//2, down[1] + roi_height//2]
        roi = image[roi_left_top[1]:roi_right_down[1], roi_left_top[0]:roi_right_down[0], :]

        roi_blur = cv2.GaussianBlur(roi, (3, 3), 9)
        roi_gray = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2GRAY)
        ret, roi_er = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # roi_data = remove_small(roi_er, 100, 1, 1)


        roi_h, roi_w = roi_er.shape

        data = {}
        for i in range(roi_w):
            up_j, down_j = 0, 0
            for j in range(roi_h):
                if roi_er[j][i] == 255:
                    up_j = j
                    break
            for j in range(roi_h-1, -1, -1):
                if roi_er[j][i] == 255:
                    down_j = j
                    break
            data[i] = [down_j - up_j, up_j, down_j, i]
        data = {i:data[i] for i in data.keys() if data[i][0] != 0}
        min_l = []
        tmp = float("inf")
        for d in data.values():
            if d[0] < tmp:
                tmp = d[0]
                min_l = d

        new_up = [min_l[-1], min_l[1]]
        new_down = [min_l[-1], min_l[2]]

        # new_up = [new_up[0] + up[0], new_up[1] + up[1]]
        # new_down = [new_down[0] + up[0], new_down[1] + up[1]]

        draw_point(roi, int_point_tuple(new_up))
        draw_point(roi, int_point_tuple(new_down))
        Fish.showImage(roi, "roi")
        Fish.showImage(roi_er, "roi_er")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_point(image, point0):
    cv2.circle(image, point0, 2, (255, 0, 0), 2)

def int_point_tuple(point):
    return (int(point[0]), int(point[1]))

if __name__ == "__main__":
    from glob import glob
    left_path = "../image/"
    left_images =glob(os.path.join(left_path, "*_side.bmp"))
    up_images = glob(os.path.join(left_path, "*_top.bmp"))
    up_path = "../image/"

    for i in range(100):
        # left
        left_name = left_images[i]
        up_name = up_images[i]    # left_images[i].replace("side", "top")
        src_left_image = cv2.imread(left_name)
        src_up_image = cv2.imread(up_name)
        start = time.time()
        a = Fish(src_left_image, src_up_image)
        left, up, data = a.get_data()
        print(data)
        end = time.time()
        cv2.namedWindow(left_name + "cost:{}".format(end - start), 0)
        cv2.imshow(left_name + "cost:{}".format(end - start), left)

        cv2.namedWindow(up_name + "cost:{}".format(end - start), 0)
        cv2.imshow(up_name + "cost:{}".format(end - start), up)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
