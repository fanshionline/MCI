# coding=utf-8
import argparse
import os
import time
from math import ceil

import cv2
import numpy as np
from cv2 import dnn

image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
strides = [8.0, 16.0, 32.0, 64.0]

onnx_path = "model/version-RFB-320_simplified.onnx"
caffe_prototxt_path = "model/RFB-320.prototxt"
caffe_model_path = "model/RFB-320.caffemodel"
threshold = 0.7
input_size = [320, 240]
#     net = dnn.readNetFromONNX(onnx_path)  # onnx version
net = dnn.readNetFromCaffe(caffe_prototxt_path, caffe_model_path)  # caffe model converted from onnx
width = input_size[0]
height = input_size[1]


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    return np.clip(priors, 0.0, 1.0)


def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors


priors = define_img_size(input_size)


# 距离计算函数
def distance_to_camera(knownWidth, focalLength, perWidth):
    """
    knownWidth：知道的目标宽度 厘米
    focalLength：摄像头焦距
    perWidth：检测框宽度  像素
    
    #读入第一张图，通过已知距离计算相机焦距
    image = cv2.imread(IMAGE_PATHS[0]) 
    marker = find_marker(image)           
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH 
    """
    return (knownWidth * focalLength) / perWidth


##马赛克
def do_mosaic(frame, x, y, w, h, neighbor=9):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param frame: opencv frame
    :param int x :  马赛克左顶点
    :param int y:  马赛克右顶点
    :param int w:  马赛克宽
    :param int h:  马赛克高
    :param int neighbor:  马赛克每一块的宽
    """
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neightbour 防止溢出
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
            cv2.rectangle(frame, left_up, right_down, color, -1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)


def inference():
    cap = cv2.VideoCapture(1)
    ret = cap.set(3, 3040)
    ret = cap.set(4, 1520)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img_ori = frame
            rect = cv2.resize(img_ori, (width, height))
            rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
            net.setInput(dnn.blobFromImage(rect, 1 / image_std, (width, height), 127))
            time_time = time.time()
            boxes, scores = net.forward(["boxes", "scores"])
            #             print("inference time: {} s".format(round(time.time() - time_time, 4)))
            boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
            scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
            boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
            boxes = center_form_to_corner_form(boxes)
            boxes, labels, probs = predict(img_ori.shape[1], img_ori.shape[0], scores, boxes, threshold)
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                cv2.rectangle(img_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.circle(img_ori, (box[0], box[1]), 1, (0, 0, 255), 4)
                cv2.circle(img_ori, (box[2], box[3]), 1, (0, 0, 255), 4)
                ya_max = box[1]
                yb_max = box[3]
                pix_person_height = yb_max - ya_max
                # focalLength = ((box[3]-box[1]) * 60) / 15
                # print(focalLength)
                distance = distance_to_camera(15, 2300, pix_person_height)

                if distance / 100 > 1.0:
                    cv2.putText(img_ori, "%.2fm" % (distance / 100),
                                (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                10, (255, 0, 255), 2)
                else:
                    cv2.putText(img_ori, "%.2fcm" % (distance),
                                (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                10, (255, 0, 255), 2)
                #                 人脸马赛克
                do_mosaic(img_ori, box[0] + 5, box[1] + 5, box[2] - box[0] - 5, box[3] - box[1] - 5, neighbor=16)
            img = cv2.resize(img_ori, (960, 540))
            cv2.imshow("Face_Distance_Mosaic", img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def face_detect(frame):
    img_ori = frame
    rect = cv2.resize(img_ori, (width, height))
    rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
    net.setInput(dnn.blobFromImage(rect, 1 / image_std, (width, height), 127))
    time_time = time.time()
    boxes, scores = net.forward(["boxes", "scores"])
    #             print("inference time: {} s".format(round(time.time() - time_time, 4)))
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
    boxes = center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(img_ori.shape[1], img_ori.shape[0], scores, boxes, threshold)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        # cv2.rectangle(img_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # # cv2.circle(img_ori, (box[0], box[1]), 1, (0, 0, 255), 4)
        # # cv2.circle(img_ori, (box[2], box[3]), 1, (0, 0, 255), 4)
        # ya_max = box[1]
        # yb_max = box[3]
        # pix_person_height = yb_max - ya_max
        # focalLength = ((box[3]-box[1]) * 60) / 15
        # print(focalLength)
        # distance = distance_to_camera(15, 375, pix_person_height)

        # if distance / 100 > 1.0:
        #     cv2.putText(img_ori, "%.2fm" % (distance / 100),
        #                 (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #                 10, (255, 0, 255), 2)
        # else:
        #     cv2.putText(img_ori, "%.2fcm" % (distance),
        #                 (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #                 10, (255, 0, 255), 2)
        #                 人脸马赛克
        do_mosaic(img_ori, box[0] + 5, box[1] + 5, box[2] - box[0] - 5, box[3] - box[1] - 5, neighbor=16)
    return boxes, img_ori


if __name__ == '__main__':
    cv2.namedWindow('Face_Distance_Mosaic')
    cv2.resizeWindow('Face_Distance_Mosaic', 960, 540)
    cv2.moveWindow('Face_Distance_Mosaic', 0, 0)
    inference()
