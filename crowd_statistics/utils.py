import cv2
import math
import numpy as np



classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# -------------------------------------------图像预处理-------------------------------------------
def resize_image(srcimg, keep_ratio=True):
        input_shape = (416,416)
        top, left, newh, neww = 0, 0, input_shape[0],input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
                hw_scale = srcimg.shape[0] / srcimg.shape[1]
                if hw_scale > 1:
                        newh, neww = input_shape[0], int(input_shape[1] / hw_scale)
                        img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                        left = int((input_shape[1] - neww) * 0.5)
                        img = cv2.copyMakeBorder(img, 0, 0, left, input_shape[1] - neww - left,
                                                 cv2.BORDER_CONSTANT,
                                                 value=0)  # add border
                else:
                        newh, neww = int(input_shape[0] * hw_scale), input_shape[1]
                        img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                        top = int((input_shape[0] - newh) * 0.5)
                        img = cv2.copyMakeBorder(img, top, input_shape[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                                 value=0)
        else:
                img = cv2.resize(srcimg, input_shape, interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

def pre_process(srcimg):
        img, newh, neww, top, left = resize_image(srcimg, keep_ratio=False)

        mean = [103.53, 116.28, 123.675]
        img = img - mean

        norm_vals = [0.017429, 0.017507, 0.017125]
        img = img * norm_vals

        img = img.transpose((2, 0, 1))

        img = img[np.newaxis, :].astype(np.float32)

        return img, newh, neww, top, left
# -------------------------------------------图像预处理-------------------------------------------

# -------------------------------------------图像后处理-------------------------------------------
def sigmoid(z):
        return 1 / (1 + np.exp(-z))


def softmax(x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s


def _make_grid(featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        return np.stack((xv, yv), axis=-1)


def distance2bbox(points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
                x1 = np.clip(x1, 0, max_shape[1])
                y1 = np.clip(y1, 0, max_shape[0])
                x2 = np.clip(x2, 0, max_shape[1])
                y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)


def post_process(preds, scale_factor=1, rescale=False):
        prob_threshold = 0.4
        iou_threshold = 0.3
        reg_max = 7
        input_shape = (416, 416)
        project = np.arange(reg_max + 1)
        strides = [8, 16, 32, 64]
        num_classes = 80
        mlvl_anchors = []
        for i in range(len(strides)):
                anchors = _make_grid(
                        (math.ceil(input_shape[0] / strides[i]), math.ceil(input_shape[1] / strides[i])),
                        strides[i])
                mlvl_anchors.append(anchors)

        mlvl_bboxes = []
        mlvl_scores = []
        ind = 0
        for stride, anchors in zip(strides, mlvl_anchors):
                cls_score, bbox_pred = preds[ind:(ind + anchors.shape[0]), :num_classes], preds[ind:(ind + anchors.shape[0]),num_classes:]
                cls_score = 1 / (1 + np.exp(-cls_score))

                ind += anchors.shape[0]
                bbox_pred = softmax(bbox_pred.reshape(-1, reg_max + 1), axis=1)
                # bbox_pred = np.sum(bbox_pred * np.expand_dims(project, axis=0), axis=1).reshape((-1, 4))
                bbox_pred = np.dot(bbox_pred, project).reshape(-1, 4)
                bbox_pred *= stride

                # nms_pre = cfg.get('nms_pre', -1)
                nms_pre = 1000
                if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                        max_scores = cls_score.max(axis=1)
                        topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                        anchors = anchors[topk_inds, :]
                        bbox_pred = bbox_pred[topk_inds, :]
                        cls_score = cls_score[topk_inds, :]

                bboxes = distance2bbox(anchors, bbox_pred, max_shape=input_shape)
                mlvl_bboxes.append(bboxes)
                mlvl_scores.append(cls_score)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
                mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), prob_threshold,
                                   iou_threshold).flatten()
        if len(indices) > 0:
                mlvl_bboxes = mlvl_bboxes[indices]
                confidences = confidences[indices]
                classIds = classIds[indices]
                return mlvl_bboxes, confidences, classIds
        else:
                print('nothing detect')
                return np.array([]), np.array([]), np.array([])


# results = []
def img_draw(srcimg, det_bboxes, det_conf, det_classid,newh, neww, top, left):
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        for i in range(det_bboxes.shape[0]):
                xmin, ymin, xmax, ymax = max(int((det_bboxes[i, 0] - left) * ratiow), 0), max(
                        int((det_bboxes[i, 1] - top) * ratioh), 0), min(
                        int((det_bboxes[i, 2] - left) * ratiow), srcimg.shape[1]), min(
                        int((det_bboxes[i, 3] - top) * ratioh),
                        srcimg.shape[0])
                # results.append((xmin, ymin, xmax, ymax, classes[det_classid[i]], det_conf[i]))
                cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
                print(classes[det_classid[i]] + ': ' + str(round(det_conf[i], 3)))
                cv2.putText(srcimg, classes[det_classid[i]] + ': ' + str(round(det_conf[i], 3)), (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        #         cv2.imwrite('result.jpg', srcimg)
        return srcimg

# -------------------------------------------图像后处理-------------------------------------------

