import torch
from torch.autograd import Variable
import math
from PIL import Image
import numpy as np
from .box_utils import nms, _preprocess
import cv2
from cuda import cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

def run_first_stage(image, net, scale, threshold):
    """Run P-Net, generate bounding boxes, and do NMS.

    Arguments:
        image: an instance of PIL.Image.
        net: an instance of pytorch's nn.Module, P-Net.
        scale: a float number,
            scale width and height of the image by this number.
        threshold: a float number,
            threshold on the probability of a face when generating
            bounding boxes from predictions of the net.

    Returns:
        a float numpy array of shape [n_boxes, 9],
            bounding boxes with scores and offsets (4 + 1 + 4).
    """

    # scale the image and convert it to a float array
    width, height = image.shape[1],image.shape[0]
    sw, sh = math.ceil(width*scale), math.ceil(height*scale)
    if not image.is_contiguous():
        image = image.contiguous()
    gpuMat_cv = cv2.cuda.createGpuMatFromCudaMemory(height
                ,width,cv2.CV_8UC3,image.data_ptr())
    img = cv2.cuda.resize(gpuMat_cv,(sw, sh), interpolation=cv2.INTER_LINEAR)
    
    # 将图片拷贝到tensor
    w,h = img.size()
    gpuTensor = torch.zeros((h,w,3),device=device,dtype=torch.uint8)
    cpy = cuda.CUDA_MEMCPY2D()
    cpy.WidthInBytes = w*3
    cpy.Height = h
    cpy.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
    cpy.srcDevice = img.cudaPtr()
    cpy.srcXInBytes = 0
    cpy.srcY = 0
    cpy.srcPitch = img.step
    cpy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
    cpy.dstDevice = gpuTensor.data_ptr()
    cpy.dstXInBytes = 0
    cpy.dstY = 0
    ret = cuda.cuMemcpy2D(cpy)

    if not gpuTensor.is_contiguous():
            gpuTensor = gpuTensor.contiguous()

    img = gpuTensor.to(torch.float32)
    img = _preprocess(img)
    with torch.no_grad():
        output = net(img)
        probs = output[1][0, 1, :, :]
        offsets = output[0]
        # probs: probability of a face at each sliding window
        # offsets: transformations to true bounding boxes

        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)

    return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.

    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.

    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = torch.where(probs > threshold)
    if inds[0].shape[0] == 0:
        return np.array([])

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = torch.vstack([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]
    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = torch.vstack([
        torch.round((stride*inds[1] + 1.0)/scale),
        torch.round((stride*inds[0] + 1.0)/scale),
        torch.round((stride*inds[1] + 1.0 + cell_size)/scale),
        torch.round((stride*inds[0] + 1.0 + cell_size)/scale),
        score, offsets[0],offsets[1],offsets[2],offsets[3]
    ])
    # why one is added?
    

    return bounding_boxes.T
