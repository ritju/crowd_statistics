# import argparse
# import os
# import time

# import cv2
# import torch

# from crowd_statistics.nanodet.data.batch_process import stack_batch_img
# from crowd_statistics.nanodet.data.collate import naive_collate
# from crowd_statistics.nanodet.data.transform import Pipeline
# from crowd_statistics.nanodet.model.arch import build_model
# from crowd_statistics.nanodet.util import Logger, cfg, load_config, load_model_weight



# class Predictor(object):
#     def __init__(self,):
#         parser = argparse.ArgumentParser()
#         parser.add_argument("--config", default=str(os.path.join(os.path.dirname(__file__), "model/nanodet-plus-m_416_person_face_817.yml")),
#                         help="model config file path")
#         parser.add_argument("--model", default=str(os.path.join(os.path.dirname(__file__), "model/nanodet-plus-m_416_person_face_817.pth")),
#                         help="model file path")
#         parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
#         self.args = parser.parse_args(args=[])

#         local_rank = 0
#         torch.backends.cudnn.enabled = True
#         torch.backends.cudnn.benchmark = True

#         load_config(cfg, self.args.config)
#         self.logger = Logger(local_rank, use_tensorboard=False)
#         self.device="cuda:0"

#         self.cfg = cfg
#         model = build_model(cfg.model)
#         ckpt = torch.load(self.args.model, map_location=lambda storage, loc: storage)
#         load_model_weight(model, ckpt, self.logger)
#         if cfg.model.arch.backbone.name == "RepVGG":
#             deploy_config = cfg.model
#             deploy_config.arch.backbone.update({"deploy": True})
#             deploy_model = build_model(deploy_config)
#             from nanodet.model.backbone.repvgg import repvgg_det_model_convert

#             model = repvgg_det_model_convert(model, deploy_model)
#         self.model = model.to(self.device).eval()
#         self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

#     def inference(self, img):
#         img_info = {"id": 0}
#         if isinstance(img, str):
#             img_info["file_name"] = os.path.basename(img)
#             img = cv2.imread(img)
#         else:
#             img_info["file_name"] = None

#         height, width = img.shape[:2]
#         img_info["height"] = height
#         img_info["width"] = width
#         meta = dict(img_info=img_info, raw_img=img, img=img)
#         meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
#         meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
#         meta = naive_collate([meta])
#         meta["img"] = stack_batch_img(meta["img"], divisible=32)
#         with torch.no_grad():
#             results = self.model.inference(meta)
#         return meta, results

#     def visualize(self, dets, meta, class_names, score_thres, wait=0):
#         time1 = time.time()
#         result_img,all_box = self.model.head.show_result(
#             meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
#         )
#         print("viz time: {:.3f}s".format(time.time() - time1))
#         return result_img,all_box


#     def detect_person(self,):
#         self.logger.log('Press "Esc", "q" or "Q" to exit.')
#         cap = cv2.VideoCapture(self.args.camid)
#         while True:
#             ret_val, frame = cap.read()
#             if ret_val:
#                 meta, res = self.inference(frame)
#                 person_res = {}
#                 person_res[0] = res[0][0]
#                 # all_boxs：label, x0, y0, x1, y1, score
#                 result_frame,all_box = self.visualize(person_res, meta, cfg.class_names, 0.35)

#                 cv2.imshow("det", result_frame)
#                 ch = cv2.waitKey(1)
#                 if ch == 27 or ch == ord("q") or ch == ord("Q"):
#                     break
#             else:
#                 break


# if __name__ == "__main__":
#     detect_person = Predictor()
#     detect_person.detect_person()


import cv2, sys
from cuda import cuda
import numpy as np
from ctypes import *

cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cam = 1

class EventData(Structure):
    _fields_ = [('client_id', c_int32),
                ('type', c_int32),
                ('user_data', c_void_p),
                ('data', c_void_p)]
    
class ImageData(Structure):
    _fields_ = [('image_id', c_int32),
                ('format', c_int32),
                ('width', c_int32),
                ('height', c_int32),
                ('pitch', c_int32),
                ('devptr', c_void_p)]
    
    
PEventData = POINTER(EventData)
PImageData = POINTER(ImageData)
EventCallback = CFUNCTYPE(None, PEventData)
# cv2.cuda.createGpuMatFromCudaMemory().copyTo(cv2.cuda.GpuMat())

cpuMat = np.zeros([1920, 1080, 4], np.uint8)

print(cpuMat.__array_interface__['shape'])
def event_cb(data):
    print('*******************',data[0].type)
    if data[0].type == 2: # IMAGE_UPDATED
        image = cast(data[0].data, PImageData)[0]
        print('image image_id:',image.image_id)
        print('image format:', image.format)
        print('image width:', image.width)
        print('image height:', image.height)
        print('image pitch:', image.pitch)
        print('image devptr:', image.devptr)



        cpy = cuda.CUDA_MEMCPY2D()
        cpy.WidthInBytes = 1080 * 4
        cpy.Height = 1920
        cpy.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
        cpy.srcDevice = image.devptr
        cpy.srcXInBytes = 0
        cpy.srcY = 0
        cpy.srcPitch = image.pitch
        cpy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_HOST
        cpy.dstHost = cpuMat.__array_interface__['data'][0]
        cpy.dstXInBytes = 0
        cpy.dstY = 0
        ret = cuda.cuMemcpy2D(cpy)

        cv2.imshow('CUDA CAM {}'.format(cam), cv2.cvtColor(cv2.resize(cpuMat, (360, 640)), cv2.COLOR_RGBA2BGR))
        cv2.waitKey(1)

# cudash = cdll.LoadLibrary("./libcudash/libcudash.so")

cudash = CDLL("/workspaces/capella_ros_docker/build/crowd_statistics/crowd_statistics/libcudash/libcudash.so")

client_id=c_int32()
cudash.cudash_create_client(byref(client_id), b"localhost", 6533, None, True)
print("Client id: {}".format(client_id.value))

cb_func = EventCallback(event_cb)
cudash.cudash_client_set_event_callback(client_id, cb_func, None)
cudash.cudash_open_image(client_id, c_int32(cam))

input('Press ENTER to exit.')

cudash.cudash_destroy_client(client_id)

