import time

import cv2
import numpy as np

from .model import Backbone, l2_norm, MobileFaceNet
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from PIL import Image
from .mtcnn import MTCNN

from torchvision import transforms as trans
from torchvision.transforms import Compose, ToTensor, Normalize

import threading
import uuid

from cuda import cuda




class Match_Face():
    def __init__(self,face_feature_path=None,face_name_path=None,model_path=None,device=None):

        # 获取device
        if device == None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 初始化MTCNN模型
        self.mtcnn = MTCNN(self.device)
        # 读取特征和名字数据
        # 将时间戳转换为本地时间的struct_time对象
        local_time = time.localtime(time.time())
        # 使用strftime()方法将struct_time对象格式化为指定的时间字符串
        self.current_time = time.strftime("%Y_%m_%d", local_time)
        self.feature_save_path = os.path.join(os.path.dirname(__file__),r'work_space/{}_face_feature.npy'.format(self.current_time))
        self.name_save_path = os.path.join(os.path.dirname(__file__),r'work_space/{}_face_name.npy'.format(self.current_time))
        # 读取人脸特征文件
        if not os.path.exists(self.feature_save_path):
            face_feature = np.array([])

            np.save(self.feature_save_path,face_feature)
        if not os.path.exists(self.name_save_path):
            face_name = np.array([])
            np.save(self.name_save_path,face_name)
        self.face_features = torch.tensor(np.load(self.feature_save_path,allow_pickle=True)).to(self.device)
        self.face_names = np.load(self.name_save_path,allow_pickle=True)


        # 初始化脸部特征提取模型
        # 使用ir-se模型
        self.model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),r'model_ir_se50.pth'),map_location=torch.device(self.device)))
        # 使用MobileFaceNet模型
        # self.model = MobileFaceNet(embedding_size=512)
        # self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),r'model_mobilefacenet.pth'),map_location=torch.device(self.device)))
        self.model.eval()
        self.model.to(self.device)

        self.threshold = 1.54
        self.face_feature_len = 0

        save_thread = threading.Thread(target=self.save_face_feature,daemon=True)
        save_thread.start()
        print('Match_Face 模型初始化完成......')


    def save_face_feature(self,):
        self.face_feature_len = len(self.face_features)
        while True:
            time.sleep(60)
            if self.face_feature_len != len(self.face_features):
                np.save(self.feature_save_path, self.face_features.cpu().detach().numpy())
                np.save(self.name_save_path,self.face_names)
                self.face_feature_len = len(self.face_features)
                print('face feature and name 已经保存')
            else:
                continue

    def match_feature(self, emb):
        if len(self.face_features) == 0:
            print('来了一个新面孔，加入特征库')
            self.face_features = torch.cat([self.face_features,emb])
            self.face_names = np.append(self.face_names,str(uuid.uuid1()))

            min_idx, minimum = -1,-1
        else:
            diff = emb - self.face_features
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            min_idx = torch.argmin(dist).detach().item()
            minimum = dist[min_idx].detach().item()

            if minimum > self.threshold:
                print(f'来了一个新面孔，加入特征库.')
                self.face_features = torch.cat([self.face_features, emb])
                self.face_names = np.append(self.face_names, str(uuid.uuid1()))
                min_idx, minimum = -1,-1

        return min_idx, minimum

    # 获取并预处理图片：对齐、归一化
    def preperation_img(self,image):
        if not image.is_contiguous():
            image = image.contiguous()
        img = cv2.cuda.createGpuMatFromCudaMemory(
            image.shape[0],image.shape[1],cv2.CV_8UC3,image.data_ptr())
        img = cv2.cuda.resize(img, (112, 112))
        img = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2RGB)

        # face = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # # mtcnn.align只要PIL.Image类型的图片

        # 将图片拷贝到tensor
        w,h = img.size()
        gpuTensor = torch.zeros((h,w,3),device=self.device,dtype=torch.uint8)
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

        face = self.mtcnn.align(gpuTensor)
        if face is None:
            return None
        face = face.permute((2,0,1)).contiguous().div(255)
        transfroms = Compose(
            [Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) #ToTensor(),
        
        return transfroms(face).to(self.device).unsqueeze(0)

    def get_face_feature(self,img,tta=True):
        t1 = time.time()
        img = self.preperation_img(img)
        if img is None:
            min_idx, minimum, min_name = -1, -1, -1
            return min_idx, minimum, min_name
        # 是否翻转图片
        if tta:
            # img = torch.tensor(np.array([[1],[2],[3]] * 12544).reshape((112,112,3)).T.astype(float),device=self.device).unsqueeze(0).float()
            mirror = trans.functional.hflip(img)
            emb = self.model(img)
            emb_mirror = self.model(mirror)
            emb = l2_norm(emb + emb_mirror)
        else:
            emb = self.model(img)

        min_idx, minimum = self.match_feature(emb)
        min_name = self.face_names[min_idx]


        return min_idx, minimum, min_name
if __name__ == "__main__":
    face_ = Match_Face()
    min_idx, minimum, min_name = face_.get_face_feature(1)

