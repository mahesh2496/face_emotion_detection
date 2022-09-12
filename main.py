import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from PIL import Image
from utils.vgg import create_RepVGG_A0 as create

root = os.getcwd()

model = create(deploy=True)


class EmotionDet:
    def init(self, device):
        # Initialise model

        global dev
        dev = device
        model.to(device)
        checkpoint = torch.load("weights/vgg.pth")
        model.load_state_dict(checkpoint)

        # Change to classify only 8 features
        model.linear.out_features = 1000
        model.linear._parameters["weight"] = model.linear._parameters["weight"][:1000, :]
        model.linear._parameters["bias"] = model.linear._parameters["bias"][:1000]

        cudnn.benchmark = True
        model.eval()

    def detect_emotion(self,images, conf=True):

        emotions = ("anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")
        with torch.no_grad():
            # Normalise and transform images
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            x = torch.stack([transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])(Image.fromarray(image)) for image in images])
            # Feed through the model
            y = model(x.to(dev))
            result = []
            for i in range(y.size()[0]):
                # Add emotion to result
                emotion = (max(y[i]) == y[i]).nonzero().item()
                # Add appropriate label if required
                result.append(
                    [f"{emotions[emotion]}{f' ({100 * y[i][emotion].item():.1f}%)' if conf else ''}", emotion])
        return result


class Detection(EmotionDet):
    
    def __init__(self, url, name):
        
        self.iou_thres = 0.45
        self.conf_thres = 0.5
        self.line_thickness = 2
        self.imgsz = 512
        self.show_conf = True
        self.source = url
        self.name = name

    def detect(self):

        self.paths = root + "/" + str(name)
        if not os.path.exists(self.paths):
            os.mkdir(self.paths)
        else:
            shutil.rmtree(self.paths)
            os.mkdir(self.paths)

        
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        device = select_device()

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init(device)

        model = attempt_load("weights/yolo.pt", map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsize = check_img_size(self.imgsz, s=stride)  # check img_size

        if webcam:

            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=imgsize, stride=stride)
        else:
            dataset = LoadImages(self.source, img_size=imgsize, stride=stride)

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = (
            (0, 52, 255), (121, 3, 195), (176, 34, 118), (87, 217, 255), (69, 199, 79), (233, 219, 155), (203, 139, 77),
            (214, 246, 255))

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment='store_true')[0]

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic='store_true')

            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    images = []
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = xyxy
                        images.append(im0.astype(np.uint8)[int(y1):int(y2), int(x1): int(x2)])

                    if images:
                        emotions = self.detect_emotion(images, self.show_conf)
                    # Write results
                    i = 0
                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image with emotions on
                        label = emotions[i][0]
                        colour = colors[emotions[i][1]]
                        i += 1
                        plot_one_box(xyxy, im0, label=label, color=colour, line_thickness=self.line_thickness)

                        display_img = cv2.resize(im0, (im0.shape[1] * 2, im0.shape[0] * 2))
                        rez = cv2.resize(display_img, (640, 450))
                        cv2.imshow(self.source, rez)
                        cv2.waitKey(1)  # 1 millisecond
                        filename = self.paths + "/image_" + str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + ".jpg"
                        cv2.imwrite(filename, rez)


if __name__ == '__main__':

    arr = [
           {"name": "rtsp1", "url": r"rtsp://admin:Admin123$@10.11.25.64:554/stream1"},
           # {"name": "rtsp2","url": "0"},
           # {"name": "rtsp3","url": r"D:\videos\face_em.mp4"},
           # {"name": "rtsp4","url": r"rtsp://admin:Admin123$@10.11.25.65:554/stream1"},
           {"name": "rtsp5","url": r"rtsp://admin:Admin123$@10.11.25.60:554/stream1"},
           {"name": "rtsp6","url": r"rtsp://admin:Admin123$@10.11.25.62:554/stream1"}

           ]
    for i in arr:
        url = i['url']
        name = i["name"]
        t1 = threading.Thread(target=Detection(url,name).detect)
        t1.start()

