import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized, TracedModel
from utils.parking import whRatio, plot_lines, plot_card, readVertices, isPointInside

import firebase_admin
from firebase_admin import credentials, db

conf_thres = 0.55
iou_thres = 0.65

def detect(weights, imgsz):
    # cv2_img = cv2.imread(source, 1)

    # 이 부분은 내가 임의로 필터를 씌운 곳.
    # sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # cv2_img = cv2.filter2D(cv2_img, -1, sharpening_mask1)

    set_logging()
    device = select_device('cpu')
    half = False

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    model = TracedModel(model, device, imgsz)

    # Set Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams('0', img_size=imgsz, stride=stride)

    # # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors=[[0,255,0],[255,0,0]]
    # # colors = [[255, 0, 0] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    vertices, max_slot = readVertices('./slot_vertices.txt')

    slots_buff1 = {}
    slots_buff2 = {}
    slots_buff3 = {}

    for j in range(max_slot):
        slots_buff1[f'slot_{str(j).zfill(2)}'] = False
        slots_buff2[f'slot_{str(j).zfill(2)}'] = False
        slots_buff3[f'slot_{str(j).zfill(2)}'] = False
    
    slots = {}

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                # model(img, augment=opt.augment)[0]
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            # pred = model(img, augment=opt.augment)[0]
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=True)
        t3 = time_synchronized()

        slots_buff3 = dict.copy(slots_buff2)
        slots_buff2 = dict.copy(slots_buff1)
        

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            for j in range(max_slot):
                slots_buff1[f'slot_{str(j).zfill(2)}'] = False

            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / p.stem) + ('')  # img.txt

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results

                for *xyxy, conf, _ in reversed(det):
                    # this is for check slots availability
                    xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()

                    x, y, w, h = xywh[0:4]

                    # w, h의 비율에 따라 좌표값을 정함.
                    y_ratioed = y + (h / 4) * whRatio(w, h)
                    plot_card(im0, xyxy, conf)
                    cv2.circle(im0, (int(x), int(y_ratioed)), 2, (0,0,255), -1)

                    for i in range(max_slot):
                        slots_buff1[f"slot_{str(i).zfill(2)}"] = slots_buff1[f"slot_{str(i).zfill(2)}"] or isPointInside(int(x), int(y_ratioed), vertices[i])

        for i in range(max_slot):
            slots[f"slot_{str(i).zfill(2)}"] = slots_buff1[f"slot_{str(i).zfill(2)}"] or slots_buff2[f"slot_{str(i).zfill(2)}"] or slots_buff3[f"slot_{str(i).zfill(2)}"]

        plot_lines(im0, slots, max_slot, vertices)

        try:
            dir = db.reference()
            dir.update(slots)
        except:
            print("(WARNING) cannot update FireBase")

        # cv2.imwrite(save_path, im0)
        cv2.imshow('image', im0)
        
        if cv2.waitKey(30)==27:
            exit()

        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        # print(slots_buff1['slot_01'], slots_buff2['slot_01'], slots_buff3['slot_01'])
        # time.sleep(2)

        # 매 프레임마다 firebase에 업로드를 하면 지연시간이 굉장히 길어지는 것 같음.
        # 따라서 몇몇 프레임마다 업데이트 하거나, 10초에 한번 씩 업데이트 하는 방법 등을 이용해야 할 것 같음.


if __name__ == '__main__':

    # source = 'img/' # should attach '/' at last
    weights = 'weights/best.pt'
    imgsz = 640
    # dir_name = 'test_1'
        
    try:
        cred = credentials.Certificate('certificate/yolov7-pklot-firebase-adminsdk-3wz6y-ce022d9c98.json')
        firebase_admin.initialize_app(cred,{
            'databaseURL' : 'https://yolov7-pklot-default-rtdb.firebaseio.com'
        })
    except:
        print("Unable to connect FireBase.")

    detect(weights, imgsz)
    cv2.destroyAllWindows()