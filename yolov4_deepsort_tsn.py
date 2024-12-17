import cv2
import torch
import warnings
import numpy as np

# yolov4检测器
from detector.yolo import YOLO
# deepsort跟踪器
from deep_sort import build_tracker

from utils.draw import draw_boxes

# from video_class.net import R3DClassifier
from video_class.net import C3D

from kls_image_process import cv2_letterbox_image
from mmaction2.file.inference import inference_recognizer, init_recognizer

config_file = r'/home/kls/Code_warehouse/yolov4_deepsort_jyj2/mmaction2/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb.py'
checkpoint = r'/home/kls/桌面/work_dirs/tsn_r50_video_1x1x8_100e_kinetics400_rgb/epoch_100.pth'
video_path = r'/mnt/marathon/pingpangclips/数据集/VideoCls-data/数据集/侧面素材（WTT）/pp_class/train/playball/kls27.mp4'
label = 'mmaction2/tools/data/kinetics/label_map_k400.txt'


class VideoTracker(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.list_box = []
        self.list_conf = []
        self.sf_num = 2.5
        self.desk_zb = None
        self.frame_interval = 2
        # 使用GPU并且用cuda加速
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("在cpu模式下运行，可能非常慢!", UserWarning)
        self.vdo = cv2.VideoCapture()
        self.detector = YOLO()
        self.deepsort = build_tracker(use_cuda=True)
        # self.class_names = self.detector.class_names

    def __enter__(self):
        self.vdo.open(self.video_path)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self, device, model, step):
        idx_frame = 0  # 帧数
        video_sub_clips = []  # 剪辑视频列表
        id_dict = {}  # 存目标信息、目标roi图，注意这里面的值都是按照16帧存储
        # show take ball
        is_show = -1
        is_show_count = 0
        last_show_idx_frame = 0

        while self.vdo.grab():
            idx_frame += 1  # 计算帧数
            # if idx_frame < 15000:  # 用于剪辑的视频，小于4秒，跳出循环
            #     continue
            if idx_frame % self.frame_interval:
                continue
            ref, ori_im = self.vdo.retrieve()  # 抓取每一帧
            if ref is True:  # 如果摄像头是开着的，为True
                frame = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)  # 转pil格式
                bbox_xywh, cls_conf, cls_ids = self.detector.detect_image(frame)  # 每一帧送入检测，返回目标位置信息、置信度、类别id
                # print(bbox_xywh)
                table_index = -1
                for i in range(len(cls_ids)):
                    # judgement table class
                    if 1 == cls_ids[i]:
                        table_index = i
                        break
                if table_index >= 0:
                    self.desk_zb = bbox_xywh[table_index]
                else:
                    if self.desk_zb is None:
                        self.desk_zb = [int(frame.shape[1] / 2), int(frame.shape[0] / 2), frame.shape[1],
                                        frame.shape[0]]
                    else:
                        pass
                # self.desk_zb = bbox_xywh[cls_ids.index(1)]

                nlen_box = len(bbox_xywh)
                person_boxes, person_confs, person_clss = [], [], []
                for i in range(nlen_box):
                    if int(self.desk_zb[1] / self.sf_num) < int(bbox_xywh[i][1]) < int(self.desk_zb[1] * self.sf_num):
                        if (int(cls_ids[i])) == 0:
                            person_boxes.append(bbox_xywh[i])
                            person_confs.append(cls_conf[i])
                            person_clss.append(cls_ids[i])
                        else:
                            pass
                if cls_conf is not None and len(cls_conf) > 0:  # 判断当前帧检测到目标，则向下执行
                    # -----#-----掩模处理过滤掉无用的部分
                    new_bbox = np.array(person_boxes).astype(np.float32)
                    cls_conf = np.array(person_confs).astype(np.float32)
                    cls_ids = np.array(person_clss).astype(np.float32)

                    if len(cls_ids) > 0:
                        self.list_box.append(new_bbox)
                        self.list_conf.append(cls_conf)
                        outputs = self.deepsort.update(new_bbox, cls_conf, ori_im)
                    else:
                        if (self.list_box is not None) and len(self.list_box) > 0:
                            new_bboxx = self.list_box[-1]
                            cls_conff = self.list_conf[-1]
                            outputs = self.deepsort.update(new_bboxx, cls_conff, ori_im)
                        else:
                            continue
                    if len(self.list_box) > 10:
                        self.list_box.pop(0)
                        self.list_conf.pop(0)

                    # 可视化跟踪，想看跟踪效果就放开
                    if len(outputs) > 0:  # 判断当前帧跟踪成功目标
                        bbox_xyxy = outputs[:, :4]  # 取当前帧跟踪所有目标的位置信息，生成bbox_xyxy
                        identities = outputs[:, -1].tolist()  # 取当前帧跟踪的所有目标的id,生成identities
                        for id in identities:  # 遍历当前帧跟踪的所有目标的id列表
                            if id == is_show:  # 如果当前帧的所有目标中，有目标的id = -1，则向下执行
                                last_show_idx_frame = idx_frame  # last_show_idx_frame为当前帧数，表示截止帧数
                                cv2.imshow("takeball",  # 并显示这个目标
                                           ori_im[
                                           bbox_xyxy[identities.index(id)][1]:
                                           bbox_xyxy[identities.index(id)][3],
                                           bbox_xyxy[identities.index(id)][0]:
                                           bbox_xyxy[identities.index(id)][2]])
                                is_show_count += 1
                                if is_show_count > 20:
                                    is_show_count = 0
                                    is_show = -1
                                    cv2.destroyWindow("takeball")
                        img = draw_boxes(ori_im, bbox_xyxy[0:2], identities[0:2])  # 绘制目标框并显示
                        cv2.imshow('imshow', img)
                        cv2.waitKey(1)

                    # 如果跟踪到目标，则向下执行
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1].tolist()
                        # 当前帧根据目标id存储目标的位置信息和roi图
                        for id in identities:
                            if id not in id_dict.keys():
                                id_dict[id] = {'id_list': [], 'frame_list': [], "left_time": 0}

                                id_dict[id]['id_list'].append(bbox_xyxy[identities.index(id)])
                                fram1 = cv2_letterbox_image(ori_im[bbox_xyxy[identities.index(id)][1]:bbox_xyxy[identities.index(id)][3],
                                    bbox_xyxy[identities.index(id)][0]:bbox_xyxy[identities.index(id)][2]],(416,416))
                                id_dict[id]['frame_list'].append(fram1)
                            else:
                                id_dict[id]['id_list'].append(bbox_xyxy[identities.index(id)])
                                fram2 = cv2_letterbox_image(ori_im[bbox_xyxy[identities.index(id)][1]:bbox_xyxy[identities.index(id)][3],
                                    bbox_xyxy[identities.index(id)][0]:bbox_xyxy[identities.index(id)][2]],(416,416))
                                id_dict[id]['frame_list'].append(fram2)

                                # 之前的问题：之前是保存每个目标的位置信息和roi图后，没有考虑显存问题去做删减处理，
                                # 导致位置信息列表和roi列表一直在存储数据，导致显存一直在增长，导致电脑死机！！！

                                # 改进之后：每次存完当前帧所有目标的位置信息和roi图后，对列表进行判断长度是否大于16
                                # 如果超过就在每个目标的位置信息和roi图列表中删除第一个值，每一帧存完都有做删减处理
                                # 降低显存的占用！！！
                                if len(id_dict[id]['id_list']) > 16:  # 如果每个目标的位置信息列表大于16，就把位置信息列表中的每个目标的第一个位置信息删除
                                    id_dict[id]['id_list'].pop(0)
                                if len(id_dict[id]['frame_list']) > 16:  # 如果每个目标的roi列表大于16，就把位置信息列表中的每个目标的第一个roi删除
                                    id_dict[id]['frame_list'].pop(0)
                        to_del = []  # 标记当前帧不该出现的目标

                        for k in id_dict.keys():  # 如果在当前帧出现了不该出现的目标，
                            if k not in identities:
                                id_dict[k]["left_time"] += 1
                                if id_dict[k]["left_time"] >= 9:  # 如果在连续9帧都出现
                                    to_del.append(k)  # to_del保存不该出现目标的id号
                        for de in to_del:
                            id_dict.pop(de)  # 就从字典中删除掉这个目标
                        if idx_frame % step == 0:
                            for i in range(len(identities)):
                                # if i == 2:
                                #     break
                                if len(id_dict[identities[i]]['id_list']) >= 16:
                                    temp_sub = {
                                        "id_list": id_dict[identities[i]]['id_list'][:16],
                                        "frame_list": id_dict[identities[i]]['frame_list'][:16],
                                        "track_id": identities[i]
                                    }
                                    video_sub_clips.append(temp_sub)
                        else:
                            video_sub_clips = []
                    else:
                        print('未能检测到两个目标')
                else:
                    print('置信度为None')

                sub_clip_recognition_result = []
                for sub_clip_ind in range(len(video_sub_clips)):
                    ret = video_class_single(video_sub_clips[sub_clip_ind]["id_list"],
                                             video_sub_clips[sub_clip_ind]["frame_list"],
                                             device, model, video_sub_clips[sub_clip_ind]["track_id"])
                    sub_clip_recognition_result.append(ret)
                video_sub_clips = []
                print('sub_clip==============', sub_clip_recognition_result)
                conf, max_wz, max_track_id = take_locate(sub_clip_recognition_result, self.desk_zb)
                if max_wz == 0 and conf > 0.05:
                    print("上半区发球")
                    print("jia --------------:", idx_frame, " conf", conf, " track_id ", max_track_id)
                    if idx_frame - last_show_idx_frame > 125:
                        if is_show_count < 0:
                            is_show_count = 0
                        is_show = max_track_id

                elif max_wz == 1 and conf > 0.05:
                    print("下半区发球")
                    print("jia --------------:", idx_frame, " conf", conf, " track_id ", max_track_id)
                    if idx_frame - last_show_idx_frame > 125:
                        if is_show_count < 0:
                            is_show_count = 0
                        is_show = max_track_id
                else:
                    pass
            else:
                print('ret为False')


# 先只看前两个数据 待后续考虑
def take_locate(sub_clip_recognition_result, desk_zb):
    ret = -1
    max_conf = 0.0
    timestamp = -1

    index_ret = -1
    if sub_clip_recognition_result is not None:
        for i in range(len(sub_clip_recognition_result)):
            if sub_clip_recognition_result[i][0] == 1 and sub_clip_recognition_result[i][1] > max_conf:
                index_ret = i
                max_conf = sub_clip_recognition_result[i][1]

    if index_ret >= 0:
        timestamp = sub_clip_recognition_result[index_ret][-1]
        if sub_clip_recognition_result[index_ret][2] < desk_zb[0]:
            ret = 0
        else:
            ret = 1
    return max_conf, ret, timestamp


def video_class_single(iid1_list, rframe1_list, device, model, track_id):
    mean_x = 0
    mean_y = 0
    label, max_score = 0, 0
    if len(rframe1_list) == 16:
        for i in range(len(rframe1_list)):
            mean_x += round((iid1_list[i][0] + iid1_list[i][2]) / (2 * len(rframe1_list)))
            mean_y += round((iid1_list[i][1] + iid1_list[i][3]) / (2 * len(rframe1_list)))
        label, max_score = inference_recognizer(model, rframe1_list)
    return label, max_score, mean_x, mean_y, track_id


def main(gen_root):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Device being used:", device)
    #
    # model = C3D(num_classes=2)
    # # model = R3DClassifier(num_classes=2, layer_sizes=(2, 2, 2, 2))
    # checkpoint = torch.load(r'video_class/model/C3D-myself_epoch-49.pth.tar',
    #                         map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.to(device)
    # model.eval()
    model = init_recognizer(config_file, checkpoint, device='cuda:0')
    with VideoTracker(video_path=gen_root) as vdo_trk:
        vdo_trk.run(device, model, 16)


if __name__ == "__main__":
    gen_root = r'/mnt/marathon/pingpangclips/miguwtt/LWY/2021WTT常规挑战赛拉什科站男单14决赛：王楚钦2_3刘丁硕.mp4'
    main(gen_root)
