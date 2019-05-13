# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 14:51
# @Author  : Vic Woo
# @Email   : vic.woo@vip.163.com
# @File    : PlanPZ.py
# @Software: PyCharm

# Common Issues
#
# Issue: Illegal instruction (core dumped) when using face_recognition or running examples.
#
# Solution: dlib is compiled with SSE4 or AVX support, but your CPU is too old and doesn't support that. You'll need to recompile dlib after making the code change outlined here .
#
# Issue: RuntimeError: Unsupported image type, must be 8bit gray or RGB image. when running the webcam examples.
#
# Solution: Your webcam probably isn't set up correctly with OpenCV. Look here for more .
#
# Issue: MemoryError when running pip2 install face_recognition
#
# Solution: The face_recognition_models file is too big for your available pip cache memory. Instead, try pip2 --no-cache-dir install face_recognition to avoid the issue.
#
# Issue: AttributeError: 'module' object has no attribute 'face_recognition_model_v1'
#
# Solution: The version of dlib you have installed is too old. You need version 19.7 or newer. Upgrade dlib .
#
# Issue: Attribute Error: 'Module' object has no attribute 'cnn_face_detection_model_v1'
#
# Solution: The version of dlib you have installed is too old. You need version 19.7 or newer. Upgrade dlib .
#
# Issue: TypeError: imread() got an unexpected keyword argument 'mode'
#
# Solution: The version of scipy you have installed is too old. You need version 0.17 or newer. Upgrade scipy .

import os, sys
current_path = os.getcwd()  # 获取当前路径
sys.path.append("D:\\TortoiseGit_GitLab\\bi-AIQC\\AIQC_Server\\PyLibs")
import numpy as np
import PyLibs.cv320.cv2 as cv2
import PyLibs.dlib1916.dlib as dlib

# 利用Dlib自带的检测器(无需装换成灰度图)
detector = dlib.get_frontal_face_detector()

# landmarkdat
predictor_path = "Models/shape_predictor_68_face_landmarks.dat"
# 初始化landmark
predictor = dlib.shape_predictor(predictor_path)

# 导入cnn模型(CPU太卡)
cnn_face_detector_path = "Models/mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)

# 导入人脸识别模型（models/Models不一样，Python对大小写敏感。）
# face_rec_model_path = "Models/dlib_face_recognition_resnet_model_v1.dat"
# face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)



# 从图像中检测人脸部分
def Dlib_detectFace(img):

    shapes = []
    isFaceDetected = 0
    # # 对目标图像进行采样，貌似是第二个参数越大识别精度越高。
    # faces = detector(img, 0)
    # cnn模型进行检测
    faces = cnn_face_detector(img, 0)

    # print len(faces)
    if len(faces) > 0:
        isFaceDetected = 1

    # 对检测出的模型进行计算
    for i, d in enumerate(faces):
        face = d.rect
        # print "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom())
        print "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, face.left(), face.top(), face.right(), face.bottom())
        # 读取人脸区域坐标
        # left, right, top, bottom = d.left(), d.right(), d.top(), d.bottom()
        # 在图片中标出人脸
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        # 利用opencv中的函数进行画出人脸方框。（另：dlib库中有自带的方法可以画出人脸）
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        # shape = predictor(img, d)
        # shapes.append(shape)

    # # Draw the face and landmarks on the screen.
    # win.clear_overlay()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # win.set_image(img)
    # win.add_overlay(faces)
    # if len(shapes) != 0:
    #     for i in range(len(shapes)):
    #         win.add_overlay(shapes[i])

    # # 待会要写的字体
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # # 标 68 个点
    # if len(faces) != 0:
    #     # 检测到人脸
    #     for i in range(len(faces)):
    #         landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[i]).parts()])
    #
    #         for idx, point in enumerate(landmarks):
    #             # 68 点的坐标
    #             pos = (point[0, 0], point[0, 1])
    #
    #             # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
    #             cv2.circle(img, pos, 2, color=(139, 0, 0))
    #
    #             # 利用 cv2.putText 输出 1-68
    #             cv2.putText(img, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
    #
    #     cv2.putText(img, "faces: " + str(len(faces)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    # else:
    #     # 没有检测到人脸
    #     cv2.putText(img, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    #
    # # # 添加说明
    # # im_rd = cv2.putText(img, "press 'S': screenshot", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    # # im_rd = cv2.putText(img, "press 'Q': quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 返回检测结果
    return isFaceDetected



net = cv2.dnn.readNetFromCaffe("Models/deploy.prototxt", "Models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
def MTCNN_detectFace(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    isFaceDetected = 0
    text = 0
    h, w, c = image.shape
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # 默认0.6
        if confidence > 0.2:
            isFaceDetected = 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 255, 0), 1)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return isFaceDetected, text

# 从图像中识别人脸特征
def Dlib_recogFace(img):
    FaceID = []
    _, RecShapes = Dlib_detectFace(img)
    face_descriptor = face_rec_model.compute_face_descriptor(img, RecShapes)  # 计算人脸的128维的向量

    return FaceID


def FaceDR(VideoPath, intervalTime = 5):
    t_start = 60 * 10
    t_stop = 60 * 20
    # VideoMinSize = 80 * 1024 * 1024  # 80M

    # 网络视频
    # camera = cv2.VideoCapture("https://1251606527.vod2.myqcloud.com/30eb845dvodbj1251606527/bac332f65285890782260259595/k9EpAPlV1n8A.mp4")
    # camera = cv2.VideoCapture(0)
    # camera = cv2.VideoCapture("TestVideo/f3.mp4")
    camera = cv2.VideoCapture(VideoPath)
    # camera.set(5, 30)
    camera_open_success = camera.isOpened()
    video_fps = round(camera.get(5))
    video_frames = round(camera.get(7))
    video_frames_PrintPercent = video_frames/100.0
    print u'摄像头FPS:%d, FRAMES:%d!' % (video_fps, video_frames)



    # start frame
    f_start = t_start * video_fps + 1
    # end frame
    f_stop = t_stop * video_fps
    if f_stop > video_frames:
        f_stop = video_frames

    frs = f_start
    frsAll = f_stop - f_start + 1
    # 设置视频帧的读取位置CAP_PROP_POS_FRAMES(按帧数https: // www.cnblogs.com / ronny / p / opencv_road_10.html)
    camera.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    FaceShowRatio_Mean = []
    if not camera_open_success:
        print u"摄像头未打开，请重启摄像头！"
    else:
        FramesCNT = 0
        FaceShowRatio_Res = []
        BrightnessValue_Res =[]
        while FramesCNT < video_frames:
            ret, cv_img = camera.read()
            # 【高y，宽x】
            frame = cv_img[0:240, :]
            if not ret:
                print u"未正常获取视频，请检查摄像头！"
                break
            else:
                # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # faces = detector(frame, 0)
                # print("Number of faces detected: {}".format(len(faces)))

                # print Dlib_detectFace(frame)
                if intervalTime == 0:
                    # bFaceDetected, _ = Dlib_detectFace(frame)
                    bFaceDetected, _ = MTCNN_detectFace(frame)

                    FaceShowRatio_Res.append(bFaceDetected)



                    # FaceShowRatio_Mean = float(np.sum(FaceShowRatio_Res)) / len(FaceShowRatio_Res)
                    # print FaceShowRatio_Mean
                else:
                    intervalFrames = video_fps * intervalTime
                    if FramesCNT % intervalFrames == 0:
                        bFaceDetected = Dlib_detectFace(frame)
                        FaceShowRatio_Res.append(bFaceDetected)


                        # FaceShowRatio_Mean = float(np.sum(FaceShowRatio_Res)) / len(FaceShowRatio_Res)
                        # print FaceShowRatio_Mean

                FramesCNT += 1
                # if FramesCNT % video_frames_PrintPercent == 0:
                #     # FaceShowRatio_Mean = float(np.sum(FaceShowRatio_Res)) / len(FaceShowRatio_Res)
                #     # print FaceShowRatio_Mean
                #     print str(round(float(FramesCNT) / video_frames, 2) * 100) + "%"
                cv2.imshow("Demo", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        # FaceShowRatio_Mean = np.sum(FaceShowRatio_Res)*1.0 / np.size(FaceShowRatio_Res)
        FaceShowRatio_Mean = float(np.sum(FaceShowRatio_Res)) / len(FaceShowRatio_Res)
        # print FaceShowRatio_Mean.dtype
        # camera.release()
        # cv2.destroyAllWindows()
    return str(round(FaceShowRatio_Mean, 4)*100) + "%"


if __name__ == '__main__':
    VideoPath = "TestVideo/video_52158706.mp4"
    print FaceDR(VideoPath, 0)
