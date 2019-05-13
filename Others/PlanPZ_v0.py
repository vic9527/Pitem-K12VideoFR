# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 14:51
# @Author  : Vic Woo
# @Email   : vic.woo@vip.163.com
# @File    : PlanPZ.py
# @Software: PyCharm


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

# 导入人脸识别模型
face_rec_model_path = "Models/dlib_face_recognition_resnet_model_v1.dat"
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)


# 从图像中检测人脸部分
def Dlib_detectFace(img):
    shapes = []
    isFaceDetected = 0
    # 对目标图像进行采样，貌似是第二个参数越大识别精度越高。
    faces = detector(img, 0)

    # print len(faces)
    if len(faces) > 0:
        isFaceDetected = 1

    # 对检测出的模型进行计算
    for i, d in enumerate(faces):
        # print "Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom())
        # 读取人脸区域坐标
        # left, right, top, bottom = d.left(), d.right(), d.top(), d.bottom()
        # 利用opencv中的函数进行画出人脸方框。（另：dlib库中有自带的方法可以画出人脸）
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        shape = predictor(img, d)
        shapes.append(shape)

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

    # 返回检测结果
    return isFaceDetected, shapes


# 从图像中识别人脸特征
def Dlib_recogFace(img1, FaceShape1, img2, FaceShape2):
    IsOneFace = 0

    # 计算人脸的128维的向量
    face1_descriptor = face_rec_model.compute_face_descriptor(img1, FaceShape1)
    # print(face_descriptor)
    # for i, num in enumerate(face_descriptor):
    #   print(num)
    #   print(type(num))
    face2_descriptor = face_rec_model.compute_face_descriptor(img2, FaceShape2)

    npVectors1 = np.array([])
    npVectors2 = np.array([])
    for i, num in enumerate(face1_descriptor):
        npVectors1 = np.append(npVectors1, num)
        # print(num)
    for i, num in enumerate(face2_descriptor):
        npVectors2 = np.append(npVectors2, num)

    npVectorsdiffdiff = 0
    # for v1, v2 in data1, data2:
    # diff += (v1 - v2)**2
    for i in xrange(len(npVectors1)):
        npVectorsdiffdiff += (npVectors1[i] - npVectors2[i]) ** 2
    npVectorsdiffdiff = np.sqrt(npVectorsdiffdiff)
    print npVectorsdiffdiff
    if (npVectorsdiffdiff < 0.6):
        IsOneFace = 1
        print "It's the same person"
    else:
        print "It's not the same person"

    return IsOneFace


def FaceDR(VideoPath, intervalTime=5):
    camera = cv2.VideoCapture(VideoPath)
    camera_open_success = camera.isOpened()
    video_fps = round(camera.get(5))
    # video_frames = round(camera.get(7))
    # video_frames_PrintPercent = video_frames / 100.0
    # print u'摄像头FPS:%d, FRAMES:%d!' % (video_fps, video_frames)
    FaceShowRatio_Mean = []
    if not camera_open_success:
        print u"摄像头未打开，请重启摄像头！"
    else:
        FramesCNT = 0
        FaceShowRatio_Res = []
        BrightnessValue_Res = []

        while camera_open_success:
        # while FramesCNT < video_frames:
            ret, cv_img = camera.read()
            frame = cv_img
            # 【高y，宽x】
            # frame = cv_img[0:240, :]
            if not ret:
                print u"未正常获取视频，请检查摄像头！"
                break
            else:
                if intervalTime == 0:
                    # 检测人脸
                    bFaceDetected, Faceshapes = Dlib_detectFace(frame)
                    FaceShowRatio_Res.append(bFaceDetected)

                    # 识别人脸
                    if bFaceDetected:
                        img2 = cv2.imread("TestVideo/myface.jpg")
                        _, FaceShapes2 = Dlib_detectFace(img2)
                        IsOneFace = Dlib_recogFace(frame, Faceshapes[0], img2, FaceShapes2[0])
                        print IsOneFace

                    # FaceShowRatio_Mean = float(np.sum(FaceShowRatio_Res)) / len(FaceShowRatio_Res)
                    # print FaceShowRatio_Mean
                else:
                    intervalFrames = video_fps * intervalTime
                    if FramesCNT % intervalFrames == 0:
                        bFaceDetected, Faceshapes = Dlib_detectFace(frame)
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
        FaceShowRatio_Mean = float(np.sum(FaceShowRatio_Res)) / len(FaceShowRatio_Res)
    return FaceShowRatio_Mean


if __name__ == '__main__':
    VideoPath = "TestVideo/video_52158706.mp4"
    FaceDR(0, 0)
