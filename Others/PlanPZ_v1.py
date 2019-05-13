# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 14:51
# @Author  : Vic Woo
# @Email   : vic.woo@vip.163.com
# @File    : PlanPZ.py
# @Software: PyCharm


import os, sys
current_path = os.getcwd()  # 获取当前路径
PyLibs_path = current_path + "/PyLibs"
sys.path.append(PyLibs_path)
import numpy as np
import PyLibs.scipy.misc as sm

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
    if len(faces) > 0:
        isFaceDetected = 1

    # 对检测出的模型进行计算
    for i, d in enumerate(faces):
        shape = predictor(img, d)
        shapes.append(shape)

    # 返回检测结果
    return isFaceDetected, shapes



# 从图像中识别人脸特征
def Dlib_recogFace(img1, FaceShape1, img2, FaceShape2):
    IsOneFace = 0

    # 计算人脸的128维的向量
    face1_descriptor = face_rec_model.compute_face_descriptor(img1, FaceShape1)
    face2_descriptor = face_rec_model.compute_face_descriptor(img2, FaceShape2)

    npVectors1 = np.array([])
    npVectors2 = np.array([])
    for i, num in enumerate(face1_descriptor):
        npVectors1 = np.append(npVectors1, num)
    for i, num in enumerate(face2_descriptor):
        npVectors2 = np.append(npVectors2, num)

    npVectorsdiffdiff = 0
    for i in xrange(len(npVectors1)):
        npVectorsdiffdiff += (npVectors1[i] - npVectors2[i]) ** 2
    npVectorsdiffdiff = np.sqrt(npVectorsdiffdiff)
    # print npVectorsdiffdiff
    if npVectorsdiffdiff < 0.5:
        IsOneFace = 1
    return IsOneFace



def PairWiseComparison(imgV, FaceShapeV):
    lenNum = len(imgV)
    IsIllegal, ComparisonMatrix = 0, np.eye(lenNum, k=0)
    ComparisonResults = []
    if lenNum >= 2:
        for i in range(0, lenNum):
            for j in range(i+1, lenNum):
                if Dlib_recogFace(imgV[i], FaceShapeV[i], imgV[j], FaceShapeV[j]) == 1:
                    ComparisonMatrix[i][j] = 1
                    ComparisonResults.append([i+1,j+1])

    lenCR = len(ComparisonResults)
    if lenCR > 0:
        IsIllegal = 1

    for i in range(0, lenCR - 1):
        for j in range(i+1,lenCR):
            CRtmp = [val for val in ComparisonResults[i] if val in ComparisonResults[j]]
            if CRtmp:
                ComparisonResults[i] = list(set(set(ComparisonResults[i]).union(set(ComparisonResults[j]))))
                ComparisonResults[j] = []
    ComparisonResults_New = [k for k in ComparisonResults if k != []]

    return IsIllegal, ComparisonResults_New



def FaceDR(VideoPath, intervalTime=5):
    fileName = VideoPath.split("/")[-1]
    camera = cv2.VideoCapture(VideoPath)
    camera_open_success = camera.isOpened()
    video_fps = round(camera.get(5))
    video_frames = round(camera.get(7))
    video_frames_PrintPercent = int(video_frames / 100)
    print u'摄像头FPS:%d, FRAMES:%d!' % (video_fps, video_frames)

    FaceShowRatio_Mean1 = []
    FaceShowRatio_Mean2 = []
    bIsIllegal = []
    bComparisonMatrix = []
    if not camera_open_success:
        print u"摄像头未打开，请重启摄像头！"
    else:
        FramesCNT = 0
        FaceShowRatio_Res1 = []
        FaceShowRatio_Res2 = []

        imgV1 = []
        FaceShapeV1 = []

        imgV2 = []
        FaceShapeV2 = []

        firstflag = 1
        # while camera_open_success:
        while FramesCNT < video_frames:
            ret, cv_img = camera.read()
            # frame = cv_img
            # 【高y，宽x】
            frame1 = cv_img[0:240, :]
            frame2 = cv_img[240:480, :]
            if not ret:
                print u"未正常获取视频，请检查摄像头！"
                break
            else:
                if intervalTime == 0:
                    # 检测人脸
                    bFaceDetected1, Faceshapes1 = Dlib_detectFace(frame1)
                    bFaceDetected2, Faceshapes2 = Dlib_detectFace(frame2)
                    FaceShowRatio_Res1.append(bFaceDetected1)
                    FaceShowRatio_Res2.append(bFaceDetected2)

                    # 识别人脸
                    if bFaceDetected1:
                        imgV1.append(frame1)
                        FaceShapeV1.append(Faceshapes1[0])
                    if bFaceDetected2:
                        imgV2.append(frame2)
                        FaceShapeV2.append(Faceshapes2[0])

                    if len(imgV1) > 20 and len(imgV2) > 20 and firstflag:
                        imgV = [imgV1[10], imgV2[10]]
                        FaceShapeV = [FaceShapeV1[10], FaceShapeV2[10]]
                        bIsIllegal, bComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                        if bIsIllegal:
                            cv2.imwrite(fileName + '_Detected_Picture_1.jpg', imgV[0])
                            cv2.imwrite(fileName + '_Detected_Picture_2.jpg', imgV[1])
                        firstflag = 0

                else:
                    intervalFrames = video_fps * intervalTime
                    if FramesCNT % intervalFrames == 0:
                        # 检测人脸
                        bFaceDetected1, Faceshapes1 = Dlib_detectFace(frame1)
                        bFaceDetected2, Faceshapes2 = Dlib_detectFace(frame2)
                        FaceShowRatio_Res1.append(bFaceDetected1)
                        FaceShowRatio_Res2.append(bFaceDetected2)

                        # 识别人脸
                        if bFaceDetected1:
                            imgV1.append(frame1)
                            FaceShapeV1.append(Faceshapes1[0])
                        if bFaceDetected2:
                            imgV2.append(frame2)
                            FaceShapeV2.append(Faceshapes2[0])

                        if len(imgV1) > 100 and len(imgV2) > 0 and firstflag:
                            imgV = [imgV1[10], imgV1[90]]
                            FaceShapeV = [FaceShapeV1[10], FaceShapeV1[90]]
                            bIsIllegal, bComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                            if bIsIllegal:
                                cv2.imwrite(fileName + '_Detected_Picture_1.jpg', imgV[0])
                                cv2.imwrite(fileName + '_Detected_Picture_2.jpg', imgV[1])
                            firstflag = 0

                FramesCNT += 1
                if FramesCNT % video_frames_PrintPercent == 0:
                    print str(round(float(FramesCNT) / video_frames, 2) * 100) + "%"
                # cv2.imshow("Demo", frame)
                # if cv2.waitKey(1) & 0xFF == 27:
                #     break

        FaceShowRatio_Mean1 = float(np.sum(FaceShowRatio_Res1)) / len(FaceShowRatio_Res1)
        FaceShowRatio_Mean2 = float(np.sum(FaceShowRatio_Res2)) / len(FaceShowRatio_Res2)
    return FaceShowRatio_Mean1, FaceShowRatio_Mean2, bIsIllegal, bComparisonMatrix


if __name__ == '__main__':
    VideoPath = "TestVideo/video_52158706.mp4"
    # VideoPath = "TestVideo/f3.mp4"

    print FaceDR(VideoPath, 3)
