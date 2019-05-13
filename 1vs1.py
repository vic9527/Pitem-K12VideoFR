# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 14:51
# @Author  : Vic Woo
# @Email   : vic.woo@vip.163.com
# @File    : 1vs1.py
# @Software: PyCharm


import os, sys
current_path = os.getcwd()  # 获取当前路径
PyLibs_path = current_path + "/PyLibs"
sys.path.append(PyLibs_path)
import numpy as np

import random
import string

import PyLibs.cv343.cv2 as cv2
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

cv_interpolation_mode = cv2.INTER_CUBIC
cv_resize_value = 2


# 从图像中检测人脸部分
def Dlib_detectFace(img):
    shapes = []
    isFaceDetected = 0
    # 对目标图像进行采样，貌似是第二个参数越大识别精度越高。
    faces = detector(img, 0)
    # for i, d in enumerate(faces):
    #     left, right, top, bottom = d.left(), d.right(), d.top(), d.bottom()
    #     cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
    faceNum = len(faces)
    if faceNum > 0:
        isFaceDetected = 1

    # 对检测出的模型进行计算
    for i, d in enumerate(faces):
        shape = predictor(img, d)
        shapes.append(shape)

    # 返回检测结果
    return isFaceDetected, faceNum, shapes


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
    for i in range(len(npVectors1)):
        npVectorsdiffdiff += (npVectors1[i] - npVectors2[i]) ** 2
    npVectorsdiffdiff = np.sqrt(npVectorsdiffdiff)
    # print npVectorsdiffdiff
    if npVectorsdiffdiff < 0.3:
        IsOneFace = 1
    return IsOneFace


def PairWiseComparison(imgV, FaceShapeV):
    lenNum = len(imgV)
    IsIllegal, ComparisonMatrix = 0, np.eye(lenNum, k=0)
    ComparisonResults = []
    if lenNum >= 2:
        for i in range(0, lenNum):
            for j in range(i + 1, lenNum):
                if Dlib_recogFace(imgV[i], FaceShapeV[i], imgV[j], FaceShapeV[j]) == 1:
                    ComparisonMatrix[i][j] = 1
                    ComparisonResults.append([i + 1, j + 1])

    # print ComparisonResults

    lenCR = len(ComparisonResults)
    if lenCR > 0:
        IsIllegal = 1

    for i in range(0, lenCR - 1):
        for j in range(i + 1, lenCR):
            CRtmp = [val for val in ComparisonResults[i] if val in ComparisonResults[j]]
            if CRtmp:
                ComparisonResults[i] = list(set(set(ComparisonResults[i]).union(set(ComparisonResults[j]))))
                ComparisonResults[j] = []
    ComparisonResults_New = [k for k in ComparisonResults if k != []]

    return IsIllegal, ComparisonResults_New


# 1 个学生 --- 320*240(320*480) --- 老师 320*240
# 2 个学生 --- 160*120(320*360) --- 老师 320*240
# 3-4个学生 --- 160*120(320*480) --- 老师 320*240
# 5-6个学生 --- 160*120(320*600) --- 老师 320*240


def QingAI_StuDetect(VideoPath, intervalTime):

    fileName = VideoPath.split("/")[-1]
    camera = cv2.VideoCapture(VideoPath)
    camera_open_success = camera.isOpened()
    video_fps = round(camera.get(5))
    video_frames = round(camera.get(7))
    video_frames_PrintPercent = int(video_frames / 100)
    # print u'摄像头FPS:%d, FRAMES:%d!' % (video_fps, video_frames)
    # 从a-zA-Z0-9生成指定数量的随机字符：
    # ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    
    if not camera_open_success:
        print fileName + "-->" + u"读取视频出错！"

    else:
        FramesCNT = 0
        firstflag = 1

        IsItIllegal = 0
        CaseReason = []
        Stu_FaceShowRatio = 0
        Stu_faceNum = 0
        Stu_imgV = None
        Stu_FaceShapeV = None

        Stu_FaceDetected = []

        while FramesCNT < video_frames:
            ret, cv_img = camera.read()

            if not ret:
                pass
                # print fileName + "-->" + u"视频当前帧信息损坏！"

            else:
                Stu_Img = cv_img[240:480, :]

                if intervalTime == 0:
                    # 检测人脸
                    Stu_isFaceDetected, Stu_faceNum, Stu_Faceshapes = Dlib_detectFace(Stu_Img)
                    Stu_FaceDetected.append(Stu_isFaceDetected)
                    if Stu_isFaceDetected:
                        Stu_imgV = Stu_Img
                        Stu_FaceShapeV = Stu_Faceshapes[0]
                    if Stu_faceNum > 1 and firstflag:
                        firstflag = 0
                        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                        cv2.imwrite(fileName + '_Detected_Picture_'+ ran_str +'.jpg', Stu_Img)
                        IsItIllegal = 1
                        CaseReason.append(fileName + "-->" + u"1对1课程出现人脸数量超标！")
                        # 识别人脸


                else:
                    intervalFrames = video_fps * intervalTime
                    if FramesCNT % intervalFrames == 0:
                        # 检测人脸
                        Stu_isFaceDetected, Stu_faceNum, Stu_Faceshapes = Dlib_detectFace(Stu_Img)
                        Stu_FaceDetected.append(Stu_isFaceDetected)
                        if Stu_isFaceDetected:
                            Stu_imgV = Stu_Img
                            Stu_FaceShapeV = Stu_Faceshapes[0]
                        if Stu_faceNum > 1 and firstflag:
                            firstflag = 0
                            ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                            cv2.imwrite(fileName + '_Detected_Picture_'+ ran_str +'.jpg', Stu_Img)
                            IsItIllegal = 1
                            CaseReason.append(fileName + "-->" + u"1对1课程出现人脸数量超标！")
                            # 识别人脸


            FramesCNT += 1
            if FramesCNT % video_frames_PrintPercent == 0:
                print str(round(float(FramesCNT) / video_frames, 2) * 100) + "%"
            # cv2.imshow("Demo", frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

        if len(Stu_FaceDetected) > 0:
            Stu_FaceShowRatio = float(np.sum(Stu_FaceDetected)) / len(Stu_FaceDetected)
            if Stu_FaceShowRatio < 0.3:
                IsItIllegal = 1
                if Stu_FaceShowRatio == 0:
                    CaseReason.append(fileName + "-->" + u"1对1课程学生全程未检测到脸！")
                else:
                    CaseReason.append(fileName + "-->" + u"1对1课程学生露脸时长过低！")
        # else:
        #     pass
        #     # print fileName + "-->" + u"视频所有帧信息损坏！"


    return IsItIllegal, CaseReason, Stu_faceNum, Stu_FaceShowRatio, Stu_imgV, Stu_FaceShapeV 


def processVideo(VideoPath1, VideoPath2, intervalTime=5):
    bIsItIllegal = []
    bCaseReason = None
    bRecogComparisonMatrix = None
    bStu_FaceShowRatio = []

    fileName1 = VideoPath1.split("/")[-1]
    fileName2 = VideoPath2.split("/")[-1]

    bIsItIllegal1, CaseReason1, _, Stu1_FaceShowRatio, Stu1_imgV, Stu1_FaceShapeV = QingAI_StuDetect(VideoPath1, intervalTime)
    bIsItIllegal2, CaseReason2, _, Stu2_FaceShowRatio, Stu2_imgV, Stu2_FaceShapeV = QingAI_StuDetect(VideoPath2, intervalTime)

    # print Stu1_imgV
    # print Stu2_imgV

    if Stu1_imgV is None:
        pass
        # print fileName1 + "-->" + u"未获取人脸图片信息！"

    elif Stu2_imgV is None:
        pass
        # print fileName2 + "-->" + u"未获取人脸图片信息！"

    else:
        imgV = [Stu1_imgV, Stu2_imgV]
        FaceShapeV = [Stu1_FaceShapeV, Stu2_FaceShapeV]
        bRecogIsItIllegal, bRecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
        if bRecogIsItIllegal:
            ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
            cv2.imwrite(fileName1 + '_Recognized_Picture_'+ ran_str +'.jpg', imgV[0])
            ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
            cv2.imwrite(fileName2 + '_Recognized_Picture_'+ ran_str +'.jpg', imgV[1])

    bIsItIllegal.append(bIsItIllegal1)
    bIsItIllegal.append(bIsItIllegal2)
    # bCaseReason.append(CaseReason1)
    # bCaseReason.append(CaseReason2)
    bCaseReason = CaseReason1 + CaseReason2
    bStu_FaceShowRatio.append(Stu1_FaceShowRatio)
    bStu_FaceShowRatio.append(Stu2_FaceShowRatio)

    if not bCaseReason:
        bCaseReason = u"此视频无异常！"

    if not bRecogComparisonMatrix:
        bRecogComparisonMatrix = "No Similarity!"
    else:
        # bIsItIllegal = 1
        bRecogComparisonMatrix = "Similarity Detected!"

    return bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio


if __name__ == '__main__':
    VideoPath1 = "TestVideo/480_1vs1.mp4"
    VideoPath2 = "TestVideo/480_1vs1.mp4"

    # processVideo(VideoPath1, VideoPath2, intervalTime=5) 程序接口，VideoPath1，VideoPath2是待检测视频路径，intervalTime是间隔检测时长（0表示全帧检测，正整数表示每隔几秒检测一帧。）
    
    #返回参数说明，总共返回四个参数，每个参数说明如下：
    # bbIsItIllegal 返回当前每个被检测视频中学生界面是否违规，0表示不违规，1表示违规。
    # bbCaseReason 返回当前每个被检测视频违规的具体原因。
    # bbRecogComparisonMatrix 返回当前每个被检测视频中每个学生界面是否含有相似的人脸。
    # bbStu_FaceShowRatio 返回当前每个被检测视频中学生界面的人脸出现占比。
    bbIsItIllegal, bbCaseReason, bbRecogComparisonMatrix, bbStu_FaceShowRatio = processVideo(VideoPath1, VideoPath2, 10)


    print bbIsItIllegal
    print u"——————————"
    for i in bbCaseReason:
        print i
    print u"——————————"
    print bbRecogComparisonMatrix
    print bbStu_FaceShowRatio
