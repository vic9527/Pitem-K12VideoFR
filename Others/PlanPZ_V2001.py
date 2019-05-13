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
    for i in xrange(len(npVectors1)):
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


def RecogClass1vs1(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime):
    CaseReason = []
    RecogComparisonMatrix = None
    Stu_FaceShowRatio = []
    Stu1_FaceDetected = []
    IsItIllegal = 0
    FramesCNT = 0
    firstflag = 1
    while FramesCNT < video_frames:
        ret, cv_img = camera.read()
        if not ret:
            print u"未正确读取视频信息！"
            break
        else:
            Stu1_Img = cv_img[240:480, :]

            if intervalTime == 0:
                # 检测人脸
                Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                Stu1_FaceDetected.append(Stu1_isFaceDetected)
                if Stu1_faceNum > 1 and firstflag:
                    firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对1课程出现人脸数量超标！")
                    # 识别人脸


            else:
                intervalFrames = video_fps * intervalTime
                if FramesCNT % intervalFrames == 0:
                    # 检测人脸
                    Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                    Stu1_FaceDetected.append(Stu1_isFaceDetected)
                    if Stu1_faceNum > 1 and firstflag:
                        firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对1课程出现人脸数量超标！")
                        # 识别人脸


            FramesCNT += 1
            if FramesCNT % video_frames_PrintPercent == 0:
                print str(round(float(FramesCNT) / video_frames, 2) * 100) + "%"
            # cv2.imshow("Demo", frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    Stu_FaceShowRatio = float(np.sum(Stu1_FaceDetected)) / len(Stu1_FaceDetected)
    if Stu_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu_FaceShowRatio == 0:
            CaseReason.append(u"1对1课程学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对1课程学生露脸时长过低！")

    if not CaseReason:
        CaseReason = u"此视频无异常！"
    if not RecogComparisonMatrix:
        RecogComparisonMatrix = "No Similar!"

    return IsItIllegal, CaseReason, RecogComparisonMatrix, Stu_FaceShowRatio

def RecogClass1vs2(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime):
    CaseReason = []
    RecogComparisonMatrix = None
    Stu_FaceShowRatio = []
    Stu1_FaceDetected = []
    Stu2_FaceDetected = []
    Stu1_imgV = []
    Stu2_imgV = []
    Stu1_FaceShapeV = []
    Stu2_FaceShapeV = []
    IsItIllegal = 0
    FramesCNT = 0
    Stu1_firstflag = 1
    Stu2_firstflag = 1

    Recog_flag = 0

    while FramesCNT < video_frames:

        ret, cv_img = camera.read()
        if not ret:
            print u"未正确读取视频信息！"
            break
        else:
            Stu1_Img = cv_img[240:360, 0:160]
            Stu2_Img = cv_img[240:360, 160:320]
            Stu1_Img = cv2.resize(Stu1_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value, interpolation=cv_interpolation_mode)
            Stu2_Img = cv2.resize(Stu2_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value, interpolation=cv_interpolation_mode)


            if intervalTime == 0:
                if FramesCNT == (video_frames - 1):
                    Recog_flag = 1
                # cv2.imshow("Demo1", Stu1_Img)
                # cv2.imshow("Demo2", Stu2_Img)
                # if cv2.waitKey(1) & 0xFF == 27:
                #     break
                # 检测人脸
                Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                Stu1_FaceDetected.append(Stu1_isFaceDetected)
                if Stu1_isFaceDetected:
                    # Stu1_imgV.append(Stu1_Img)
                    # Stu1_FaceShapeV.append(Stu1_Faceshapes[0])
                    Stu1_imgV = Stu1_Img
                    Stu1_FaceShapeV = Stu1_Faceshapes[0]
                Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                Stu2_FaceDetected.append(Stu2_isFaceDetected)
                if Stu2_isFaceDetected:
                    Stu2_imgV = Stu2_Img
                    Stu2_FaceShapeV = Stu2_Faceshapes[0]
                if Stu1_faceNum > 1 and Stu1_firstflag:
                    Stu1_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对2课程中[No.1]学生视频出现人脸数量超标！")
                if Stu2_faceNum > 1 and Stu2_firstflag:
                    Stu2_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对2课程中[No.2]学生视频出现人脸数量超标！")

                # 识别人脸
                if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and Recog_flag:
                    Recog_flag = 0
                    imgV = [Stu1_imgV, Stu2_imgV]
                    FaceShapeV = [Stu1_FaceShapeV, Stu2_FaceShapeV]
                    RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                    if RecogIsIllegal:
                        cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                        cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                        # print RecogComparisonMatrix


            else:
                intervalFrames = video_fps * intervalTime
                if FramesCNT % intervalFrames == 0:
                    if (FramesCNT + intervalFrames) > video_frames:
                        Recog_flag = 1
                    # cv2.imshow("Demo1", Stu1_Img)
                    # cv2.imshow("Demo2", Stu2_Img)
                    # if cv2.waitKey(1) & 0xFF == 27:
                    #     break
                    # 检测人脸
                    Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                    Stu1_FaceDetected.append(Stu1_isFaceDetected)
                    if Stu1_isFaceDetected:
                        Stu1_imgV = Stu1_Img
                        Stu1_FaceShapeV = Stu1_Faceshapes[0]
                    Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                    Stu2_FaceDetected.append(Stu2_isFaceDetected)
                    if Stu2_isFaceDetected:
                        Stu2_imgV = Stu2_Img
                        Stu2_FaceShapeV = Stu2_Faceshapes[0]
                    if Stu1_faceNum > 1 and Stu1_firstflag:
                        Stu1_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对2课程中[No.1]学生视频出现人脸数量超标！")
                    if Stu2_faceNum > 1 and Stu2_firstflag:
                        Stu2_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对2课程中[No.2]学生视频出现人脸数量超标！")

                        # 识别人脸
                        if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and Recog_flag:
                            Recog_flag = 0
                            imgV = [Stu1_imgV, Stu2_imgV]
                            FaceShapeV = [Stu1_FaceShapeV, Stu2_FaceShapeV]
                            RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                            if RecogIsIllegal:
                                cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                                cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                                # print RecogComparisonMatrix

            FramesCNT += 1
            if FramesCNT % video_frames_PrintPercent == 0:
                print str(round(float(FramesCNT) / video_frames, 2) * 100) + "%"
            # cv2.imshow("Demo1", Stu1_Img)
            # cv2.imshow("Demo2", Stu2_Img)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    Stu1_FaceShowRatio = float(np.sum(Stu1_FaceDetected)) / len(Stu1_FaceDetected)
    if Stu1_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu1_FaceShowRatio == 0:
            CaseReason.append(u"1对2课程中[No.1]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对2课程中[No.1]学生露脸时长过低！")
    Stu2_FaceShowRatio = float(np.sum(Stu2_FaceDetected)) / len(Stu2_FaceDetected)
    if Stu2_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu2_FaceShowRatio == 0:
            CaseReason.append(u"1对2课程中[No.2]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对2课程中[No.2]学生露脸时长过低！")

    Stu_FaceShowRatio = [Stu1_FaceShowRatio, Stu2_FaceShowRatio]

    if not CaseReason:
        CaseReason = u"此视频无异常！"
    if not RecogComparisonMatrix:
        RecogComparisonMatrix = "No Similar!"

    return IsItIllegal, CaseReason, RecogComparisonMatrix, Stu_FaceShowRatio


def RecogClass1vs3(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime):
    CaseReason = []
    RecogComparisonMatrix = None
    Stu_FaceShowRatio = []
    Stu1_FaceDetected = []
    Stu2_FaceDetected = []
    Stu3_FaceDetected = []
    Stu1_imgV = []
    Stu2_imgV = []
    Stu3_imgV = []
    Stu1_FaceShapeV = []
    Stu2_FaceShapeV = []
    Stu3_FaceShapeV = []
    IsItIllegal = 0
    FramesCNT = 0
    Stu1_firstflag = 1
    Stu2_firstflag = 1
    Stu3_firstflag = 1
    Recog_flag = 0
    while FramesCNT < video_frames:
        ret, cv_img = camera.read()
        if not ret:
            print u"未正确读取视频信息！"
            break
        else:
            Stu1_Img = cv_img[240:360, 0:160]
            Stu2_Img = cv_img[240:360, 160:320]
            Stu3_Img = cv_img[360:480, 0:160]
            Stu1_Img = cv2.resize(Stu1_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu2_Img = cv2.resize(Stu2_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu3_Img = cv2.resize(Stu3_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)


            if intervalTime == 0:
                if FramesCNT == (video_frames - 1):
                    Recog_flag = 1
                # 检测人脸
                Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                Stu1_FaceDetected.append(Stu1_isFaceDetected)
                if Stu1_isFaceDetected:
                    Stu1_imgV = Stu1_Img
                    Stu1_FaceShapeV = Stu1_Faceshapes[0]

                Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                Stu2_FaceDetected.append(Stu2_isFaceDetected)
                if Stu2_isFaceDetected:
                    Stu2_imgV = Stu2_Img
                    Stu2_FaceShapeV = Stu2_Faceshapes[0]

                Stu3_isFaceDetected, Stu3_faceNum, Stu3_Faceshapes = Dlib_detectFace(Stu3_Img)
                Stu3_FaceDetected.append(Stu3_isFaceDetected)
                if Stu3_isFaceDetected:
                    Stu3_imgV = Stu3_Img
                    Stu3_FaceShapeV = Stu3_Faceshapes[0]

                if Stu1_faceNum > 1 and Stu1_firstflag:
                    Stu1_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对3课程中[No.1]学生视频出现人脸数量超标！")
                if Stu2_faceNum > 1 and Stu2_firstflag:
                    Stu2_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对3课程中[No.2]学生视频出现人脸数量超标！")
                if Stu3_faceNum > 1 and Stu3_firstflag:
                    Stu3_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_3.jpg', Stu3_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对3课程中[No.3]学生视频出现人脸数量超标！")

                # 识别人脸
                if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and len(Stu3_imgV) > 0 and Recog_flag:
                    Recog_flag = 0
                    imgV = [Stu1_imgV, Stu2_imgV, Stu3_imgV]
                    FaceShapeV = [Stu1_FaceShapeV, Stu2_FaceShapeV, Stu3_FaceShapeV]
                    RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                    if RecogIsIllegal:
                        cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                        cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                        cv2.imwrite(fileName + '_Recognized_Picture_3.jpg', imgV[2])
                        # print RecogComparisonMatrix


            else:
                intervalFrames = video_fps * intervalTime
                if FramesCNT % intervalFrames == 0:
                    if (FramesCNT + intervalFrames) > video_frames:
                        Recog_flag = 1
                    # 检测人脸
                    Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                    Stu1_FaceDetected.append(Stu1_isFaceDetected)
                    if Stu1_isFaceDetected:
                        Stu1_imgV = Stu1_Img
                        Stu1_FaceShapeV = Stu1_Faceshapes[0]

                    Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                    Stu2_FaceDetected.append(Stu2_isFaceDetected)
                    if Stu2_isFaceDetected:
                        Stu2_imgV = Stu2_Img
                        Stu2_FaceShapeV = Stu2_Faceshapes[0]

                    Stu3_isFaceDetected, Stu3_faceNum, Stu3_Faceshapes = Dlib_detectFace(Stu3_Img)
                    Stu3_FaceDetected.append(Stu3_isFaceDetected)
                    if Stu3_isFaceDetected:
                        Stu3_imgV = Stu3_Img
                        Stu3_FaceShapeV = Stu3_Faceshapes[0]

                    if Stu1_faceNum > 1 and Stu1_firstflag:
                        Stu1_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对3课程中[No.1]学生视频出现人脸数量超标！")
                    if Stu2_faceNum > 1 and Stu2_firstflag:
                        Stu2_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对3课程中[No.2]学生视频出现人脸数量超标！")
                    if Stu3_faceNum > 1 and Stu3_firstflag:
                        Stu3_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_3.jpg', Stu3_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对3课程中[No.3]学生视频出现人脸数量超标！")

                    # 识别人脸
                    if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and len(Stu3_imgV) > 0 and Recog_flag:
                        Recog_flag = 0
                        imgV = [Stu1_imgV, Stu2_imgV, Stu3_imgV]
                        FaceShapeV = [Stu1_FaceShapeV, Stu2_FaceShapeV, Stu3_FaceShapeV]
                        RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                        if RecogIsIllegal:
                            cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                            cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                            cv2.imwrite(fileName + '_Recognized_Picture_3.jpg', imgV[2])
                            # print RecogComparisonMatrix

            FramesCNT += 1
            if FramesCNT % video_frames_PrintPercent == 0:
                print str(round(float(FramesCNT) / video_frames, 2) * 100) + "%"
            # cv2.imshow("Demo", frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    Stu1_FaceShowRatio = float(np.sum(Stu1_FaceDetected)) / len(Stu1_FaceDetected)
    if Stu1_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu1_FaceShowRatio == 0:
            CaseReason.append(u"1对3课程中[No.1]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对3课程中[No.1]学生露脸时长过低！")
    Stu2_FaceShowRatio = float(np.sum(Stu2_FaceDetected)) / len(Stu2_FaceDetected)
    if Stu2_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu2_FaceShowRatio == 0:
            CaseReason.append(u"1对3课程中[No.2]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对3课程中[No.2]学生露脸时长过低！")
    Stu3_FaceShowRatio = float(np.sum(Stu3_FaceDetected)) / len(Stu3_FaceDetected)
    if Stu3_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu3_FaceShowRatio == 0:
            CaseReason.append(u"1对3课程中[No.3]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对3课程中[No.3]学生露脸时长过低！")

    Stu_FaceShowRatio = [Stu1_FaceShowRatio, Stu2_FaceShowRatio, Stu3_FaceShowRatio]

    if not CaseReason:
        CaseReason = u"此视频无异常！"
    if not RecogComparisonMatrix:
        RecogComparisonMatrix = "No Similar!"

    return IsItIllegal, CaseReason, RecogComparisonMatrix, Stu_FaceShowRatio


def RecogClass1vs4(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime):
    CaseReason = []
    RecogComparisonMatrix = None
    Stu_FaceShowRatio = []
    Stu1_FaceDetected = []
    Stu2_FaceDetected = []
    Stu3_FaceDetected = []
    Stu4_FaceDetected = []
    Stu1_imgV = []
    Stu2_imgV = []
    Stu3_imgV = []
    Stu4_imgV = []
    Stu1_FaceShapeV = []
    Stu2_FaceShapeV = []
    Stu3_FaceShapeV = []
    Stu4_FaceShapeV = []
    IsItIllegal = 0
    FramesCNT = 0
    Stu1_firstflag = 1
    Stu2_firstflag = 1
    Stu3_firstflag = 1
    Stu4_firstflag = 1
    Recog_flag = 0
    while FramesCNT < video_frames:
        ret, cv_img = camera.read()
        if not ret:
            print u"未正确读取视频信息！"
            break
        else:
            Stu1_Img = cv_img[240:360, 0:160]
            Stu2_Img = cv_img[240:360, 160:320]
            Stu3_Img = cv_img[360:480, 0:160]
            Stu4_Img = cv_img[360:480, 160:320]
            Stu1_Img = cv2.resize(Stu1_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu2_Img = cv2.resize(Stu2_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu3_Img = cv2.resize(Stu3_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu4_Img = cv2.resize(Stu4_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)

            if intervalTime == 0:
                if FramesCNT == (video_frames - 1):
                    Recog_flag = 1
                # 检测人脸
                Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                Stu1_FaceDetected.append(Stu1_isFaceDetected)
                if Stu1_isFaceDetected:
                    Stu1_imgV = Stu1_Img
                    Stu1_FaceShapeV = Stu1_Faceshapes[0]

                Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                Stu2_FaceDetected.append(Stu2_isFaceDetected)
                if Stu2_isFaceDetected:
                    Stu2_imgV = Stu2_Img
                    Stu2_FaceShapeV = Stu2_Faceshapes[0]

                Stu3_isFaceDetected, Stu3_faceNum, Stu3_Faceshapes = Dlib_detectFace(Stu3_Img)
                Stu3_FaceDetected.append(Stu3_isFaceDetected)
                if Stu3_isFaceDetected:
                    Stu3_imgV = Stu3_Img
                    Stu3_FaceShapeV = Stu3_Faceshapes[0]

                Stu4_isFaceDetected, Stu4_faceNum, Stu4_Faceshapes = Dlib_detectFace(Stu4_Img)
                Stu4_FaceDetected.append(Stu4_isFaceDetected)
                if Stu4_isFaceDetected:
                    Stu4_imgV = Stu4_Img
                    Stu4_FaceShapeV = Stu4_Faceshapes[0]

                if Stu1_faceNum > 1 and Stu1_firstflag:
                    Stu1_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对4课程中[No.1]学生视频出现人脸数量超标！")
                if Stu2_faceNum > 1 and Stu2_firstflag:
                    Stu2_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对4课程中[No.2]学生视频出现人脸数量超标！")
                if Stu3_faceNum > 1 and Stu3_firstflag:
                    Stu3_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_3.jpg', Stu3_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对4课程中[No.3]学生视频出现人脸数量超标！")
                if Stu4_faceNum > 1 and Stu4_firstflag:
                    Stu4_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_4.jpg', Stu4_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对4课程中[No.4]学生视频出现人脸数量超标！")

                # 识别人脸
                if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and len(Stu3_imgV) > 0 and len(Stu4_imgV) > 0 \
                        and Recog_flag:
                    Recog_flag = 0
                    imgV = [Stu1_imgV, Stu2_imgV, Stu3_imgV, Stu4_imgV]
                    FaceShapeV = [Stu1_FaceShapeV, Stu2_FaceShapeV, Stu3_FaceShapeV, Stu4_FaceShapeV]
                    RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                    if RecogIsIllegal:
                        cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                        cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                        cv2.imwrite(fileName + '_Recognized_Picture_3.jpg', imgV[2])
                        cv2.imwrite(fileName + '_Recognized_Picture_4.jpg', imgV[3])
                        # print RecogComparisonMatrix

            else:
                intervalFrames = video_fps * intervalTime
                if FramesCNT % intervalFrames == 0:
                    if (FramesCNT + intervalFrames) > video_frames:
                        Recog_flag = 1
                    # 检测人脸
                    Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                    Stu1_FaceDetected.append(Stu1_isFaceDetected)
                    if Stu1_isFaceDetected:
                        Stu1_imgV = Stu1_Img
                        Stu1_FaceShapeV = Stu1_Faceshapes[0]

                    Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                    Stu2_FaceDetected.append(Stu2_isFaceDetected)
                    if Stu2_isFaceDetected:
                        Stu2_imgV = Stu2_Img
                        Stu2_FaceShapeV = Stu2_Faceshapes[0]

                    Stu3_isFaceDetected, Stu3_faceNum, Stu3_Faceshapes = Dlib_detectFace(Stu3_Img)
                    Stu3_FaceDetected.append(Stu3_isFaceDetected)
                    if Stu3_isFaceDetected:
                        Stu3_imgV = Stu3_Img
                        Stu3_FaceShapeV = Stu3_Faceshapes[0]

                    Stu4_isFaceDetected, Stu4_faceNum, Stu4_Faceshapes = Dlib_detectFace(Stu4_Img)
                    Stu4_FaceDetected.append(Stu4_isFaceDetected)
                    if Stu4_isFaceDetected:
                        Stu4_imgV = Stu4_Img
                        Stu4_FaceShapeV = Stu4_Faceshapes[0]

                    if Stu1_faceNum > 1 and Stu1_firstflag:
                        Stu1_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对4课程中[No.1]学生视频出现人脸数量超标！")
                    if Stu2_faceNum > 1 and Stu2_firstflag:
                        Stu2_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对4课程中[No.2]学生视频出现人脸数量超标！")
                    if Stu3_faceNum > 1 and Stu3_firstflag:
                        Stu3_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_3.jpg', Stu3_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对4课程中[No.3]学生视频出现人脸数量超标！")
                    if Stu4_faceNum > 1 and Stu4_firstflag:
                        Stu4_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_4.jpg', Stu4_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对4课程中[No.4]学生视频出现人脸数量超标！")

                    # 识别人脸
                    if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and len(Stu3_imgV) > 0 and len(Stu4_imgV) > 0 \
                            and Recog_flag:
                        Recog_flag = 0
                        imgV = [Stu1_imgV[0], Stu2_imgV[0], Stu3_imgV[0], Stu4_imgV[0]]
                        FaceShapeV = [Stu1_FaceShapeV[0], Stu2_FaceShapeV[0], Stu3_FaceShapeV[0], Stu4_FaceShapeV[0]]
                        RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                        if RecogIsIllegal:
                            cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                            cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                            cv2.imwrite(fileName + '_Recognized_Picture_3.jpg', imgV[2])
                            cv2.imwrite(fileName + '_Recognized_Picture_4.jpg', imgV[3])
                            # print RecogComparisonMatrix

            FramesCNT += 1
            if FramesCNT % video_frames_PrintPercent == 0:
                print str(round(float(FramesCNT) / video_frames, 2) * 100) + "%"
            # cv2.imshow("Demo", frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    Stu1_FaceShowRatio = float(np.sum(Stu1_FaceDetected)) / len(Stu1_FaceDetected)
    if Stu1_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu1_FaceShowRatio == 0:
            CaseReason.append(u"1对4课程中[No.1]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对4课程中[No.1]学生露脸时长过低！")
    Stu2_FaceShowRatio = float(np.sum(Stu2_FaceDetected)) / len(Stu2_FaceDetected)
    if Stu2_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu2_FaceShowRatio == 0:
            CaseReason.append(u"1对4课程中[No.2]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对4课程中[No.2]学生露脸时长过低！")
    Stu3_FaceShowRatio = float(np.sum(Stu3_FaceDetected)) / len(Stu3_FaceDetected)
    if Stu3_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu3_FaceShowRatio == 0:
            CaseReason.append(u"1对4课程中[No.3]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对4课程中[No.3]学生露脸时长过低！")
    Stu4_FaceShowRatio = float(np.sum(Stu4_FaceDetected)) / len(Stu4_FaceDetected)
    if Stu4_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu4_FaceShowRatio == 0:
            CaseReason.append(u"1对4课程中[No.4]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对4课程中[No.4]学生露脸时长过低！")

    Stu_FaceShowRatio = [Stu1_FaceShowRatio, Stu2_FaceShowRatio, Stu3_FaceShowRatio, Stu4_FaceShowRatio]

    if not CaseReason:
        CaseReason = u"此视频无异常！"
    if not RecogComparisonMatrix:
        RecogComparisonMatrix = "No Similar!"

    return IsItIllegal, CaseReason, RecogComparisonMatrix, Stu_FaceShowRatio


def RecogClass1vs5(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime):
    CaseReason = []
    RecogComparisonMatrix = None
    Stu_FaceShowRatio = []
    Stu1_FaceDetected = []
    Stu2_FaceDetected = []
    Stu3_FaceDetected = []
    Stu4_FaceDetected = []
    Stu5_FaceDetected = []
    Stu1_imgV = []
    Stu2_imgV = []
    Stu3_imgV = []
    Stu4_imgV = []
    Stu5_imgV = []
    Stu1_FaceShapeV = []
    Stu2_FaceShapeV = []
    Stu3_FaceShapeV = []
    Stu4_FaceShapeV = []
    Stu5_FaceShapeV = []
    IsItIllegal = 0
    FramesCNT = 0
    Stu1_firstflag = 1
    Stu2_firstflag = 1
    Stu3_firstflag = 1
    Stu4_firstflag = 1
    Stu5_firstflag = 1
    Recog_flag = 0
    while FramesCNT < video_frames:
        ret, cv_img = camera.read()
        if not ret:
            print u"未正确读取视频信息！"
            break
        else:
            Stu1_Img = cv_img[240:360, 0:160]
            Stu2_Img = cv_img[240:360, 160:320]
            Stu3_Img = cv_img[360:480, 0:160]
            Stu4_Img = cv_img[360:480, 160:320]
            Stu5_Img = cv_img[480:600, 0:160]
            Stu1_Img = cv2.resize(Stu1_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu2_Img = cv2.resize(Stu2_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu3_Img = cv2.resize(Stu3_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu4_Img = cv2.resize(Stu4_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu5_Img = cv2.resize(Stu5_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)

            if intervalTime == 0:
                if FramesCNT == (video_frames - 1):
                    Recog_flag = 1
                # 检测人脸
                Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                Stu1_FaceDetected.append(Stu1_isFaceDetected)
                if Stu1_isFaceDetected:
                    Stu1_imgV = Stu1_Img
                    Stu1_FaceShapeV = Stu1_Faceshapes[0]

                Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                Stu2_FaceDetected.append(Stu2_isFaceDetected)
                if Stu2_isFaceDetected:
                    Stu2_imgV = Stu2_Img
                    Stu2_FaceShapeV = Stu2_Faceshapes[0]

                Stu3_isFaceDetected, Stu3_faceNum, Stu3_Faceshapes = Dlib_detectFace(Stu3_Img)
                Stu3_FaceDetected.append(Stu3_isFaceDetected)
                if Stu3_isFaceDetected:
                    Stu3_imgV = Stu3_Img
                    Stu3_FaceShapeV = Stu3_Faceshapes[0]

                Stu4_isFaceDetected, Stu4_faceNum, Stu4_Faceshapes = Dlib_detectFace(Stu4_Img)
                Stu4_FaceDetected.append(Stu4_isFaceDetected)
                if Stu4_isFaceDetected:
                    Stu4_imgV = Stu4_Img
                    Stu4_FaceShapeV = Stu4_Faceshapes[0]

                Stu5_isFaceDetected, Stu5_faceNum, Stu5_Faceshapes = Dlib_detectFace(Stu5_Img)
                Stu5_FaceDetected.append(Stu5_isFaceDetected)
                if Stu5_isFaceDetected:
                    Stu5_imgV = Stu5_Img
                    Stu5_FaceShapeV = Stu5_Faceshapes[0]

                if Stu1_faceNum > 1 and Stu1_firstflag:
                    Stu1_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对5课程中[No.1]学生视频出现人脸数量超标！")
                if Stu2_faceNum > 1 and Stu2_firstflag:
                    Stu2_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对5课程中[No.2]学生视频出现人脸数量超标！")
                if Stu3_faceNum > 1 and Stu3_firstflag:
                    Stu3_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_3.jpg', Stu3_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对5课程中[No.3]学生视频出现人脸数量超标！")
                if Stu4_faceNum > 1 and Stu4_firstflag:
                    Stu4_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_4.jpg', Stu4_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对5课程中[No.4]学生视频出现人脸数量超标！")
                if Stu5_faceNum > 1 and Stu5_firstflag:
                    Stu5_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_5.jpg', Stu5_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对5课程中[No.5]学生视频出现人脸数量超标！")

                # 识别人脸
                if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and len(Stu3_imgV) > 0 and len(Stu4_imgV) > 0 \
                        and len(Stu5_imgV) > 0 and Recog_flag:
                    Recog_flag = 0
                    imgV = [Stu1_imgV[0], Stu2_imgV[0], Stu3_imgV[0], Stu4_imgV[0], Stu5_imgV[0]]
                    FaceShapeV = [Stu1_FaceShapeV[0], Stu2_FaceShapeV[0], Stu3_FaceShapeV[0], Stu4_FaceShapeV[0], Stu5_FaceShapeV[0]]
                    RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                    if RecogIsIllegal:
                        cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                        cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                        cv2.imwrite(fileName + '_Recognized_Picture_3.jpg', imgV[2])
                        cv2.imwrite(fileName + '_Recognized_Picture_4.jpg', imgV[3])
                        cv2.imwrite(fileName + '_Recognized_Picture_5.jpg', imgV[4])
                        # print RecogComparisonMatrix

            else:
                intervalFrames = video_fps * intervalTime
                if FramesCNT % intervalFrames == 0:
                    if (FramesCNT + intervalFrames) > video_frames:
                        Recog_flag = 1
                    # 检测人脸
                    Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                    Stu1_FaceDetected.append(Stu1_isFaceDetected)
                    if Stu1_isFaceDetected:
                        Stu1_imgV = Stu1_Img
                        Stu1_FaceShapeV = Stu1_Faceshapes[0]

                    Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                    Stu2_FaceDetected.append(Stu2_isFaceDetected)
                    if Stu2_isFaceDetected:
                        Stu2_imgV = Stu2_Img
                        Stu2_FaceShapeV = Stu2_Faceshapes[0]

                    Stu3_isFaceDetected, Stu3_faceNum, Stu3_Faceshapes = Dlib_detectFace(Stu3_Img)
                    Stu3_FaceDetected.append(Stu3_isFaceDetected)
                    if Stu3_isFaceDetected:
                        Stu3_imgV = Stu3_Img
                        Stu3_FaceShapeV = Stu3_Faceshapes[0]

                    Stu4_isFaceDetected, Stu4_faceNum, Stu4_Faceshapes = Dlib_detectFace(Stu4_Img)
                    Stu4_FaceDetected.append(Stu4_isFaceDetected)
                    if Stu4_isFaceDetected:
                        Stu4_imgV = Stu4_Img
                        Stu4_FaceShapeV = Stu4_Faceshapes[0]

                    Stu5_isFaceDetected, Stu5_faceNum, Stu5_Faceshapes = Dlib_detectFace(Stu5_Img)
                    Stu5_FaceDetected.append(Stu5_isFaceDetected)
                    if Stu5_isFaceDetected:
                        Stu5_imgV = Stu5_Img
                        Stu5_FaceShapeV = Stu5_Faceshapes[0]

                    if Stu1_faceNum > 1 and Stu1_firstflag:
                        Stu1_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对5课程中[No.1]学生视频出现人脸数量超标！")
                    if Stu2_faceNum > 1 and Stu2_firstflag:
                        Stu2_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对5课程中[No.2]学生视频出现人脸数量超标！")
                    if Stu3_faceNum > 1 and Stu3_firstflag:
                        Stu3_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_3.jpg', Stu3_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对5课程中[No.3]学生视频出现人脸数量超标！")
                    if Stu4_faceNum > 1 and Stu4_firstflag:
                        Stu4_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_4.jpg', Stu4_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对5课程中[No.4]学生视频出现人脸数量超标！")
                    if Stu5_faceNum > 1 and Stu5_firstflag:
                        Stu5_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_5.jpg', Stu5_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对5课程中[No.5]学生视频出现人脸数量超标！")

                    # 识别人脸
                    if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and len(Stu3_imgV) > 0 and len(Stu4_imgV) > 0 \
                            and len(Stu5_imgV) > 0 and Recog_flag:
                        Recog_flag = 0
                        imgV = [Stu1_imgV[0], Stu2_imgV[0], Stu3_imgV[0], Stu4_imgV[0], Stu5_imgV[0]]
                        FaceShapeV = [Stu1_FaceShapeV[0], Stu2_FaceShapeV[0], Stu3_FaceShapeV[0], Stu4_FaceShapeV[0],
                                      Stu5_FaceShapeV[0]]
                        RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                        if RecogIsIllegal:
                            cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                            cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                            cv2.imwrite(fileName + '_Recognized_Picture_3.jpg', imgV[2])
                            cv2.imwrite(fileName + '_Recognized_Picture_4.jpg', imgV[3])
                            cv2.imwrite(fileName + '_Recognized_Picture_5.jpg', imgV[4])
                            # print RecogComparisonMatrix

            FramesCNT += 1
            if FramesCNT % video_frames_PrintPercent == 0:
                print str(round(float(FramesCNT) / video_frames, 2) * 100) + "%"
            # cv2.imshow("Demo", frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    Stu1_FaceShowRatio = float(np.sum(Stu1_FaceDetected)) / len(Stu1_FaceDetected)
    if Stu1_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu1_FaceShowRatio == 0:
            CaseReason.append(u"1对5课程中[No.1]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对5课程中[No.1]学生露脸时长过低！")
    Stu2_FaceShowRatio = float(np.sum(Stu2_FaceDetected)) / len(Stu2_FaceDetected)
    if Stu2_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu2_FaceShowRatio == 0:
            CaseReason.append(u"1对5课程中[No.2]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对5课程中[No.2]学生露脸时长过低！")
    Stu3_FaceShowRatio = float(np.sum(Stu3_FaceDetected)) / len(Stu3_FaceDetected)
    if Stu3_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu3_FaceShowRatio == 0:
            CaseReason.append(u"1对5课程中[No.3]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对5课程中[No.3]学生露脸时长过低！")
    Stu4_FaceShowRatio = float(np.sum(Stu4_FaceDetected)) / len(Stu4_FaceDetected)
    if Stu4_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu4_FaceShowRatio == 0:
            CaseReason.append(u"1对5课程中[No.4]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对5课程中[No.4]学生露脸时长过低！")
    Stu5_FaceShowRatio = float(np.sum(Stu5_FaceDetected)) / len(Stu5_FaceDetected)
    if Stu5_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu5_FaceShowRatio == 0:
            CaseReason.append(u"1对5课程中[No.5]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对5课程中[No.5]学生露脸时长过低！")

    Stu_FaceShowRatio = [Stu1_FaceShowRatio, Stu2_FaceShowRatio, Stu3_FaceShowRatio, Stu4_FaceShowRatio,
                         Stu5_FaceShowRatio]

    if not CaseReason:
        CaseReason = u"此视频无异常！"
    if not RecogComparisonMatrix:
        RecogComparisonMatrix = "No Similar!"

    return IsItIllegal, CaseReason, RecogComparisonMatrix, Stu_FaceShowRatio


def RecogClass1vs6(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime):
    CaseReason = []
    RecogComparisonMatrix = None
    Stu_FaceShowRatio = []
    Stu1_FaceDetected = []
    Stu2_FaceDetected = []
    Stu3_FaceDetected = []
    Stu4_FaceDetected = []
    Stu5_FaceDetected = []
    Stu6_FaceDetected = []
    Stu1_imgV = []
    Stu2_imgV = []
    Stu3_imgV = []
    Stu4_imgV = []
    Stu5_imgV = []
    Stu6_imgV = []
    Stu1_FaceShapeV = []
    Stu2_FaceShapeV = []
    Stu3_FaceShapeV = []
    Stu4_FaceShapeV = []
    Stu5_FaceShapeV = []
    Stu6_FaceShapeV = []
    IsItIllegal = 0
    FramesCNT = 0
    Stu1_firstflag = 1
    Stu2_firstflag = 1
    Stu3_firstflag = 1
    Stu4_firstflag = 1
    Stu5_firstflag = 1
    Stu6_firstflag = 1
    Recog_flag = 0
    while FramesCNT < video_frames:
        ret, cv_img = camera.read()
        if not ret:
            print u"未正确读取视频信息！"
            break
        else:
            Stu1_Img = cv_img[240:360, 0:160]
            Stu2_Img = cv_img[240:360, 160:320]
            Stu3_Img = cv_img[360:480, 0:160]
            Stu4_Img = cv_img[360:480, 160:320]
            Stu5_Img = cv_img[480:600, 0:160]
            Stu6_Img = cv_img[480:600, 160:320]
            Stu1_Img = cv2.resize(Stu1_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu2_Img = cv2.resize(Stu2_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu3_Img = cv2.resize(Stu3_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu4_Img = cv2.resize(Stu4_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu5_Img = cv2.resize(Stu5_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)
            Stu6_Img = cv2.resize(Stu6_Img, (0, 0), fx=cv_resize_value, fy=cv_resize_value,
                                  interpolation=cv_interpolation_mode)

            if intervalTime == 0:
                if FramesCNT == (video_frames - 1):
                    Recog_flag = 1
                # 检测人脸
                Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                Stu1_FaceDetected.append(Stu1_isFaceDetected)
                if Stu1_isFaceDetected:
                    Stu1_imgV = Stu1_Img
                    Stu1_FaceShapeV = Stu1_Faceshapes[0]

                Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                Stu2_FaceDetected.append(Stu2_isFaceDetected)
                if Stu2_isFaceDetected:
                    Stu2_imgV = Stu2_Img
                    Stu2_FaceShapeV = Stu2_Faceshapes[0]

                Stu3_isFaceDetected, Stu3_faceNum, Stu3_Faceshapes = Dlib_detectFace(Stu3_Img)
                Stu3_FaceDetected.append(Stu3_isFaceDetected)
                if Stu3_isFaceDetected:
                    Stu3_imgV = Stu3_Img
                    Stu3_FaceShapeV = Stu3_Faceshapes[0]

                Stu4_isFaceDetected, Stu4_faceNum, Stu4_Faceshapes = Dlib_detectFace(Stu4_Img)
                Stu4_FaceDetected.append(Stu4_isFaceDetected)
                if Stu4_isFaceDetected:
                    Stu4_imgV = Stu4_Img
                    Stu4_FaceShapeV = Stu4_Faceshapes[0]

                Stu5_isFaceDetected, Stu5_faceNum, Stu5_Faceshapes = Dlib_detectFace(Stu5_Img)
                Stu5_FaceDetected.append(Stu5_isFaceDetected)
                if Stu5_isFaceDetected:
                    Stu5_imgV = Stu5_Img
                    Stu5_FaceShapeV = Stu5_Faceshapes[0]

                Stu6_isFaceDetected, Stu6_faceNum, Stu6_Faceshapes = Dlib_detectFace(Stu6_Img)
                Stu6_FaceDetected.append(Stu6_isFaceDetected)
                if Stu6_isFaceDetected:
                    Stu6_imgV = Stu6_Img
                    Stu6_FaceShapeV = Stu6_Faceshapes[0]

                if Stu1_faceNum > 1 and Stu1_firstflag:
                    Stu1_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对6课程中[No.1]学生视频出现人脸数量超标！")
                if Stu2_faceNum > 1 and Stu2_firstflag:
                    Stu2_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对6课程中[No.2]学生视频出现人脸数量超标！")
                if Stu3_faceNum > 1 and Stu3_firstflag:
                    Stu3_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_3.jpg', Stu3_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对6课程中[No.3]学生视频出现人脸数量超标！")
                if Stu4_faceNum > 1 and Stu4_firstflag:
                    Stu4_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_4.jpg', Stu4_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对6课程中[No.4]学生视频出现人脸数量超标！")
                if Stu5_faceNum > 1 and Stu5_firstflag:
                    Stu5_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_5.jpg', Stu5_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对6课程中[No.5]学生视频出现人脸数量超标！")
                if Stu6_faceNum > 1 and Stu6_firstflag:
                    Stu6_firstflag = 0
                    cv2.imwrite(fileName + '_Detected_Picture_6.jpg', Stu6_Img)
                    IsItIllegal = 1
                    CaseReason.append(u"1对6课程中[No.6]学生视频出现人脸数量超标！")

                # 识别人脸
                if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and len(Stu3_imgV) > 0 and len(Stu4_imgV) > 0 \
                        and len(Stu5_imgV) > 0 and len(Stu6_imgV) > 0 and Recog_flag:
                    Recog_flag = 0
                    imgV = [Stu1_imgV[0], Stu2_imgV[0], Stu3_imgV[0], Stu4_imgV[0], Stu5_imgV[0], Stu6_imgV[0]]
                    FaceShapeV = [Stu1_FaceShapeV[0], Stu2_FaceShapeV[0], Stu3_FaceShapeV[0], Stu4_FaceShapeV[0],
                                  Stu5_FaceShapeV[0], Stu6_FaceShapeV[0]]
                    RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                    if RecogIsIllegal:
                        cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                        cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                        cv2.imwrite(fileName + '_Recognized_Picture_3.jpg', imgV[2])
                        cv2.imwrite(fileName + '_Recognized_Picture_4.jpg', imgV[3])
                        cv2.imwrite(fileName + '_Recognized_Picture_5.jpg', imgV[4])
                        cv2.imwrite(fileName + '_Recognized_Picture_6.jpg', imgV[5])
                        # print RecogComparisonMatrix

            else:
                intervalFrames = video_fps * intervalTime
                if FramesCNT % intervalFrames == 0:
                    if (FramesCNT + intervalFrames) > video_frames:
                        Recog_flag = 1
                    # 检测人脸
                    Stu1_isFaceDetected, Stu1_faceNum, Stu1_Faceshapes = Dlib_detectFace(Stu1_Img)
                    Stu1_FaceDetected.append(Stu1_isFaceDetected)
                    if Stu1_isFaceDetected:
                        Stu1_imgV = Stu1_Img
                        Stu1_FaceShapeV = Stu1_Faceshapes[0]

                    Stu2_isFaceDetected, Stu2_faceNum, Stu2_Faceshapes = Dlib_detectFace(Stu2_Img)
                    Stu2_FaceDetected.append(Stu2_isFaceDetected)
                    if Stu2_isFaceDetected:
                        Stu2_imgV = Stu2_Img
                        Stu2_FaceShapeV = Stu2_Faceshapes[0]

                    Stu3_isFaceDetected, Stu3_faceNum, Stu3_Faceshapes = Dlib_detectFace(Stu3_Img)
                    Stu3_FaceDetected.append(Stu3_isFaceDetected)
                    if Stu3_isFaceDetected:
                        Stu3_imgV = Stu3_Img
                        Stu3_FaceShapeV = Stu3_Faceshapes[0]

                    Stu4_isFaceDetected, Stu4_faceNum, Stu4_Faceshapes = Dlib_detectFace(Stu4_Img)
                    Stu4_FaceDetected.append(Stu4_isFaceDetected)
                    if Stu4_isFaceDetected:
                        Stu4_imgV = Stu4_Img
                        Stu4_FaceShapeV = Stu4_Faceshapes[0]

                    Stu5_isFaceDetected, Stu5_faceNum, Stu5_Faceshapes = Dlib_detectFace(Stu5_Img)
                    Stu5_FaceDetected.append(Stu5_isFaceDetected)
                    if Stu5_isFaceDetected:
                        Stu5_imgV = Stu5_Img
                        Stu5_FaceShapeV = Stu5_Faceshapes[0]

                    Stu6_isFaceDetected, Stu6_faceNum, Stu6_Faceshapes = Dlib_detectFace(Stu6_Img)
                    Stu6_FaceDetected.append(Stu6_isFaceDetected)
                    if Stu6_isFaceDetected:
                        Stu6_imgV = Stu6_Img
                        Stu6_FaceShapeV = Stu6_Faceshapes[0]

                    if Stu1_faceNum > 1 and Stu1_firstflag:
                        Stu1_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_1.jpg', Stu1_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对6课程中[No.1]学生视频出现人脸数量超标！")
                    if Stu2_faceNum > 1 and Stu2_firstflag:
                        Stu2_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_2.jpg', Stu2_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对6课程中[No.2]学生视频出现人脸数量超标！")
                    if Stu3_faceNum > 1 and Stu3_firstflag:
                        Stu3_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_3.jpg', Stu3_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对6课程中[No.3]学生视频出现人脸数量超标！")
                    if Stu4_faceNum > 1 and Stu4_firstflag:
                        Stu4_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_4.jpg', Stu4_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对6课程中[No.4]学生视频出现人脸数量超标！")
                    if Stu5_faceNum > 1 and Stu5_firstflag:
                        Stu5_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_5.jpg', Stu5_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对6课程中[No.5]学生视频出现人脸数量超标！")
                    if Stu6_faceNum > 1 and Stu6_firstflag:
                        Stu6_firstflag = 0
                        cv2.imwrite(fileName + '_Detected_Picture_6.jpg', Stu6_Img)
                        IsItIllegal = 1
                        CaseReason.append(u"1对6课程中[No.6]学生视频出现人脸数量超标！")

                    # 识别人脸
                    if len(Stu1_imgV) > 0 and len(Stu2_imgV) > 0 and len(Stu3_imgV) > 0 and len(Stu4_imgV) > 0 \
                            and len(Stu5_imgV) > 0 and len(Stu6_imgV) > 0 and Recog_flag:
                        Recog_flag = 0
                        imgV = [Stu1_imgV[0], Stu2_imgV[0], Stu3_imgV[0], Stu4_imgV[0], Stu5_imgV[0], Stu6_imgV[0]]
                        FaceShapeV = [Stu1_FaceShapeV[0], Stu2_FaceShapeV[0], Stu3_FaceShapeV[0], Stu4_FaceShapeV[0],
                                      Stu5_FaceShapeV[0], Stu6_FaceShapeV[0]]
                        print FaceShapeV
                        RecogIsIllegal, RecogComparisonMatrix = PairWiseComparison(imgV, FaceShapeV)
                        if RecogIsIllegal:
                            cv2.imwrite(fileName + '_Recognized_Picture_1.jpg', imgV[0])
                            cv2.imwrite(fileName + '_Recognized_Picture_2.jpg', imgV[1])
                            cv2.imwrite(fileName + '_Recognized_Picture_3.jpg', imgV[2])
                            cv2.imwrite(fileName + '_Recognized_Picture_4.jpg', imgV[3])
                            cv2.imwrite(fileName + '_Recognized_Picture_5.jpg', imgV[4])
                            cv2.imwrite(fileName + '_Recognized_Picture_6.jpg', imgV[5])
                            # print RecogComparisonMatrix

            FramesCNT += 1
            if FramesCNT % video_frames_PrintPercent == 0:
                print str(round(float(FramesCNT) / video_frames, 2) * 100) + "%"
            # cv2.imshow("Demo1", Stu1_Img)
            # cv2.imshow("Demo2", Stu2_Img)
            # cv2.imshow("Demo3", Stu3_Img)
            # cv2.imshow("Demo4", Stu4_Img)
            # cv2.imshow("Demo5", Stu5_Img)
            # cv2.imshow("Demo6", Stu6_Img)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

    Stu1_FaceShowRatio = float(np.sum(Stu1_FaceDetected)) / len(Stu1_FaceDetected)
    if Stu1_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu1_FaceShowRatio == 0:
            CaseReason.append(u"1对6课程中[No.1]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对6课程中[No.1]学生露脸时长过低！")
    Stu2_FaceShowRatio = float(np.sum(Stu2_FaceDetected)) / len(Stu2_FaceDetected)
    if Stu2_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu2_FaceShowRatio == 0:
            CaseReason.append(u"1对6课程中[No.2]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对6课程中[No.2]学生露脸时长过低！")
    Stu3_FaceShowRatio = float(np.sum(Stu3_FaceDetected)) / len(Stu3_FaceDetected)
    if Stu3_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu3_FaceShowRatio == 0:
            CaseReason.append(u"1对6课程中[No.3]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对6课程中[No.3]学生露脸时长过低！")
    Stu4_FaceShowRatio = float(np.sum(Stu4_FaceDetected)) / len(Stu4_FaceDetected)
    if Stu4_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu4_FaceShowRatio == 0:
            CaseReason.append(u"1对6课程中[No.4]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对6课程中[No.4]学生露脸时长过低！")
    Stu5_FaceShowRatio = float(np.sum(Stu5_FaceDetected)) / len(Stu5_FaceDetected)
    if Stu5_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu5_FaceShowRatio == 0:
            CaseReason.append(u"1对6课程中[No.5]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对6课程中[No.5]学生露脸时长过低！")
    Stu6_FaceShowRatio = float(np.sum(Stu6_FaceDetected)) / len(Stu6_FaceDetected)
    if Stu6_FaceShowRatio < 0.3:
        IsItIllegal = 1
        if Stu6_FaceShowRatio == 0:
            CaseReason.append(u"1对6课程中[No.6]学生全程未检测到脸！")
        else:
            CaseReason.append(u"1对6课程中[No.6]学生露脸时长过低！")

    Stu_FaceShowRatio = [Stu1_FaceShowRatio, Stu2_FaceShowRatio, Stu3_FaceShowRatio, Stu4_FaceShowRatio,
                         Stu5_FaceShowRatio, Stu6_FaceShowRatio]

    if not CaseReason:
        CaseReason = u"此视频无异常！"
    if not RecogComparisonMatrix:
        RecogComparisonMatrix = "No Similar!"

    return IsItIllegal, CaseReason, RecogComparisonMatrix, Stu_FaceShowRatio


def processVideo(VideoPath, GroupSize, intervalTime=5):
    fileName = VideoPath.split("/")[-1]
    camera = cv2.VideoCapture(VideoPath)
    camera_open_success = camera.isOpened()
    video_fps = round(camera.get(5))
    video_frames = round(camera.get(7))
    video_frames_PrintPercent = int(video_frames / 100)
    # print u'摄像头FPS:%d, FRAMES:%d!' % (video_fps, video_frames)

    bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio = None, None, None, None
    if not camera_open_success:
        print u"读取视频出错！"
    else:
        if GroupSize == 1:
            bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio = RecogClass1vs1(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime)
            print '----------> 1'
        elif GroupSize == 2:
            bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio = RecogClass1vs2(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime)
            print '----------> 2'
        elif GroupSize == 3:
            bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio = RecogClass1vs3(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime)
            print '----------> 3'
        elif GroupSize == 4:
            bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio = RecogClass1vs4(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime)
            print '----------> 4'
        elif GroupSize == 5:
            bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio = RecogClass1vs5(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime)
            print '----------> 5'
        elif GroupSize == 6:
            bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio = RecogClass1vs6(fileName, camera, video_fps, video_frames, video_frames_PrintPercent, intervalTime)
            print '----------> 6'
        else:
            print u'小组课参数输入错误! '

    return bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio


if __name__ == '__main__':
    VideoPath = "TestVideo/600_1vs6.mp4"
    # VideoPath = "TestVideo/f3.mp4"

    bIsItIllegal, bCaseReason, bRecogComparisonMatrix, bStu_FaceShowRatio = processVideo(VideoPath, 6, 0)
    print bIsItIllegal
    print u"——————————"
    for i in bCaseReason:
        print i
    print u"——————————"
    print bRecogComparisonMatrix
    print bStu_FaceShowRatio
