import pybullet as p
import numpy as np
import time
from camera import Camera
import cv2
import math

# считаем матрицу перехода
def computeInterMatrix(Z, sd0):
    L = np.zeros((8,6))
    for idx in range(4):
        x = sd0[2*idx, 0]
        y = sd0[2*idx+1, 0]
        L[2*idx] = np.array([-1/Z,0, x/Z, x*y, -(1+x*x), y])
        L[2*idx+1] = np.array([0,-1/Z, y/Z, 1+y*y, -x*y, -x])
    return L

# обновляем положение камеры
def updateCamPos(cam):
    linkState = p.getLinkState(boxId, linkIndex=eefLinkIdx)
    xyz = linkState[0] - np.array([0, 0, 0.05]) # position
    quat = linkState[1] # orientation
    rotMat = p.getMatrixFromQuaternion(quat)
    rotMat = np.reshape(np.array(rotMat),(3,3))
    camera.set_new_position(xyz, rotMat)

# инициализация Aruco детектора
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
###

# инициализация камеры
CAM_IMG_SIDE = 300
CAM_IMG_HALF = CAM_IMG_SIDE / 2
camera = Camera(imgSize = [CAM_IMG_SIDE, CAM_IMG_SIDE])
###

# инициализация среды
physicsClient = p.connect(p.GUI, options="--background_color_red=1 --background_color_blue=1 --background_color_green=1")
p.setGravity(0,0,-10)
boxId = p.loadURDF("./simple.urdf.xml", useFixedBase=True) # подгружаем робота

# add aruco cube and aruco texture
c = p.loadURDF('aruco.urdf', (0.5, 0.5, 0.0), useFixedBase=True)
x = p.loadTexture('aruco_cube.png')
p.changeVisualShape(c, -1, textureUniqueId=x)
###

# вид сверху
# p.resetDebugVisualizerCamera(
#     cameraDistance=0.5,
#     cameraYaw=-90,
#     cameraPitch=-89.999,
#     cameraTargetPosition=[0.5, 0.5, 0.6]
# )
###

dt = 1/240
maxTime = 10
logTime = np.arange(0.0, maxTime, dt)

jointIndices = [1, 3, 5, 7]
eefLinkIdx = 8

Z0 = 0.5 # высота базы

# go to the desired position
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[0.0, 1.5708, -0.15, 0.0], controlMode=p.POSITION_CONTROL)
for _ in range(100):
    p.stepSimulation()

updateCamPos(camera)
img = camera.get_frame()
corners, markerIds, rejectedCandidates = detector.detectMarkers(img)

sd0 = np.reshape(np.array(corners[0][0]),(8,1))
sd0 = np.array([(s-CAM_IMG_HALF)/CAM_IMG_HALF for s in sd0])

sd = np.reshape(np.array(corners[0][0]),(8,1)).astype(int) # вектор s - координаты углов маркера на картинке
###

# go to the starting position
p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=[0.2, 1.4708, 0.15, 0.1], controlMode=p.POSITION_CONTROL)
for _ in range(100):
    p.stepSimulation()
###

idx = 1
camCount = 0
w = np.zeros((6,1))

# запускаем симуляцию
for t in logTime[1:]:
    p.stepSimulation()
    camCount += 1

    if (camCount == 5):
        camCount = 0 # сбрасываем счетчик
        updateCamPos(camera)
        img = camera.get_frame()
        corners, markerIds, rejectedCandidates = detector.detectMarkers(img)
        s = corners[0][0,0]
        s0 = np.reshape(np.array(corners[0][0]),(8,1))
        s0 = np.array([(ss-CAM_IMG_HALF)/CAM_IMG_HALF for ss in s0])

        Z = Z0 + p.getLinkState(boxId, linkIndex=eefLinkIdx)[0][2] - 0.05 # координата Z - расположения камеры

        L0 = computeInterMatrix(Z0, s0) # (8x6) матрица перехода для 3х-мерного случая
        L0T = np.linalg.inv(L0.T@L0)@L0.T # псевдо-обратная матрица
        e = s0 - sd0 # невязки
        coef = 1/2
        w = -coef * L0T @ e 

    # ЯКОБИАН-1
    # находим координаты и скорости наших joint'ов
    jStates = p.getJointStates(boxId, jointIndices=jointIndices)
    jPos = [state[0] for state in jStates]
    jVel = [state[1] for state in jStates]

    (linJac,angJac) = p.calculateJacobian(
        bodyUniqueId = boxId, 
        linkIndex = eefLinkIdx,
        localPosition = [0,0,0],
        objPositions = jPos,
        objVelocities = [0,0,0,0],
        objAccelerations = [0,0,0,0]
    )

    
    J = np.block([
        [np.array(linJac)[:3,:3], np.zeros((3,1))],
        [np.array(angJac)[2,:]]
    ])

    J = np.block([[np.array(linJac)[:,:]],
                [np.array(angJac)[:,:]]])
    
    print("J :", J)
    #

    L1 = L2 = L = 0.5
    th1 = p.getJointState(boxId, 1)[0]
    th2 = p.getJointState(boxId, 3)[0]

    # Якобиан по учебнику
    # J = np.array([  
    #                 [ -L*math.cos(th1), -L*math.cos(th1)-L*math.cos(th1+th2), 0, 0],
    #                 [ L*math.sin(th1), L*math.sin(th1)+L*math.sin(th1+th2), 0, 0],
    #                 [ 0, 0, 1, 0],
    #                 [ 0, 0, 0, 0],
    #                 [ 0, 0, 0, 0],
    #                 [ 1, 1, 0, 1]
    #             ])
    ###

    # Якобиан по моему расчету
    # J = np.array([[-L1*math.cos(th1)-L2*math.cos(th1+th2), -L2*math.cos(th1+th2), 0, 0],
    #              [L1*math.sin(th1)+L2*math.sin(th1+th2),L2*math.sin(th1+th2),0, 0],
    #              [0, 0, -1, 0],
    #              [0, 0, 0, 0],
    #              [0, 0, 0, 0],
    #              [1, 1, 0, 1]
    #              ])
    # print("J :", J)
    # ###
    J_inv = np.linalg.inv(J.T @ J)@J.T
    dq = (J_inv @ w).flatten()[[1,0,2,3]] # меняем x, y местами
    dq[3] = -dq[3] # угол в другую сторону
    dq[2] = -dq[2] # ось Oz в другую сторону
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetVelocities=dq, controlMode=p.VELOCITY_CONTROL)
    
p.disconnect()