#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
# from common.transformations import camera as camera

def countApparitions(rho, theta, parameters):
    k = 0
    for param in parameters:
        if rho == param[0] and theta == param[1]:
            k += 1
    return k

def roi(frame):
#     height = int(frame.shape[0] - 300 / 2)
    height = int(frame.shape[0] / 2)
    width = frame.shape[1] - 200
    triangle = np.array([[(200, height), (width, height), (600, 400)]])
    mask = np.zeros_like(frame, dtype = "uint8")
    cv.fillPoly(mask, triangle, 255)
    maskedImage = cv.bitwise_and(frame, mask)
    return maskedImage

# takes hough lines parameters rho and theta as inputs v is assumed to be 1 (if not, use countApparitions function)
# and returns (x0, y0) coord of the vanishing point
def MatessiLombardi(parameters, outliers):
    A, B, C, D, E = (0,) * 5
    V = parameters.shape[0]
    print("shape =", parameters.shape)
    for param in parameters:
        rho = param[0]
        theta = param[1]
        if rho == 0 and theta == 0:
            v = outliers
        else:
            v = 1
#         v = countApparitions(rho, theta, parameters)
        ratio = v / V
        A += ratio * np.cos(theta)**2
        B += ratio * np.sin(theta)**2
        C += ratio * np.sin(theta) * np.cos(theta)
        D += ratio * np.cos(theta) * rho
        E += ratio * np.sin(theta) * rho
    S1 = np.array([[A, C], [C, B]])
    S2 = np.array([D, E])
    ret = np.linalg.solve(S1, S2) 
    return ret

# rho = x0*cos(theta) + y0*sin(theta)
def rho(line, coord):
    return coord[0] * np.cos(line[1]) + coord[1] * np.sin(line[1])

def linesWithHough(frame):
    bnwFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    bnwFrame = cv.GaussianBlur(bnwFrame, (5, 5), 0)
    canny = cv.Canny(bnwFrame, 100, 100)
    canny = roi(canny)
    cv.imshow("canny", canny)
    houghLines = cv.HoughLines(canny, 1, np.pi / 180, 1)
    houghLines = np.reshape(houghLines, (houghLines.shape[0], houghLines.shape[2]))
    plot(houghLines)
    return houghLines

def plot(lines, status = "before"):
    rho = []
    theta = []
    for line in lines:
        rho.append(line[0])
        theta.append(line[1])
    plt.scatter(rho, theta)
    plt.xlabel("rho")
    plt.ylabel("theta")
    if status == "before":
        plt.title("before algo")
    elif status == "after":
        plt.title("after algo")
    plt.show()

def residualsVar(lines, coord, outliers):
    sigma = 0
    for line in lines:
        if line[0] == 0 and line[1] == 0:
            ratio = outliers / lines.shape[0]
        else:
            ratio = 1 / lines.shape[0]
        sigma += ratio * (line[0] - rho(line, coord))**2
    return np.sqrt(sigma)

def rejectOutliers(lines, coord, sigma):
    ret = []
    outliers = 0
    for line in lines:
        if abs(line[0] - rho(line, coord)) <= 2 * sigma:
            ret.append(line)
        else:
            outliers += 1
            ret.append([0, 0])
    ret = np.array(ret, dtype = object)
#     plot(ret, "after")
    return ret, outliers

def vanishingPointCoord(frame, eps):
    lines = linesWithHough(frame)
    k, outliers = (0,) * 2
    coord = []
    coord = list(coord)
    coord.append(MatessiLombardi(lines, outliers))
    coord = np.asarray(coord)
    while True:
        k += 1
        sigma = residualsVar(lines, coord[k - 1], outliers)
        lines, outliers = rejectOutliers(lines, coord[k - 1], sigma)
        coord = list(coord)
        coord.append(MatessiLombardi(lines, outliers))
        coord = np.asarray(coord)
        print("k =", k, "error =", abs(coord[k][0] - coord[k - 1][0]) + abs(coord[k][1] - coord[k - 1][1]))
        if abs(coord[k][0] - coord[k - 1][0]) + abs(coord[k][1] - coord[k - 1][1]) <= eps:
            break
    coord = coord[k]
    image = cv.circle(frame, (int(coord[0]), int(coord[1])), 5, (0, 0, 255), cv.FILLED)
    cv.imshow("center", image)
    cv.waitKey(20)
    return coord

if __name__ == "__main__":
    frame = cv.imread("/Users/kerianyousfi/Downloads/vp.jpg")
    lines = linesWithHough(frame)
    k, outliers = (0,)*2
    coord = list(coord)
#     coord = vanishingPointCoord(frame, 100)
    coord = MatessiLombardi(lines, outliers)
    image = cv.circle(frame, (int(coord[0]), int(coord[1])), 5, (0, 0, 255), cv.FILLED)
    cv.imshow("result", image)
    cv.waitKey(30000000)
#     path = "/Users/kerianyousfi/fun/CHALLENGE_COMMA/calib_challenge/labeled/1.hevc"
#     cap = cv.VideoCapture(path)
#     ret, frame = cap.read()
#     k = 0
#     while ret:
#         vanishingPointCoord(frame, 1000)
#         if k == 1:
#             pass
#         if cv.waitKey(10) & 0xFF == ord("q"):
#             break
#         ret, frame = cap.read()
#         k += 1
# 
#     cap.release()
#     cv.destroyAllWindows()
