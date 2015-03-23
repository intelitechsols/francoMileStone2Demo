import numpy as np
from numpy import math, hstack

import cv2


class VidCapFile(object):
    def __init__(self, path):
        self.path = path
        self.frame = 1

    def read(self):
        img = cv2.imread(self.path.format(self.frame))
        state = img != None
        if state: self.frame += 1
        return state, img

def isImgOpened(self):
    img = cv2.imread(self.path.format(self.frame))
    return img != None

def L2norm(X):
    return np.sqrt((X ** 2).sum(axis=1))

def array2integer(arr):
    Integer = arr
    return int(Integer[0]), int(Integer[1])

def squeeze_pixelpts(X_Y):
    X_Y = X_Y.squeeze()
    if len(X_Y.shape) == 1: X_Y = np.array([X_Y])
    return X_Y


current_pos = None
tl = None
br = None

def get_bbox(img, title='get_bbox'):
    global current_pos
    global tl
    global br
    global released_once

    current_pos = None
    tl = None
    br = None
    released_once = False

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):
        global current_pos
        global tl
        global br
        global released_once

        current_pos = (x, y)

        if tl is not None and not (flags & cv2.EVENT_FLAG_LBUTTON): released_once = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if tl is None: tl = current_pos
            elif released_once: br = current_pos

    cv2.setMouseCallback(title, onMouse)
    cv2.imshow(title, img)

    while br is None:

        img_draw = np.copy(img)
        if tl is not None: cv2.rectangle(img_draw, tl, current_pos, (255, 0, 0))
        cv2.imshow(title, img_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(title)

    return (tl, br)

def init_bbox(keypts, tl, br):
    if type(keypts) is list: keypts = cv2npkeypts(keypts)
    X = keypts[:, 0]
    Y = keypts[:, 1]
    PixX1 = X > tl[0]
    PixY1 = Y > tl[1]
    PixX2 = X < br[0]
    PixY2 = Y < br[1]
    return (PixX1 & PixY1 & PixX2 & PixY2)

def tracker(img_prev, img_gray, keypts, THR_FB=20):
    if type(keypts) is list: keypts = cv2npkeypts(keypts)
    num_keypts = keypts.shape[0]
    # Status of tracked keypoint - True means successfully tracked
    status = [False] * num_keypts
    # If at least one keypoint is active
    if num_keypts > 0:
        # Prepare data for opencv: Use only first and second column
        # ensure data type is float32
        pts = keypts[:, None, :2].astype(np.float32)
        # Calculate optical flow forward:  for prev_location
        Pts_fwd, status, _ = cv2.calcOpticalFlowPyrLK(img_prev, img_gray, pts, None)
        # Calculate optical flow backward: for prev_location
        pts_bck, _, _ = cv2.calcOpticalFlowPyrLK(img_gray, img_prev, Pts_fwd, None)
        # Get rid of  singleton dimension
        pts_bck = squeeze_pixelpts(pts_bck)
        pts = squeeze_pixelpts(pts)
        Pts_fwd = squeeze_pixelpts(Pts_fwd)
        status = status.squeeze()
        # Calculate forward-backward error
        fwdbck_err = np.sqrt(np.power(pts_bck - pts, 2).sum(axis=1))
        # Set status depending on fwdbck_err and lk error
        large_fb = fwdbck_err > THR_FB
        status = ~large_fb & status.astype(np.bool)
        Pts_fwd = Pts_fwd[status, :]
        tracked_keypts = keypts[status, :]
        tracked_keypts[:, :2] = Pts_fwd

    else: tracked_keypts = np.array([])

    return tracked_keypts, status

def cv2npkeypts(keypts_cv):
    keypts = np.array([k.pt for k in keypts_cv])
    return keypts

def locate_closest_keypts(keypts, pos, number=1):
    if type(pos) is tuple: pos = np.array(pos)
    if type(keypts) is list: keypts = cv2npkeypts(keypts)
    positions_cv2npkeypts = np.sqrt(np.power(keypts - pos, 2).sum(axis=1))
    ind = np.argsort(positions_cv2npkeypts)
    return ind[:number]

def project_keypoints(keypts, img, color=(255, 0, 0)):
    for k in keypts:
        r = 2  # int(k.size / 2)
        c = (int(k[0]), int(k[1]))
        # Project points through a circle
        cv2.circle(img, c, r, color)


def rotation(pnt, rad):
    if (rad == 0): return pnt
    pnt_rot = np.empty(pnt.shape)
    sint, cost = [f(rad) for f in (math.sin, math.cos)]
    pnt_rot[:, 0] = cost * pnt[:, 0] - sint * pnt[:, 1]
    pnt_rot[:, 1] = sint * pnt[:, 0] + cost * pnt[:, 1]
    return pnt_rot

def bboxresults(bbox):
    # return BBOX results
    return hstack((bbox[:, [0]] + bbox[:, [2]] - 1, bbox[:, [1]] + bbox[:, [3]] - 1))

def bbox2pts(bbox):
    # return points
    return hstack((bbox[:, :2], bboxresults(bbox)))
