import cv2
import numpy as np

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def with_points_drawn(img, points):
    pts = np.array([points[0],points[1],points[2],points[3]], np.int32)
    pts = pts.reshape((-1,1,2))
    return cv2.polylines(img,[pts],True,(255,0,0), 5)