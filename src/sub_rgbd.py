#!/usr/bin/env python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn, Aymeric Dujardin
@date: 20180911
"""
# pylint: disable=R, W0401, W0614, W0703
import os
import sys
import time
import logging
import random
from random import randint
import math
import statistics
import getopt
from ctypes import *
import numpy as np
import cv2
import pyzed.sl as sl
import rospy
from sensor_msgs.msg import Image

from std_msgs.msg import Header
from std_msgs.msg import String
from darknet_ros_msgs.msg import centerBdbox
from darknet_ros_msgs.msg import centerBdboxes
from darknet_ros_msgs.srv import get_camParam, get_camParamResponse

import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rospy.init_node('subRGBD', anonymous=True)
image_sub = message_filters.Subscriber('/camera/rgb/image_color', Image)
depth_sub = message_filters.Subscriber('/camera/depth/image', Image)

ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.1, allow_headerless=True)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/docker/output.avi', fourcc, 20.0, (640,480))

def callback(ros_image, ros_depth):
    global thresh, color_array, cap, out
    bridge = CvBridge()
    image = CvBridge().imgmsg_to_cv2(ros_image, "bgr8")
    image_raw = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    depth_img = bridge.imgmsg_to_cv2(ros_depth, "32FC1")
    thresh = 0.5
    # Do the detection
    detections = detect(netMain, metaMain, image, thresh)

    log.info(chr(27) + "[2J"+"**** " + str(len(detections)) + " Results ****")
    boundingboxes = []
    labelPoints = []
    id = 0
    # Create check frame
    h, w, c = image.shape
    for detection in detections:
        label = detection[0]
        id = get_id(label)
        if id == -1:
            continue
        confidence = detection[1]
        # pstring = label+": "+str(np.int(100 * confidence))+"%"
        # log.info(pstring)
        bounds = detection[2]
        y_extent = int(bounds[3])
        x_extent = int(bounds[2])
        # Coordinates are around the center
        x_coord = int(bounds[0] - bounds[2]/2)
        y_coord = int(bounds[1] - bounds[3]/2)
        #boundingBox = [[x_coord, y_coord], [x_coord, y_coord + y_extent], [x_coord + x_extent, y_coord + y_extent], [x_coord + x_extent, y_coord]]
        thickness = 1
        depth = get_depth(depth_img, bounds)
        box = centerBdbox()
        box.probability = confidence
        box.x_cen = int(bounds[0])
        box.y_cen = int(bounds[1])
        box.width = int(bounds[2])
        box.height = int(bounds[3])
        box.Class = label
        box.id = id
        box.depth = depth
        # print(label, depth)
        # id += 1
        boundingboxes.append(box)
        weight = (bounds[2]*bounds[3])*0.05
        area_div = math.sqrt(weight)/2
        cv2.rectangle(image, (int(bounds[0] - area_div)- thickness, int(bounds[1] - area_div)-thickness),
                        (int(bounds[0] + area_div) + thickness, int(bounds[1] + area_div)+thickness),
                        color_array[detection[3]], int(thickness*2))

        cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                        (x_coord + x_extent + thickness, y_coord + (18 + thickness*4)),
                        color_array[detection[3]], -1)
        cv2.putText(image, label + " " +  (str("{:.2f}".format(depth)) + " m"),
                    (x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(image, (x_coord - thickness, y_coord - thickness),
                        (x_coord + x_extent + thickness, y_coord + y_extent + thickness),
                        color_array[detection[3]], int(thickness*2))

    t = rospy.get_rostime()
    # Publish left and right image for Slam
    bridge = CvBridge()
    vis_msg_frame = bridge.cv2_to_imgmsg(image)
    img_msg_frame = bridge.cv2_to_imgmsg(image_raw)
    depth_msg_frame = bridge.cv2_to_imgmsg(depth_img)

    vis_msg_frame.encoding = "rgb8"
    img_msg_frame.encoding = "rgb8"
    depth_msg_frame.encoding = "32FC1"

    vis_msg_frame.header = Header()
    vis_msg_frame.header.stamp = t;
    vis_msg_frame.header.frame_id = "vis";

    img_msg_frame.header = Header()
    img_msg_frame.header.stamp = t;
    img_msg_frame.header.frame_id = "camera_left";
    depth_msg_frame.header = Header()
    depth_msg_frame.header.stamp = t;
    depth_msg_frame.header.frame_id = "camera_right";

    boundingbox_msg = centerBdboxes()
    boundingbox_msg.header = Header()
    boundingbox_msg.header.stamp = t;
    boundingbox_msg.header.frame_id = "object_detection"
    boundingbox_msg.centerBdboxes = boundingboxes
    
    vis_img_pub.publish(vis_msg_frame)
    img_pub.publish(img_msg_frame)
    depth_pub.publish(depth_msg_frame)
    boundingbox_pub.publish(boundingbox_msg)
    
    out.write(image)
    cv2.imshow("ZED", image)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()




ts.registerCallback(callback)

# Public image for openvslam
vis_img_pub = rospy.Publisher('/camera/vis', Image, queue_size=10)
img_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
depth_pub = rospy.Publisher('/camera/image_depth', Image, queue_size=10)
# img_pub = rospy.Publisher('/camera/rgb/image_color', Image, queue_size=10)
# depth_pub = rospy.Publisher('/camera/depth/image', Image, queue_size=10)
blend_img_pub = rospy.Publisher('/camera/blend/image_raw', Image, queue_size=10)
boundingbox_pub = rospy.Publisher('camera/boundingbox', centerBdboxes, queue_size=10)


# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/docker/catkin_ws/src/zed_yolo/src/libdarknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                log.info("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # log.info(os.environ.keys())
            # log.warning("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            log.warning("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            log.warning("Environment variables indicated a CPU run, but we didn't find `" +
                        winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("/home/docker/catkin_ws/src/zed_yolo/src/libdarknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            name_tag = meta.names[i]
        else:
            name_tag = altNames[i]
        res.append((name_tag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the detection
    """
    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(
        net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    if debug:
        log.debug("about to range")
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    name_tag = meta.names[i]
                else:
                    name_tag = altNames[i]
                res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res


netMain = None
metaMain = None
altNames = None


def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.
    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []
    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median

def get_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.
    Return:
        x, y, z: Location of object in meters.
    '''
    weight = (bounds[2]*bounds[3])*0.05
    area_div = math.sqrt(weight)/2
    # area_div = 5

    z_vect = []
    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j]
            # print(ret, z)
            # if str(ret) == 'SUCCESS' and not np.isnan(z) and not np.isinf(z):
            if not np.isnan(z) and not np.isinf(z):
                z_vect.append(z)
    try:
        z_median = statistics.median(z_vect)
    except Exception:
        z_median = -1
        pass

    return z_median

def get_target_pcl(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.
    Return:
        x, y, z: Location of object in meters.
    '''
    weight = (bounds[2]*bounds[3])*0.05
    area_div = math.sqrt(weight)/2
    list = []
    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                # x, y, z
                pointPix = {}
                pointPix["point"]=(depth[i, j, 0], depth[i, j, 1], depth[i, j, 2])
                pointPix["pixel"]=(i,j)
                list.append(pointPix)
    return list

def generate_color(meta_path):
    '''
    Generate random colors for the number of classes mentioned in data file.
    Arguments:
    meta_path: Path to .data file.

    Return:
    color_array: RGB color codes for each class.
    '''
    random.seed(42)
    with open(meta_path, 'r') as f:
        content = f.readlines()
    class_num = int(content[0].split("=")[1])
    color_array = []
    for x in range(0, class_num):
        color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_array

def handle_get_camParam(request):
    global calibration_params
    response = get_camParamResponse()
    response.fx= calibration_params.left_cam.fx
    response.fy= calibration_params.left_cam.fy
    response.cx= calibration_params.left_cam.cx
    response.cy= calibration_params.left_cam.cy
    response.k1= calibration_params.left_cam.disto[0]
    response.k2= calibration_params.left_cam.disto[1]
    response.p1= calibration_params.left_cam.disto[2]
    response.p2= calibration_params.left_cam.disto[3]
    response.k3= calibration_params.left_cam.disto[4]
    response.focal_x_baseline= calibration_params.left_cam.fx * calibration_params.get_camera_baseline()

    return response

rospy.Service('add_two_ints', get_camParam, handle_get_camParam)

target_obj = {"bottle":1,
"cup":2,
"tvmonitor":3,
"laptop":4,
"mouse":5,
"remote":6,
"keyboard":7,
"cell phone":8,
"chair":9,
"diningtable":10,
"book":11,
"teddy bear":12}

def get_id(label):
    if label in target_obj:
        return target_obj[label]
    else:
        return -1

def main(argv):
    global thresh, color_array, cap, out
    thresh = 0.25
    darknet_path="/home/docker/catkin_ws/src/zed_yolo/src/libdarknet/"
    config_path = darknet_path + "cfg/yolov3.cfg"
    weight_path = darknet_path + "weights/yolov3.weights"
    meta_path = darknet_path + "cfg/coco.data"
    svo_path = None
    zed_id = 0
    
    help_str = 'darknet_zed.py -c <config> -w <weight> -m <meta> -t <threshold> -s <svo_file> -z <zed_id>'
    try:
        opts, args = getopt.getopt(
            argv, "hc:w:m:t:s:z:", ["config=", "weight=", "meta=", "threshold=", "svo_file=", "zed_id="])
    except getopt.GetoptError:
        log.exception(help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            log.info(help_str)
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg
        elif opt in ("-w", "--weight"):
            weight_path = arg
        elif opt in ("-m", "--meta"):
            meta_path = arg
        elif opt in ("-t", "--threshold"):
            thresh = float(arg)
        elif opt in ("-s", "--svo_file"):
            svo_path = arg
        elif opt in ("-z", "--zed_id"):
            zed_id = int(arg)

    input_type = sl.InputType()
    if svo_path is not None:
        log.info("SVO file : " + svo_path)
        input_type.set_from_svo_file(svo_path)
    else:
        # Launch camera by id
        input_type.set_from_camera_id(zed_id)

    # Import the global variables. This lets us instance Darknet once,
    # then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(config_path)+"`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weight_path)+"`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(meta_path)+"`")
    if netMain is None:
        netMain = load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(meta_path.encode("ascii"))
    if altNames is None:
        # In thon 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(meta_path) as meta_fh:
                meta_contents = meta_fh.read()
                import re
                match = re.search("names *= *(.*)$", meta_contents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as names_fh:
                            names_list = names_fh.read().strip().split("\n")
                            altNames = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass

    color_array = generate_color(meta_path)

    if True:
        rospy.spin()
    else:
        import csv
        import sys
        import ast
        import pandas as pd
        maxInt = sys.maxsize
        print('maxInt', maxInt)
        csv.field_size_limit(maxInt)
        img_file = '/home/docker/dataset/long_office/rgb/2000.csv'
        depth_file = '/home/docker/dataset/long_office/depth/2000.csv'
        # img_file = '/home/docker/dataset/rgbd_dataset_freiburg2_large_no_loop/rgb/2500.csv'
        # depth_file = '/home/docker/dataset/rgbd_dataset_freiburg2_large_no_loop/depth/2500.csv'
        # img_file = '/home/docker/dataset/rgbd_dataset_freiburg2_large_with_loop/rgb/2500.csv'
        # depth_file = '/home/docker/dataset/rgbd_dataset_freiburg2_large_with_loop/depth/2500.csv'
        # img_file = '/home/docker/dataset/rgbd_dataset_freiburg3_walking_xyz_fix/rgb/200.csv'
        # depth_file = '/home/docker/dataset/rgbd_dataset_freiburg3_walking_xyz_fix/depth/200.csv'
        # b = bagreader(filename)
        # img_msg = b.message_by_topic('/camera/rgb/image_color')
        # img_msg
        df_img = pd.read_csv(img_file, engine='python', error_bad_lines=False)
        df_depth = pd.read_csv(depth_file, engine='python', error_bad_lines=False)
        # seq_img = df_img.sample(n = 15)
        # seq_depth = df_depth.iloc[seq_img.index]
        seq_img = df_img
        seq_depth = df_depth

        # for i in seq_img.index:
        while True:
            i = int(input("Enter frame"))
            img_msg = Image()
            img_msg.header.seq = seq_img['header.seq'][i]
            img_msg.header.stamp.secs = seq_img['header.stamp.secs'][i]
            img_msg.header.stamp.nsecs = seq_img['header.stamp.nsecs'][i]
            img_msg.header.frame_id = seq_img['header.frame_id'][i]
            img_msg.height = seq_img['height'][i]
            img_msg.width = seq_img['width'][i]
            img_msg.encoding = seq_img['encoding'][i]
            img_msg.is_bigendian = seq_img['is_bigendian'][i]
            img_msg.step = seq_img['step'][i]
            img_msg.data = ast.literal_eval(seq_img['data'][i])

            depth_msg = Image()
            depth_msg.header.seq = seq_depth['header.seq'][i]
            depth_msg.header.stamp.secs = seq_depth['header.stamp.secs'][i]
            depth_msg.header.stamp.nsecs = seq_depth['header.stamp.nsecs'][i]
            depth_msg.header.frame_id = seq_depth['header.frame_id'][i]
            depth_msg.height = seq_depth['height'][i]
            depth_msg.width = seq_depth['width'][i]
            depth_msg.encoding = seq_depth['encoding'][i]
            depth_msg.is_bigendian = seq_depth['is_bigendian'][i]
            depth_msg.step = seq_depth['step'][i]
            depth_msg.data = ast.literal_eval(seq_depth['data'][i])

            callback(img_msg, depth_msg)
            # time.sleep(10)
    
    out.release()
    cv2.destroyAllWindows()
    print("DOBNE")
    log.info("\nFINISH")


if __name__ == "__main__":
    main(sys.argv[1:])
