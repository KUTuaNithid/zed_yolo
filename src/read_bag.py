#!/usr/bin/env python3
import bagpy
from bagpy import bagreader
import pandas as pd
from sensor_msgs.msg import Image

filename = '/home/docker/dataset/rgbd_dataset_freiburg2_large_no_loop.bag'
b = bagreader(filename)
img_msg = b.message_by_topic('/camera/rgb/image_color')
# img_msg = b.message_by_topic('/camera/depth/image')
# img_msg
# df_img = pd.read_csv(csv_file)
# seq_img = df_img[0:10]

# for img in seq_img:
#     img_msg = Image()
#     img_msg.header.seq = img['header.seq']
#     img_msg.header.stamp.secs = img['header.stamp.secs']
#     img_msg.header.stamp.nsecs = img['header.stamp.nsecs']
#     img_msg.header.frame_id = img['header.frame_id']
#     img_msg.height = img['height']
#     img_msg.width = img['width']
#     img_msg.encoding = img['encoding']
#     img_msg.is_bigendian = img['is_bigendian']
#     img_msg.step = img['step']
#     img_msg.data = img['data']

#     img_pub.publish(img_msg_frame)


