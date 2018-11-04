"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

from mpl_toolkits.mplot3d import Axes3D

flags.DEFINE_string('t_img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string('q_img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(t_img_path, q_img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    input_t_img, proc_param, t_img = preprocess_image(t_img_path, json_path)
    input_q_img, proc_param, q_img = preprocess_image(q_img_path, json_path)
    # Add batch dimension: 1 x D x D x 3
    input_t_img = np.expand_dims(input_t_img , 0)
    input_q_img = np.expand_dims(input_q_img , 0)

    t_joints, t_verts, t_cams, t_joints3d, t_theta = model.predict(
        input_t_img, get_theta=True)
    q_joints, q_verts, q_cams, q_joints3d, q_theta = model.predict(
        input_q_img, get_theta=True)

    print("Joints shape :" + str(t_joints3d.shape))
    print("Joints shape 3d:" + str(t_joints3d.shape))
    print("Joints shape 0 :" + str(t_joints3d.shape[0]))
    print("Joints shape 1 :" + str(t_joints3d.shape[1]))
    print("Joints shape 2 :" + str(t_joints3d.shape[2]))
    visualize_3d3(q_img, proc_param, q_joints[0], q_verts[0], q_cams[0], q_joints3d,
            t_img, proc_param, t_joints[0], t_verts[0], t_cams[0], t_joints3d)


def visualize_3d(img, proc_param, joints, verts, cam, joints3d):
    """
    Renders the result in original image coordinate frame.
    """
    import matplotlib.pyplot as plt

    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    print("Joints shape 3d:" + str(joints3d.shape))
    ax = vis_util.draw_skeleton_3d(joints3d, ax)
    #plt = vis_util.draw_skeleton_3d(img, joints_orig, plt)
    
    ax1 = fig.add_subplot(122)
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    ax1.imshow(skel_img)
    # plt.ion()
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()


def visualize_3d2(img_t, proc_param_t, joints_t, verts_t, cam_t, joints3d_t, 
        img_q, proc_param_q, joints_q, verts_q, cam_q, joints3d_q):
    """
    Renders the result in original image coordinate frame.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()

    ## Pose 1
    cam_for_render_t, vert_shifted_t, joints_orig_t = vis_util.get_original(
        proc_param_t, verts_t, cam_t, joints_t, img_size=img_t.shape[:2])
    # Render results
    ax_3d_t = fig.add_subplot(221, projection='3d')
    ax_3d_t = vis_util.draw_skeleton_3d(joints3d_t, ax_3d_t)
    ax_img_t = fig.add_subplot(222)
    skel_img_t = vis_util.draw_skeleton(img_t, joints_orig_t)
    ax_img_t.imshow(skel_img_t)

    ## Pose 2
    cam_for_render_q, vert_shifted_q, joints_orig_q = vis_util.get_original(
        proc_param_q, verts_q, cam_q, joints_q, img_size=img_q.shape[:2])
    # Render results
    ax_3d_q = fig.add_subplot(223, projection='3d')
    ax_3d_q = vis_util.draw_skeleton_3d(joints3d_q, ax_3d_q)
    ax_img_q = fig.add_subplot(224)
    skel_img_q = vis_util.draw_skeleton(img_q, joints_orig_q)
    ax_img_q.imshow(skel_img_q)

    plt.draw()
    plt.show()

def visualize_3d3(img_t, proc_param_t, joints_t, verts_t, cam_t, joints3d_t, 
        img_q, proc_param_q, joints_q, verts_q, cam_q, joints3d_q):
    """
    Renders the result in original image coordinate frame.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure()
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')

    cam_for_render_t, vert_shifted_t, joints_orig_t = vis_util.get_original(
        proc_param_t, verts_t, cam_t, joints_t, img_size=img_t.shape[:2])
    # Render results
    ax_3d = vis_util.draw_skeleton_3d(joints3d_t, ax_3d, 'b')
    ax_3d = vis_util.draw_skeleton_3d(joints3d_q, ax_3d, 'g')
    ax_3d = vis_util.draw_displacement(joints3d_t, joints3d_q, ax_3d)
    
    print ("Cam param :" + str(cam_t))
    print ("Cam shape :" + str(cam_t.shape))
    ax_3d = vis_util.draw_arrow(cam_t, [0, 0, 0], ax_3d)

    ## Pose 1
    cam_for_render_t, vert_shifted_t, joints_orig_t = vis_util.get_original(
        proc_param_t, verts_t, cam_t, joints_t, img_size=img_t.shape[:2])
    # Render results
    ax_img_t = fig.add_subplot(gs[0, 1])
    skel_img_t = vis_util.draw_skeleton(img_t, joints_orig_t)
    ax_img_t.imshow(skel_img_t)

    ## Pose 2
    cam_for_render_q, vert_shifted_q, joints_orig_q = vis_util.get_original(
        proc_param_q, verts_q, cam_q, joints_q, img_size=img_q.shape[:2])
    # Render results
    ax_img_q = fig.add_subplot(gs[1, 1])
    skel_img_q = vis_util.draw_skeleton(img_q, joints_orig_q)
    ax_img_q.imshow(skel_img_q)

    plt.draw()
    plt.show()

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.t_img_path, config.q_img_path, config.json_path)
