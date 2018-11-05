import rospy
import message_filters
import cv2
import numpy as np
from fcn.config import cfg
from fcn.test import _extract_vertmap
from utils.blob import im_list_to_blob, pad_im, unpad_im, add_noise
from normals import gpu_normals
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
# from synthesizer.msg import PoseCNNMsg
from posecnn_kinect.msg import PoseCNNMsg

# def vis_segmentations_vertmaps_detection(im, im_depth, im_labels, colors, center_map,
#   labels, rois, poses, poses_new, intrinsic_matrix, num_classes, classes, points):
#     """Visual debugging of detections."""
#     import matplotlib.pyplot as plt
#     from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#
#     fig = plt.figure()
#     canvas = FigureCanvas(fig)
#
#     # show image
#     ax = fig.add_subplot(3, 3, 1)
#     im = im[:, :, (2, 1, 0)]
#     im = im.astype(np.uint8)
#     plt.imshow(im)
#     ax.set_title('input image')
#
#     # canvas.draw()
#     # return np.fromstring(canvas.tostring_rgb(), dtype='uint8')
#     # show depth
#     ax = fig.add_subplot(3, 3, 2)
#     plt.imshow(im_depth)
#     ax.set_title('input depth')
#
#     # show class label
#     ax = fig.add_subplot(3, 3, 3)
#     plt.imshow(im_labels)
#     ax.set_title('class labels')
#
#     if cfg.TEST.VERTEX_REG_2D:
#         # show centers
#         for i in xrange(rois.shape[0]):
#             if rois[i, 1] == 0:
#                 continue
#             cx = (rois[i, 2] + rois[i, 4]) / 2
#             cy = (rois[i, 3] + rois[i, 5]) / 2
#             w = rois[i, 4] - rois[i, 2]
#             h = rois[i, 5] - rois[i, 3]
#             if not np.isinf(cx) and not np.isinf(cy):
#                 plt.plot(cx, cy, 'yo')
#
#                 # show boxes
#                 plt.gca().add_patch(
#                     plt.Rectangle((cx-w/2, cy-h/2), w, h, fill=False,
#                                    edgecolor='g', linewidth=3))
#
#     # show vertex map
#     ax = fig.add_subplot(3, 3, 4)
#     plt.imshow(center_map[:,:,0])
#     ax.set_title('centers x')
#
#     ax = fig.add_subplot(3, 3, 5)
#     plt.imshow(center_map[:,:,1])
#     ax.set_title('centers y')
#
#     ax = fig.add_subplot(3, 3, 6)
#     plt.imshow(center_map[:,:,2])
#     ax.set_title('centers z')
#
#     # show projection of the poses
#     if cfg.TEST.POSE_REG:
#
#         ax = fig.add_subplot(3, 3, 7, aspect='equal')
#         plt.imshow(im)
#         ax.invert_yaxis()
#         for i in xrange(rois.shape[0]):
#             cls = int(rois[i, 1])
#             if cls > 0:
#                 # extract 3D points
#                 x3d = np.ones((4, points.shape[1]), dtype=np.float32)
#                 x3d[0, :] = points[cls,:,0]
#                 x3d[1, :] = points[cls,:,1]
#                 x3d[2, :] = points[cls,:,2]
#
#                 # projection
#                 RT = np.zeros((3, 4), dtype=np.float32)
#                 RT[:3, :3] = quat2mat(poses[i, :4])
#                 RT[:, 3] = poses[i, 4:7]
#                 print classes[cls]
#                 print RT
#                 print '\n'
#                 x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
#                 x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
#                 x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
#                 plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.5)
#                 # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)
#
#         ax.set_title('projection of model points')
#         ax.invert_yaxis()
#         ax.set_xlim([0, im.shape[1]])
#         ax.set_ylim([im.shape[0], 0])
#
#         if cfg.TEST.POSE_REFINE:
#             ax = fig.add_subplot(3, 3, 8, aspect='equal')
#             plt.imshow(im)
#             ax.invert_yaxis()
#             for i in xrange(rois.shape[0]):
#                 cls = int(rois[i, 1])
#                 if cls > 0:
#                     # extract 3D points
#                     x3d = np.ones((4, points.shape[1]), dtype=np.float32)
#                     x3d[0, :] = points[cls,:,0]
#                     x3d[1, :] = points[cls,:,1]
#                     x3d[2, :] = points[cls,:,2]
#
#                     # projection
#                     RT = np.zeros((3, 4), dtype=np.float32)
#                     RT[:3, :3] = quat2mat(poses_new[i, :4])
#                     RT[:, 3] = poses_new[i, 4:7]
#                     print cls
#                     print RT
#                     print '\n'
#                     x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
#                     x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
#                     x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
#                     plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)
#
#             ax.set_title('projection refined by ICP')
#             ax.invert_yaxis()
#             ax.set_xlim([0, im.shape[1]])
#             ax.set_ylim([im.shape[0], 0])
#
#     plt.show()

class ImageListener:

    def __init__(self, sess, network, imdb, meta_data, cfg):

        self.sess = sess
        self.net = network
        self.imdb = imdb
        self.meta_data = meta_data
        self.cfg = cfg
        self.cv_bridge = CvBridge()
        self.count = 0
        # self.axs = self.create_plots()

        # initialize a node
        rospy.init_node("image_listener")
        self.posecnn_pub = rospy.Publisher('posecnn_result', PoseCNNMsg, queue_size=1)
        self.label_pub = rospy.Publisher('posecnn_label', Image, queue_size=1)
        self.center_pub = rospy.Publisher('posecnn_center', Image, queue_size=1)
        rgb_sub = message_filters.Subscriber('/camera/rgb/image_color', Image, queue_size=2)
        # depth_sub = message_filters.Subscriber('/camera/depth_registered/image', Image, queue_size=2)
        depth_sub = message_filters.Subscriber('/camera/depth_registered/sw_registered/image_rect_raw', Image, queue_size=2)

        queue_size = 1
        slop_seconds = 0.025
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)

    def callback(self, rgb, depth):
        # print "test"
        if depth.encoding == '32FC1':
            depth_32 = self.cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        # write images
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        filename = 'images/%06d-color.png' % self.count
        cv2.imwrite(filename, im)

        filename = 'images/%06d-depth.png' % self.count
        cv2.imwrite(filename, depth_cv)
        # print filename
        self.count += 1

        if (self.cfg.TEST.VERTEX_REG_2D and self.cfg.TEST.POSE_REFINE) or (self.cfg.TEST.VERTEX_REG_3D and self.cfg.TEST.POSE_REG):
            import libsynthesizer
            synthesizer = libsynthesizer.Synthesizer(self.cfg.CAD, self.cfg.POSE)
            synthesizer.setup(self.cfg.TRAIN.SYN_WIDTH, self.cfg.TRAIN.SYN_HEIGHT)

        # run network
        labels, probs, vertex_pred, rois, poses = self.im_segment_single_frame(self.sess, self.net, im, depth_cv, self.meta_data, \
            self.imdb._extents, self.imdb._points_all, self.imdb._symmetry, self.imdb.num_classes)

        im_label = self.imdb.labels_to_image(im, labels)

        # added by Aditya
        labels = unpad_im(labels, 16)
        im_scale = self.cfg.TEST.SCALES_BASE[0]
        im_depth = depth_cv

        poses_new = []
        poses_icp = []
        if self.cfg.TEST.VERTEX_REG_2D:
            if self.cfg.TEST.POSE_REG:
                # pose refinement
                fx = self.meta_data['intrinsic_matrix'][0, 0] * im_scale
                fy = self.meta_data['intrinsic_matrix'][1, 1] * im_scale
                px = self.meta_data['intrinsic_matrix'][0, 2] * im_scale
                py = self.meta_data['intrinsic_matrix'][1, 2] * im_scale
                factor = self.meta_data['factor_depth']
                znear = 0.25
                zfar = 6.0
                poses_new = np.zeros((poses.shape[0], 7), dtype=np.float32)
                poses_icp = np.zeros((poses.shape[0], 7), dtype=np.float32)
                error_threshold = 0.01
                if self.cfg.TEST.POSE_REFINE:
                    labels_icp = labels.copy();
                    rois_icp = rois
                    if self.imdb.num_classes == 2:
                        I = np.where(labels_icp > 0)
                        labels_icp[I[0], I[1]] = self.imdb._cls_index
                        rois_icp = rois.copy()
                        rois_icp[:, 1] = self.imdb._cls_index
                    im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

                    parameters = np.zeros((7, ), dtype=np.float32)
                    parameters[0] = fx
                    parameters[1] = fy
                    parameters[2] = px
                    parameters[3] = py
                    parameters[4] = znear
                    parameters[5] = zfar
                    parameters[6] = factor

                    height = labels_icp.shape[0]
                    width = labels_icp.shape[1]
                    num_roi = rois_icp.shape[0]
                    channel_roi = rois_icp.shape[1]
                    synthesizer.icp_python(labels_icp, im_depth, parameters, height, width, num_roi, channel_roi, \
                                           rois_icp, poses, poses_new, poses_icp, error_threshold)


        if self.cfg.TEST.VISUALIZE:
            vertmap = _extract_vertmap(labels, vertex_pred, self.imdb._extents, self.imdb.num_classes)
            # vis_segmentations_vertmaps_detection(im, im_depth, im_label, self.imdb._class_colors, vertmap,
                # labels, rois, poses, poses_icp, self.meta_data['intrinsic_matrix'], self.imdb.num_classes, self.imdb._classes, self.imdb._points_all)

        im_center = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
        im_center.fill(255)
        if cfg.TEST.VERTEX_REG_2D:
            # show centers
            for i in xrange(rois.shape[0]):
                if rois[i, 1] == 0:
                    continue
                cx = (rois[i, 2] + rois[i, 4]) / 2
                cy = (rois[i, 3] + rois[i, 5]) / 2
                w = rois[i, 4] - rois[i, 2]
                h = rois[i, 5] - rois[i, 3]
                if not np.isinf(cx) and not np.isinf(cy):
                    # plt.plot(cx, cy, 'yo')
                    cv2.circle(im_center, (int(cx), int(cy)), 5, (0,0,255))
                    cv2.rectangle(im_center, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (0,255,0))

                    # show boxes
                    # plt.gca().add_patch(
                        # plt.Rectangle((cx-w/2, cy-h/2), w, h, fill=False,
                                       # edgecolor='g', linewidth=3))
        # publish
        # print im_center.shape[0]
        msg = PoseCNNMsg()
        msg.height = int(im.shape[0])
        msg.width = int(im.shape[1])
        msg.roi_num = int(rois.shape[0])
        msg.roi_channel = int(rois.shape[1])
        msg.fx = float(self.meta_data['intrinsic_matrix'][0, 0])
        msg.fy = float(self.meta_data['intrinsic_matrix'][1, 1])
        msg.px = float(self.meta_data['intrinsic_matrix'][0, 2])
        msg.py = float(self.meta_data['intrinsic_matrix'][1, 2])
        msg.factor = float(self.meta_data['factor_depth'])
        msg.znear = float(0.25)
        msg.zfar = float(6.0)
        msg.label = self.cv_bridge.cv2_to_imgmsg(labels.astype(np.uint8), 'mono8')
        msg.center = self.cv_bridge.cv2_to_imgmsg(im_center)
        msg.depth = self.cv_bridge.cv2_to_imgmsg(depth_cv, 'mono16')
        msg.rois = rois.astype(np.float32).flatten().tolist()
        msg.poses = poses.astype(np.float32).flatten().tolist()
        self.posecnn_pub.publish(msg)

        label_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        label_msg.header.stamp = rospy.Time.now()
        label_msg.header.frame_id = rgb.header.frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)

        center_msg = self.cv_bridge.cv2_to_imgmsg(im_center)
        center_msg.header.stamp = rospy.Time.now()
        center_msg.header.frame_id = rgb.header.frame_id
        center_msg.encoding = 'rgb8'
        self.center_pub.publish(center_msg)


    def create_plots(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()

        # show image
        ax1 = fig.add_subplot(3, 3, 1)
        # im = im[:, :, (2, 1, 0)]
        # im = im.astype(np.uint8)
        # plt.imshow(im)
        ax1.set_title('input image')

        # show depth
        ax2 = fig.add_subplot(3, 3, 2)
        # plt.imshow(im_depth)
        ax2.set_title('input depth')

        # show class label
        ax3 = fig.add_subplot(3, 3, 3)
        # plt.imshow(im_labels)
        ax3.set_title('class labels')

            # show vertex map
        ax4 = fig.add_subplot(3, 3, 4)
        # plt.imshow(center_map[:,:,0])
        ax4.set_title('centers x')

        ax5 = fig.add_subplot(3, 3, 5)
        # plt.imshow(center_map[:,:,1])
        ax5.set_title('centers y')

        ax6 = fig.add_subplot(3, 3, 6)
        # plt.imshow(center_map[:,:,2])
        ax6.set_title('centers z')

        ax7 = fig.add_subplot(3, 3, 7, aspect='equal')
        # plt.imshow(im)
        ax7.invert_yaxis()

        # plt.ion()
        # plt.show()

        return [ax1, ax2, ax3, ax4, ax5, ax6, ax7]




    def get_image_blob(self, im, im_depth, meta_data):
        """Converts an image into a network input.

        Arguments:
            im (ndarray): a color image in BGR order

        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
               in the image pyramid
        """

        # RGB
        im_orig = im.astype(np.float32, copy=True)
        # mask the color image according to depth
        if self.cfg.EXP_DIR == 'rgbd_scene':
            I = np.where(im_depth == 0)
            im_orig[I[0], I[1], :] = 0

        processed_ims_rescale = []
        im_scale = self.cfg.TEST.SCALES_BASE[0]
        im_rescale = cv2.resize(im_orig / 127.5 - 1, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_rescale.append(im_rescale)

        im_orig -= self.cfg.PIXEL_MEANS
        processed_ims = []
        im_scale_factors = []
        assert len(self.cfg.TEST.SCALES_BASE) == 1

        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

        # depth
        im_orig = im_depth.astype(np.float32, copy=True)
        # im_orig = im_orig / im_orig.max() * 255
        im_orig = np.clip(im_orig / 2000.0, 0, 1) * 255
        im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
        im_orig -= self.cfg.PIXEL_MEANS

        processed_ims_depth = []
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_depth.append(im)

        if cfg.INPUT == 'NORMAL':
            # meta data
            K = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            # normals
            depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
            nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, 20.0, cfg.GPU_ID)
            im_normal = 127.5 * nmap + 127.5
            im_normal = im_normal.astype(np.uint8)
            im_normal = im_normal[:, :, (2, 1, 0)]
            im_normal = cv2.bilateralFilter(im_normal, 9, 75, 75)

            processed_ims_normal = []
            im_orig = im_normal.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_normal = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            processed_ims_normal.append(im_normal)
            blob_normal = im_list_to_blob(processed_ims_normal, 3)
        else:
            blob_normal = []

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims, 3)
        blob_rescale = im_list_to_blob(processed_ims_rescale, 3)
        blob_depth = im_list_to_blob(processed_ims_depth, 3)

        return blob, blob_rescale, blob_depth, blob_normal, np.array(im_scale_factors)


    def im_segment_single_frame(self, sess, net, im, im_depth, meta_data, extents, points, symmetry, num_classes):
        """segment image
        """

        # compute image blob
        im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = self.get_image_blob(im, im_depth, meta_data)
        im_scale = im_scale_factors[0]

        # construct the meta data
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        mdata = np.zeros(48, dtype=np.float32)
        mdata[0:9] = K.flatten()
        mdata[9:18] = Kinv.flatten()
        # mdata[18:30] = pose_world2live.flatten()
        # mdata[30:42] = pose_live2world.flatten()
        meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
        meta_data_blob[0,0,0,:] = mdata

        # use a fake label blob of ones
        height = int(im_depth.shape[0] * im_scale)
        width = int(im_depth.shape[1] * im_scale)
        label_blob = np.ones((1, height, width), dtype=np.int32)

        pose_blob = np.zeros((1, 13), dtype=np.float32)
        vertex_target_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)
        vertex_weight_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)

        # forward pass
        if self.cfg.INPUT == 'RGBD':
            data_blob = im_blob
            data_p_blob = im_depth_blob
        elif self.cfg.INPUT == 'COLOR':
            data_blob = im_blob
        elif self.cfg.INPUT == 'DEPTH':
            data_blob = im_depth_blob
        elif self.cfg.INPUT == 'NORMAL':
            data_blob = im_normal_blob

        if self.cfg.INPUT == 'RGBD':
            if self.cfg.TEST.VERTEX_REG_2D or self.cfg.TEST.VERTEX_REG_3D:
                feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                             net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                             net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.poses: pose_blob}
            else:
                feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}
        else:
            if self.cfg.TEST.VERTEX_REG_2D or self.cfg.TEST.VERTEX_REG_3D:
                feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                             net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                             net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}
            else:
                feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}

        sess.run(net.enqueue_op, feed_dict=feed_dict)

        if self.cfg.TEST.VERTEX_REG_2D:
            if self.cfg.TEST.POSE_REG:
                labels_2d, probs, vertex_pred, rois, poses_init, poses_pred = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), \
                              net.get_output('rois'), net.get_output('poses_init'), net.get_output('poses_tanh')])

                # non-maximum suppression
                # keep = nms(rois, 0.5)
                # rois = rois[keep, :]
                # poses_init = poses_init[keep, :]
                # poses_pred = poses_pred[keep, :]
                # print rois

                # combine poses
                num = rois.shape[0]
                poses = poses_init
                for i in xrange(num):
                    class_id = int(rois[i, 1])
                    if class_id >= 0:
                        poses[i, :4] = poses_pred[i, 4*class_id:4*class_id+4]
            else:
                labels_2d, probs, vertex_pred, rois, poses = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'), net.get_output('poses_init')])
                # print rois
                # print rois.shape
                # non-maximum suppression
                # keep = nms(rois[:, 2:], 0.5)
                # rois = rois[keep, :]
                # poses = poses[keep, :]

                #labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
                #vertex_pred = []
                #rois = []
                #poses = []
            vertex_pred = vertex_pred[0, :, :, :]
        else:
            labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
            vertex_pred = []
            rois = []
            poses = []

        return labels_2d[0,:,:].astype(np.int32), probs[0,:,:,:], vertex_pred, rois, poses
