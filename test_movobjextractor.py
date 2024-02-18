import cv2
import gflags
import numpy as np
import os
import sys
import scipy.io as sio
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from models.adversarial_learner import AdversarialLearner
from models.utils.general_utils import postprocess_mask, postprocess_image, compute_boundary_score
from common_flags import FLAGS
from test_generator import compute_IoU, compute_mae
from data.davis2016_data_utils import Davis2016Reader
from data.fbms_data_utils import FBMS59Reader
from data.segtrackv2_data_utils import SegTrackV2Reader


des_width = 640
des_height = 384
mask_threshold = 0.6


class DataLoader:
    def __init__(self, config):
        self.config = config

    def read_data(self):
        with tf.name_scope("data_loading"):
            if self.config.dataset == 'DAVIS2016':
                reader = Davis2016Reader(self.config.root_dir, num_threads=1)
                test_batch, test_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                           t_len=self.config.test_temporal_shift,
                                                           with_fname=True,
                                                           test_crop=self.config.test_crop,
                                                           partition=self.config.test_partition)

            elif self.config.dataset == 'FBMS':
                reader = FBMS59Reader(self.config.root_dir)
                test_batch, test_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                           test_crop=self.config.test_crop,
                                                           t_len=self.config.test_temporal_shift,
                                                           with_fname=True,
                                                           partition=self.config.test_partition)
            elif self.config.dataset == 'SEGTRACK':
                reader = SegTrackV2Reader(self.config.root_dir, num_threads=1)
                test_batch, test_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                           test_crop=self.config.test_crop,
                                                           t_len=self.config.test_temporal_shift,
                                                           with_fname=True)
            else:
                raise IOError("Dataset should be DAVIS2016 / FBMS / SEGTRACK")

            print(f"[INFO] batch_size in DataLoader: {self.config.batch_size}")
            print(f"[INFO] test_batch : {test_batch}")

            image_batch, gt_mask_batch, fname_batch = test_batch[0], test_batch[2], test_batch[3]
            print(
                f"[INFO] image_batch shape before reshape: {image_batch.shape}")

        # reshape
        image_batch = tf.image.resize_images(image_batch, [self.config.img_height,
                                                           self.config.img_width])
        gt_mask_batch = tf.image.resize_images(gt_mask_batch, [self.config.img_height,
                                                               self.config.img_width],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # set variables.
        self.image_batch = image_batch
        self.fname_batch = fname_batch
        self.test_samples = reader.val_samples
        self.gt_masks = gt_mask_batch
        self.test_iter = test_iter

        print(f"[INFO] image_batch shape: {self.image_batch.shape}")

    def get_data(self, sess):
        # transform tensor to numpy array
        fetches = {'image_batch': self.image_batch,
                   'gt_masks': self.gt_masks, 'fname_batch': self.fname_batch}
        return sess.run(fetches)


def _test_masks():
    CategoryIou = {}
    CategoryMae = {}

    data_loader = DataLoader(FLAGS)
    data_loader.read_data()

    print(f"[INFO] batch_size: {FLAGS.batch_size}")

    # we need session to transform data loader's output tensor to numpy array.
    sv = tf.train.Supervisor(logdir=FLAGS.test_save_dir,
                             save_summaries_secs=0,
                             saver=None)
    with sv.managed_session() as sess:
        sess.run(data_loader.test_iter.initializer)
        n_steps = int(
            np.ceil(data_loader.test_samples / float(FLAGS.batch_size)))
        progbar = Progbar(target=n_steps)

        num_processed_frames = 0
        for step in range(n_steps):
            try:
                data = data_loader.get_data(sess)
                # inference = my own result
                pass
            except tf.errors.OutOfRangeError:
                print("End of testing dataset")
                break

            # Now write images in the test folder
            for batch_num in range(FLAGS.batch_size):
                # select mask
                # this should be inference['gen_masks'][batch_num]
                # generated_mask.shape = [row=192,col=384,channel=1], dtype=fload32.
                generated_mask = data['gt_masks'][batch_num]
                gt_mask = data['gt_masks'][batch_num]
                category = data['fname_batch'][batch_num].decode(
                    "utf-8").split('/')[-2]

                # debug. display each mask
                cv2.imshow('gt_mask', gt_mask)
                cv2.imshow('generated_mask', generated_mask)
                cv2.waitKey(0)

                iou, out_mask = compute_IoU(
                    gt_mask=gt_mask, pred_mask_f=generated_mask)
                mae = compute_mae(gt_mask=gt_mask, pred_mask_f=out_mask)
                try:
                    CategoryIou[category].append(iou)
                    CategoryMae[category].append(mae)
                except:
                    CategoryIou[category] = [iou]
                    CategoryMae[category] = [mae]

                if FLAGS.generate_visualization:
                    # Verbose image generation
                    save_dir = os.path.join(FLAGS.test_save_dir, category)
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    filename = os.path.join(save_dir,
                                            "frame_{:08d}.png".format(len(CategoryIou[category])))

                    preprocessed_bgr = postprocess_image(
                        data['image_batch'][batch_num])
                    preprocessed_mask = postprocess_mask(out_mask)

                    # Overlap images
                    results = cv2.addWeighted(preprocessed_bgr, 0.5,
                                              preprocessed_mask, 0.4, 0)
                    results = cv2.resize(results, (des_width, des_height))

                    cv2.imwrite(filename, results)

                    matlab_fname = os.path.join(save_dir,
                                                'result_{}.mat'.format(len(CategoryIou[category])))
                    sio.savemat(matlab_fname,
                                {'img': cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB),
                                 'pred_mask': out_mask,
                                 'gt_mask': gt_mask})
                num_processed_frames += 1

            progbar.update(step)

        # save final result to a text file
        tot_ious = 0
        tot_maes = 0
        per_cat_iou = []
        with open(os.path.join(FLAGS.test_save_dir, 'result.txt'), 'w') as f:
            for cat, list_iou in CategoryIou.items():
                print("Category {}: IoU is {} and MAE is {}".format(
                    cat, np.mean(list_iou), np.mean(CategoryMae[cat])), file=f)
                tot_ious += np.sum(list_iou)
                tot_maes += np.sum(CategoryMae[cat])
                per_cat_iou.append(np.mean(list_iou))
            print("The Average over the dataset: IoU is {} and MAE is {}".format(
                tot_ious/float(num_processed_frames), tot_maes/float(num_processed_frames)), file=f)
            print("The Average over sequences IoU is {}".format(
                np.mean(per_cat_iou)), file=f)
            print("Success: Processed {} frames".format(
                num_processed_frames), file=f)


def main(argv):
    # Utility main to load flags
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _test_masks()


if __name__ == "__main__":
    main(sys.argv)
