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
from models.utils.flow_utils import flow_to_image

from common_flags import FLAGS

des_width = 640
des_height = 384
mask_threshold = 0.6


def compute_IoU(gt_mask, pred_mask_f, threshold=0.1):
    gt_mask = gt_mask.astype(np.bool)
    pred_mask = pred_mask_f > threshold
    pred_mask_compl = np.logical_not(pred_mask)

    # Compute the score to disambiguate foreground from background
    boundary_score = compute_boundary_score(pred_mask)
    if boundary_score < mask_threshold:
        annotation = pred_mask
    else:
        annotation = pred_mask_compl

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(gt_mask), 0):
        return 1, annotation
    else:
        return np.sum((annotation & gt_mask)) / \
            np.sum((annotation | gt_mask), dtype=np.float32), annotation


def compute_mae(gt_mask, pred_mask_f):
    mae = np.mean(np.abs(gt_mask - pred_mask_f))
    return mae


def _test_masks():
    learner = AdversarialLearner()
    learner.setup_inference(FLAGS, aug_test=False)
    saver = tf.train.Saver([var for var in tf.trainable_variables()])
    CategoryIou = {}
    CategoryMae = {}
    # manages multi-threading
    sv = tf.train.Supervisor(logdir=FLAGS.test_save_dir,
                             save_summaries_secs=0,
                             saver=None)
    with sv.managed_session() as sess:
        checkpoint = FLAGS.ckpt_file
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("Resume model from checkpoint {}".format(checkpoint))
        else:
            raise IOError("Checkpoint file not found")

        sess.run(learner.test_iterator.initializer)

        n_steps = int(np.ceil(learner.test_samples / float(FLAGS.batch_size)))

        progbar = Progbar(target=n_steps)

        i = 0

        for step in range(n_steps):
            if sv.should_stop():
                break
            try:
                inference = learner.inference(sess)
            except tf.errors.OutOfRangeError:
                print("End of testing dataset")  # ==> "End of dataset"
                break
            # Now write images in the test folder
            for batch_num in range(inference['input_image'].shape[0]):

                # select mask
                generated_mask = inference['gen_masks'][batch_num]
                gt_mask = inference['gt_masks'][batch_num]
                category = inference['img_fname'][batch_num].decode(
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
                        inference['input_image'][batch_num])
                    preprocessed_mask = postprocess_mask(out_mask)

                    # save flow image. predicted flow is named gt_flow.
                    estim_flow_batch = inference['gt_flow']
                    estim_flow = flow_to_image(estim_flow_batch)[batch_num]
                    estim_flow = cv2.resize(
                        estim_flow, (des_width, des_height))
                    cv2.imwrite(os.path.join(save_dir, 'flow_{:08d}.png'.format(
                        len(CategoryIou[category]))), estim_flow)

                    # Overlap images
                    results = cv2.addWeighted(preprocessed_bgr, 0.5,
                                              preprocessed_mask, 0.4, 0)
                    results = cv2.resize(results, (des_width, des_height))

                    cv2.imwrite(filename, results)

                    matlab_fname = os.path.join(save_dir,
                                                'result_{}.mat'.format(len(CategoryIou[category])))
                    sio.savemat(matlab_fname,
                                {'flow': inference['gt_flow'][batch_num],
                                 'img1': cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB),
                                 # inference['gen_masks'][batch_num],
                                 'pred_mask': out_mask,
                                 'gt_mask': inference['gt_masks'][batch_num]})
                i += 1

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
                tot_ious/float(i), tot_maes/float(i)), file=f)
            print("The Average over sequences IoU is {}".format(
                np.mean(per_cat_iou)), file=f)
            print("Success: Processed {} frames".format(i), file=f)


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
