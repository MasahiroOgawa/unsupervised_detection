from __future__ import division
import tensorflow as tf


def ClipIfNotNone(grad, clipvalue):
    return tf.clip_by_value(grad, -clipvalue, clipvalue)


def randomize(grad, clipvalue):
    shape = grad.get_shape().as_list()
    new_grad = tf.random_uniform(shape=shape, minval=-clipvalue, maxval=clipvalue)
    return new_grad


def train_op(loss, var_list, optimizer, gradient_clip_value=0.1, can_change=False):
    """
    Train Operation. If the can_change flag is active, then random noise is
    added to the gradients. This is important for the generator networks, since
    it helps going out of the local minimima of masking everything or nothing.
    """
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    if can_change:
        grad_avg_value = [tf.reduce_mean(tf.abs(grad)) for grad, _ in grads_and_vars]
        grad_avg_value = tf.reduce_mean(grad_avg_value)
        should_change = grad_avg_value < 0.00001
        clipped_grad_and_vars = [
            (
                tf.cond(
                    should_change,
                    lambda: tf.abs(randomize(grad, gradient_clip_value)),
                    lambda: ClipIfNotNone(grad, gradient_clip_value),
                ),
                var,
            )
            for grad, var in grads_and_vars
        ]
    else:
        clipped_grad_and_vars = [
            (ClipIfNotNone(grad, gradient_clip_value), var)
            for grad, var in grads_and_vars
        ]
    train_operation = optimizer.apply_gradients(clipped_grad_and_vars)

    return train_operation, clipped_grad_and_vars


def charbonnier_loss(gt_flows, pred_flows, masks, cbn=0.5):
    """
    This function computes the charbonnier loss between the flow predicted by
    PWC and the flow recovered after masking.
    Args:
        gt_flow: Optical Flow between the two images (Generated with PWC)
        pred_flow: Optical Flow recovered after masking (Generated by recover net)
        masks: The masks used to mask gt_flow (Generated by generator net)
        cbn: The Charbonnier Factor. 0.5 for L1 loss and 1.0 for L2 loss.
    Returns:
        error_sum: Vector of shape [batch_size,], where each element is the loss
                   summed over all pixels for a member of the batch.
    """
    epsilon = 0.001
    lp_loss = tf.square(gt_flows - pred_flows) + tf.square(epsilon)
    lp_loss = tf.pow(lp_loss, cbn)
    lp_loss = lp_loss * masks
    return tf.reduce_sum(lp_loss, axis=[1, 2, 3])  # [B,]
