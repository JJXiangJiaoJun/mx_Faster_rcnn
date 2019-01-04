import mxnet as mx
from mxnet import gluon, autograd, nd
import os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import gluoncv as gcv
import argparse
from mxnet.gluon.data import DataLoader
import logging
import time
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.accuracy import Accuracy
from mx_Faster_rcnn import faster_rcnn_resnet50_v1b_voc


def parse_args():
    # create a parser object
    parser = argparse.ArgumentParser(description='This is a script train Faster-RCNN end2end.')
    parser.add_argument('--epochs', type=str, default='',
                        help='Training epochs.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None.'
                             'For example,you can resume from ./faster_rcnn_xxx_0123,params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming,default is 0 for new training')
    parser.add_argument('--lr', type=str, default='',
                        help='set learning rate for training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD training')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='learning rate decay rate')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epoches at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay,default if 5e-4 for voc')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval,Default is 100')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval,best model will always be saved. default is 1')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validation model epoch interval.Default is 1')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set')
    args = parser.parse_args()

    args.epochs = int(args.epochs) if args.epochs else 20
    args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14,20'
    args.lr = float(args.lr) if args.lr else 0.001
    args.wd = float(args.wd) if args.wd else 5e-4

    return args


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self, name='RPNAcc', **kwargs):
        super(RPNAccMetric, self).__init__(name, **kwargs)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`  [rpn_label,rpn_weight]
            The labels of the data.

        preds : list of `NDArray`   [rpn_cls_logits]
            Predicted values.
        """
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        num_inst = nd.sum(rpn_weight)
        pred_label = mx.nd.sigmoid(rpn_cls_logits) >= 0.5
        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)
        # calculate acc
        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, name='RPNL1Loss', **kwargs):
        super(RPNL1LossMetric, self).__init__(name, **kwargs)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`   [rpn_box_target,rpn_box_weight]
            The labels of the data.

        preds : list of `NDArray`   [rpn_box_reg]
            Predicted values.
        """
        rpn_box_target, rpn_box_weight = labels
        rpn_box_reg = preds[0]

        num_inst = nd.sum(rpn_box_weight) / 4

        loss = mx.nd.sum(rpn_box_weight * mx.nd.smooth_l1(rpn_box_reg - rpn_box_target, scalar=3))
        # calculate L1Loss
        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, name='RCNNAcc', axis=-1, **kwargs):
        super(RCNNAccMetric, self).__init__(name, **kwargs)
        self._axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`   [rcnn_lable,rcnn_weight]
            The labels of the data.

        preds : list of `NDArray`   [rcnn_cls_logits]
            Predicted values.
        """
        rcnn_label, rcnn_weight = labels
        rcnn_cls_logits = preds[0]

        num_inst = nd.sum(rcnn_weight)

        pred_label = nd.argmax(rcnn_cls_logits, axis=self._axis)

        self.sum_metric += ((pred_label == rcnn_label) * rcnn_weight).sum().asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, name='RCNNL1Loss', **kwargs):
        super(RCNNL1LossMetric, self).__init__(name, **kwargs)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`   [rcnn_box_target,rcnn_box_weight]
            The labels of the data.

        preds : list of `NDArray`   [rcnn_box_reg]
            Predicted values.
        """
        rcnn_box_target, rcnn_box_weight = labels
        rcnn_box_reg = preds[0]

        num_inst = nd.sum(rcnn_box_weight) / 4

        self.sum_metric += (nd.smooth_l1(rcnn_box_reg - rcnn_box_target, scalar=1) * rcnn_box_weight).sum().asscalar()
        self.num_inst += num_inst.asscalar()


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def get_data_set(train_root='./VOCData/train_val/VOCdevkit', test_root='./VOCData/test/VOCdevkit'):
    """
       get PASCAL-VOC train & val & test dataset
    :return: 
        train_dataset :
         
        val_dataset : 
            use test data as valid dataset
    """
    train_dataset = gdata.VOCDetection(root=train_root, splits=[(2007, 'trainval')])
    val_dataset = gdata.VOCDetection(root=test_root, splits=[(2007, 'test')])
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)

    return (train_dataset, val_dataset, val_metric)


def get_data_loader(net, train_dataset, val_dataset, batch_size=1):
    """
        get PASCAL-VOC train&val dataloader
     
    :return: 
    """
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(5)])

    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(FasterRCNNDefaultTrainTransform(net.short, net.max_size, net)),
        batch_size=1, shuffle=True, batchify_fn=train_bfn, last_batch='rollover')
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(FasterRCNNDefaultValTransform(net.short, net.max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep')
    return (train_loader, val_loader)


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    for batch in val_data:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes,
                                                                        gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
    return eval_metric.get()


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params').format(prefix.epoch, current_map))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params().setattr('grad_req', 'write')
    """get trainer"""
    trainer = gluon.Trainer(
        net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'learning_rate': args.lr,
         'wd': args.wd,
         'momentum': args.momentum,
         'clip_gradient': 5})

    rpn_cls_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_l1_loss = gluon.loss.HuberLoss(rho=1 / 9)
    rcnn_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_l1_loss = gluon.loss.HuberLoss()

    # use to store loss during train
    metrics1 = [mx.metric.Loss('RPN_Conf'),
                mx.metric.Loss('RPN_SmoothL1'),
                mx.metric.Loss('RCNN_CrossEntropy'),
                mx.metric.Loss('RCNN_SmoothL1')]
    # use to calculate acc during train
    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_decay = float(args.lr_decay)

    # set logging
    logging.basicConfig(level=logging.NOTSET)
    # create a logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create a file handler
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.mkdir(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info('Trainabel parameters:')
        logger.info(net.collect_train_params().keys())
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    for epoch in range(args.start_epoch, args.epochs):
        btm = time.time()
        ttm = time.time()
        if lr_steps and epoch >= lr_steps[0]:
            current_lr = trainer.learning_rate
            lr_steps.pop(0)
            new_lr = lr_decay * current_lr
            trainer.set_learning_rate(new_lr)
            logger.info('[Epoch {}] Set learning rate to {}'.format(epoch, new_lr))
        """reset metrics"""
        for metric in metrics1:
            metric.reset()

        """get mini-batch traindata"""
        for i, batch in enumerate(train_data):
            batch = split_and_load(batch, ctx_list=ctx)
            batch_size = len(batch[0])
            losses = []
            metric_losses = [[] for _ in metrics1]
            add_losses = [[] for _ in metrics2]
            with autograd.record():
                for data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(*batch):
                    gt_label = label[:, :, 4:5]
                    gt_box = label[:, :, :4]
                    # net forward
                    rcnn_cls_preds, rcnn_box_preds, roi, samples, matches, rpn_score, rpn_box, anchors = net(data,
                                                                                                             gt_box)

                    # generate targets for rcnn
                    rcnn_cls_target, rcnn_box_target, rcnn_box_masks = net.target_generator(roi, samples, matches,
                                                                                            gt_label, gt_box)

                    # calculate loss
                    # rpn loss
                    rpn_score = rpn_score.squeeze(axis=-1)
                    num_rpn_pos = (rpn_cls_targets >= 0).sum()
                    rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets,
                                             (rpn_cls_targets >= 0)) * rpn_cls_targets.size / num_rpn_pos
                    rpn_loss2 = rpn_l1_loss(rpn_box, rpn_box_targets, rpn_box_masks) * rpn_box.size / num_rpn_pos

                    # rcnn loss
                    num_rcnn_pos = (rcnn_cls_target >= 0).sum()
                    rcnn_loss1 = rcnn_cls_loss(rcnn_cls_preds, rcnn_cls_target,
                                               (rcnn_cls_target >= 0)) * rcnn_cls_target.size / num_rcnn_pos
                    rcnn_loss2 = rcnn_l1_loss(rcnn_box_preds, rcnn_box_target,
                                              rcnn_box_masks) * rcnn_box_masks.size / num_rcnn_pos

                    rpn_loss = rpn_loss1 + rpn_loss2
                    rcnn_loss = rcnn_loss1 + rcnn_loss2

                    # autograd.backward([rpn_loss+rcnn_loss])
                    # record metrics
                    losses.append(rpn_loss.sum() + rcnn_loss.sum())
                    metric_losses[0].append(rpn_loss1.sum())
                    metric_losses[1].append(rpn_loss2.sum())
                    metric_losses[2].append(rcnn_loss1.sum())
                    metric_losses[3].append(rcnn_loss2.sum())

                    add_losses[0].append([[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]])
                    add_losses[1].append([[rpn_box_targets, rpn_box_masks], [rpn_box]])
                    add_losses[2].append([[rcnn_cls_target, rcnn_cls_target >= 0], [rcnn_cls_preds]])
                    add_losses[3].append([[rcnn_box_target, rcnn_box_masks], [rcnn_box_preds]])
                # backward
                autograd.backward(losses)
                # update matrics
                for metric, record in zip(metrics1, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(metrics2, add_losses):
                    for pred in records:
                        metric.update(pred[0], pred[1])
            # optimize
            trainer.step(batch_size=batch_size)
            if args.log_interval and not (i + 1) % args.log_interval:
                msg = ','.join(['{}={:3f}'.format(*metric.get()) for metric in metrics1 + metrics2])
                # logging
                log = '[Epoch {}][Batch {}], Speed: {:.3f} samples/sec , {}'.format(
                    epoch, i, args.log_interval * batch_size / (time.time() - btm), msg)
                logger.info(log)
                btm = time.time()
        # end of one epoch
        # logging
        msg = ','.join(['{}={.3f}'.format(*metric.get()) for metric in metrics1])
        log = '[Epoch {}] Training cost :{:.3f},{}'.format(
            epoch, (time.time() - ttm), msg)
        logger.info(log)

        if not (epoch + 1) % args.val_interval:
            # validation the model
            # TODO(Qinghong Zeng)
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # training contexts
    net_name = 'faster_rccn_voc'
    args.save_prefix += net_name
    # network

    net = faster_rcnn_resnet50_v1b_voc(pretrained=False, pretrained_base=True)

    if args.resume.strip():
        net.load_parameters(filename=args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()

    net.collect_params().reset_ctx(ctx)
    # train data
    train_dataset, val_dataset, val_metric = get_data_set()
    # train loader
    train_loader, test_loader = get_data_loader(net, train_dataset, val_dataset, batch_size=1)

    # train
    train(net, train_loader, test_loader, val_metric, ctx, args)
