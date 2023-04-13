import datetime
import os
import sys
import log
import logging
import argparse
import math

from model import Model
from iterator import Iterator
import data_provision_att_vqa
from eval_helper import evaluate
from utils import get_lr
from config import options
from log import log

from optimization_weight import *
from san_att_conv_twolayer_theano import *
from data_provision_att_vqa import *
from data_processing_vqa import *

##################
# initialization #
##################
options = OrderedDict()
# data related
options['data_path'] = 'imageqa-san/data'
options['expt_folder'] = 'imageqa-san/expt'
options['model_name'] = 'imageqa'
options['train_split'] = 'train'
options['val_split'] = 'val'
options['test_split'] = 'test'
options['shuffle'] = True
options['reverse'] = False
options['sample_answer'] = True

options['num_region'] = 196
options['region_dim'] = 512

options['n_words'] = 13746
options['n_output'] = 1000

# structure options
options['combined_num_mlp'] = 1
options['combined_mlp_drop_0'] = True
options['combined_mlp_act_0'] = 'linear'
options['sent_drop'] = False
options['use_tanh'] = False
options['use_unigram_conv'] = True
options['use_bigram_conv'] = True
options['use_trigram_conv'] = True

options['use_attention_drop'] = False
options['use_before_attention_drop'] = False

# dimensions
options['n_emb'] = 500
options['n_dim'] = 500
options['n_image_feat'] = options['region_dim']
options['n_common_feat'] = 500
options['num_filter_unigram'] = 256
options['num_filter_bigram'] = 512
options['num_filter_trigram'] = 512
options['n_attention'] = 512

# initialization
options['init_type'] = 'uniform'
options['range'] = 0.01
options['std'] = 0.01
options['init_lstm_svd'] = False

# learning parameters
options['optimization'] = 'sgd' # choices
options['batch_size'] = 100
options['lr'] = numpy.float32(0.1)
options['w_emb_lr'] = numpy.float32(80)
options['momentum'] = numpy.float32(0.9)
options['gamma'] = 1
options['step'] = 10
options['step_start'] = 100
options['max_epochs'] = 50
options['weight_decay'] = 0.0005
options['decay_rate'] = numpy.float32(0.999)
options['drop_ratio'] = numpy.float32(0.5)
options['smooth'] = numpy.float32(1e-8)
options['grad_clip'] = numpy.float32(0.1)

# log params
options['disp_interval'] = 10
options['eval_interval'] = 1000
options['save_interval'] = 500

def get_lr(options, curr_epoch):
    if options['optimization'] == 'sgd':
        power = max((curr_epoch - options['step_start']) / options['step'], 0)
        power = math.ceil(power)
        return options['lr'] * (options['gamma'] ** power)  #
    else:
        return options['lr']

def train(options):
    logger.info('compiling...')
    train_model = Model(options)
    train_model.build_train_model()

    logger.info('training...')
    data, label, data_lengths, label_lengths = data_provision_att_vqa.load_data()
    data_iterator = Iterator(data, label, data_lengths, label_lengths,
                              batch_size=options['batch_size'], shuffle=options['shuffle'])

    for epoch in range(options['max_epochs']):
        logger.info('Epoch ' + str(epoch + 1))
        lr = get_lr(options, epoch)
        logger.info('learning rate: %.4f' % lr)
        train_model.update_lr(lr)

        train_loss = 0
        num_corrects = 0
        num_samples = 0

        for i, (data, label, data_lengths, label_lengths) in enumerate(data_iterator):
            cost, accu = train_model.train(data, label, data_lengths, label_lengths)
            train_loss += cost
            num_corrects += accu
            num_samples += label_lengths.shape[0]

            if (i + 1) % options['disp_interval'] == 0:
                logger.info('epoch: %d, iteration: %d, cost: %f, accuracy: %f' %
                            (epoch + 1, i + 1, train_loss / options['disp_interval'], num_corrects / num_samples))
                train_loss = 0
                num_corrects = 0
                num_samples = 0

            if (i + 1) % options['eval_interval'] == 0:
                logger.info('evaluating...')
                eval_accu = evaluate(options, train_model, 'val', epoch, i + 1)
                logger.info('epoch: %d, iteration: %d, eval accuracy: %f' % (epoch + 1, i + 1, eval_accu))

            if (i + 1) % options['save_interval'] == 0:
                logger.info('saving...')
                train_model.save_model(options['expt_folder'], options['model_name'], epoch, i + 1)

    logger.info('training finished')
    logger.info('saving model...')
    train_model.save_model(options['expt_folder'], options['model_name'], epoch, i + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_folder', type=str, default='expt')
    parser.add_argument('--model_name', type=str, default='imageqa')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--val', type=int, default=0)
    args = parser.parse_args()

    options['expt_folder'] = args.expt_folder
    options['model_name'] = args.model_name

    log.configure_logging()
    logger = logging.getLogger('root')
    logger.info('Running %s' % str(sys.argv))
    logger.info(options)

    if args.train:
        train(options)

    if args.test:
        evaluate(options, None, 'test', -1, -1)

    if args.val:
        evaluate(options, None, 'val', -1, -1)