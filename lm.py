import os
import argparse
import tensorflow as tf
from model import Model
from six import text_type
from six.moves import cPickle

args = None
model = None
saved_args = None
chars = None
vocab = None


def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    args = parser.parse_args()
    return args


def get_log_prob(input_strings=["what a day it is!"]):
    # batch_size=1, data_dir='data/tinyshakespeare', decay_rate=0.97, grad_clip=5.0, init_from=None,
    # learning_rate=0.002, model='lstm', num_epochs=50, num_layers=2, rnn_size=128, save_dir='save', save_every=1000,
    # seq_length=1, vocab_size=65)

    global args, model, saved_args, chars, vocab

    if args is None:
        args = get_parsed_arguments()
    else:
        print "Reusing args"

    if saved_args is None:
        with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
    else:
        print "Reusing saved args"

    if chars is None or vocab is None:
        with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
            chars, vocab = cPickle.load(f)
    else:
        print "Reusing char and vocab"
    lm_values = dict()

    # print("Getting models using saved arguments: " + repr(saved_args))
    if model is None:
        model = Model(saved_args, True)
    else:
        print "Reusing Model"
    # print(model.probs)
    get_lm_probs_impl(args, input_strings, model, lm_values, vocab)
    return lm_values


def get_lm_probs_impl(args, input_strings, model, values, vocab):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        ckpt = tf.train.get_checkpoint_state(args.save_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for input_string in input_strings:
                log_probability = model.log_probability(sess, vocab, input_string)
                custom_log_probability = log_probability / len(input_string)
                values[input_string] = custom_log_probability
                # print "The probability of '" + input_string + "': ", log_probability/len(input_string)


from pprint import pprint

if __name__ == "__main__":
    pprint(get_log_prob(["AABBCCDDEEFFGGHHHHHHHHHHHHHHHHHHHHZZZZZZZZZZZZZZ",
                  "I love Bangalore city as it has good people",
                  "wht a dyy"]))
    pprint(get_log_prob(["AABBCCDDEEFFGGHHHHHHHHHHHHHHHHHHHHZZZZZZZZZZZZZZ",
                  "I love Bangalore city as it has good people",
                  "wht a dyy"]))

