import os
import argparse
import tensorflow as tf
from model import Model
from six import text_type
from six.moves import cPickle


def main(input_strings=["what a day it is!"]):
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
    test_log_prob(args, input_strings)


def test_log_prob(args, input_strings):
    # batch_size=1, data_dir='data/tinyshakespeare', decay_rate=0.97, grad_clip=5.0, init_from=None,
    # learning_rate=0.002, model='lstm', num_epochs=50, num_layers=2, rnn_size=128, save_dir='save', save_every=1000,
    # seq_length=1, vocab_size=65)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
        print chars
        print vocab

    print("Getting models using saved arguments: " + repr(saved_args))
    model = Model(saved_args, True)
    print(model.probs)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        ckpt = tf.train.get_checkpoint_state(args.save_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for input_string in input_strings:
                log_probability = model.log_probability(sess, vocab, input_string)
                print(log_probability)

if __name__ == "__main__":
    main(["AABBCCDDEEFFGGHHHHHHHHHHHHHHHHHHHHZZZZZZZZZZZZZZ", "I love Bangalore city as it has good people"])
