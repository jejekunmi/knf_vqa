# Import modules needed
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import pickle

import util
from vqa_model import Encoder, Decoder, VQASystem
from squeezenet import SqueezeNet
from vgg16 import VGG16

from compact_bilinear_pooling import compact_bilinear_pooling_layer

def build_bipool_kunmi():
    # Load data (excluding images)

    # Restrict the number of possible answers using this
    # Decreasing this will increase the number of classes 
    train_min_count = 99
    val_cutoff = 107183
    # Load data
    dataset = util.load_data_all(train_min_count, val_cutoff=107183, limit=100000000)
    np_embeddings = np.load("data/glove.trimmed.100.npz")["glove"]
    answer_to_id, id_to_answer = util.load_answer_map(min_count=train_min_count)
    with open('data/qid_to_anstype.dat', 'rb') as fp:
        qid_to_anstype = pickle.load(fp)
        
    class Config:
        """Holds model hyperparams and data information.
        """
        epochs = 20

        learning_rate = 0.001
        lr_decay = 0.9
        optimizer = tf.train.AdamOptimizer
        max_gradient_norm = 10.0
        clip_gradients = True
        dropout_keep_prob = 1.0
        l2_reg = 0.0
        batch_size = 64
        train_all = False

        train_limit = 10000000

        max_question_length = 25    
        num_answers_per_question = 10
        num_classes = 3004
        image_size = [224, 224, 3]
        cbpl_output_dim = 8000
        att_conv1_dim = 512

        vgg_out_dim = [14, 14]

        images_input_sequece_len = vgg_out_dim[0] * vgg_out_dim[1]

        rnn_hidden_size = 512 # RNN
        fc_state_size = 100 # Fully connected
        embedding_size = 100

        num_evaluate = 5000

        eval_every = 500
        print_every = 100 

        model_dir = "bilinear_model"
        squeeze_net_dir = "sq_net_model/squeezenet.ckpt"
        vgg16_weight_file = "vgg_net_dir/vgg16_weights.npz"
        
    class BaselineEncoder(Encoder):        
    
        def encode(self, inputs, encoder_state_input, embeddings, dropout_keep_prob):
            images, _, questions, question_masks = inputs
            self.batch_size = tf.shape(images)[0]

            self.vgg_net = VGG16(imgs=images, weights=Config.vgg16_weight_file)
            self.vgg_out = self.vgg_net.conv5_3
            print("vgg_out", self.vgg_out)

            with tf.variable_scope('vqa_additional'):
                # Encode question with GRU
                questions_input = tf.nn.embedding_lookup(embeddings, questions)
                with tf.variable_scope('q_encoder') as scope:
                    gru_cell = tf.contrib.rnn.GRUCell(self.config.rnn_hidden_size)
    #                 gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell,
    #                                                          input_keep_prob=dropout_keep_prob,
    #                                                          output_keep_prob=dropout_keep_prob)
                    outputs, state = tf.nn.dynamic_rnn(cell=gru_cell,
                                                       inputs=questions_input,
                                                       sequence_length=question_masks,
                                                       dtype=tf.float32)
                # Question representation
                self.q_enc = state
                self.q_enc = tf.nn.dropout(self.q_enc, keep_prob=dropout_keep_prob)
                print("q_enc", self.q_enc)

                tile_temp = tf.reshape(self.q_enc, shape=(self.batch_size, 1, 1, self.config.rnn_hidden_size))
                self.q_tile = tf.tile(tile_temp, [1] + self.config.vgg_out_dim + [1])
                print("q_tile", self.q_tile)

                self.q_im_attn = compact_bilinear_pooling_layer(bottom1=self.vgg_out, bottom2=self.q_tile, 
                                                                output_dim=self.config.cbpl_output_dim,
                                                                sum_pool=False)
    #             self.q_im_attn = tf.sign(self.q_im_attn) * tf.sqrt(tf.abs(self.q_im_attn))
    #             self.q_im_attn = tf.nn.l2_normalize(self.q_im_attn, 0)
                print("q_im_attn", self.q_im_attn)

                self.att_Wconv1 = tf.Variable(tf.truncated_normal([3, 3, self.config.cbpl_output_dim, 
                                                                   self.config.att_conv1_dim], 
                                                                  dtype=tf.float32,
                                                                  stddev=1e-1), 
                                              name='att_Wconv1_weight')
                self.att_bconv1 = tf.Variable(tf.constant(0.0, shape=[self.config.att_conv1_dim], dtype=tf.float32),
                                     trainable=True, name='att_bconv1')
                self.attn_conv1 = tf.nn.conv2d(self.q_im_attn,self.att_Wconv1,
                                               strides=[1,1,1,1], padding='SAME') + self.att_bconv1
                self.attn_conv1 = tf.nn.relu(self.attn_conv1)
                print("attn_conv1", self.attn_conv1)

                self.att_Wconv2 = tf.Variable(tf.truncated_normal([3, 3, self.config.att_conv1_dim, 1], 
                                                                  dtype=tf.float32,
                                                                  stddev=1e-1), 
                                              name='att_Wconv2_weight')
                self.att_bconv2 = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                                     trainable=True, name='att_bconv2')
                self.attn_conv2 = tf.nn.conv2d(self.attn_conv1, self.att_Wconv2,
                                               strides=[1,1,1,1], padding='SAME') + self.att_bconv2
                print("attn_conv2", self.attn_conv2)

                self.attn_flat = tf.reshape(self.attn_conv2, shape=[-1, self.config.images_input_sequece_len])
                self.alpha = tf.nn.softmax(self.attn_flat)
                self.alpha = tf.reshape(self.alpha, shape=[-1] + self.config.vgg_out_dim + [1])
                print("alpha", self.alpha)

                weighted = self.alpha * self.vgg_out
                print("weighted",weighted)
                self.attended_image = tf.reduce_sum(weighted, axis=(1,2))
                self.attended_image = tf.nn.dropout(self.attended_image, keep_prob=dropout_keep_prob)
                print("attended_image", self.attended_image)

                a=tf.reshape(self.attended_image, shape=(-1, 1, 1, self.config.rnn_hidden_size), name="HERE1")
                b=tf.reshape(self.q_enc, shape=(-1, 1, 1, self.config.rnn_hidden_size), name="HERE2")
                print("a", a)
                print("b", b)

                self.attd_im_q = compact_bilinear_pooling_layer(bottom1=a, 
                                                                bottom2=b,  
                                                                 output_dim=self.config.cbpl_output_dim,
                                                                 sum_pool=False)
                #self.attd_im_q = tf.sign(self.attd_im_q) * tf.sqrt(tf.abs(self.attd_im_q))
    #             self.attd_im_q = tf.nn.l2_normalize(self.attd_im_q, 0)

                self.attd_im_q = tf.reshape(self.attd_im_q, shape=[-1, self.config.cbpl_output_dim])
                print("attd_im_q", self.attd_im_q)

            return self.attd_im_q
            #return state

    class BaselineDecoder(Encoder):
        def decode(self, knowledge_rep, dropout_keep_prob):
            scores = tf.layers.dense(inputs=knowledge_rep, units=self.config.num_classes, 
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
            return scores
        
    # clear old variables
    tf.reset_default_graph()

    vqa_encoder = BaselineEncoder(config=Config)
    vqa_decoder = BaselineDecoder(config=Config)

    vqa_system = VQASystem(encoder=vqa_encoder, decoder=vqa_decoder, 
                           pretrained_embeddings=np_embeddings, config=Config)  
    
    return vqa_system