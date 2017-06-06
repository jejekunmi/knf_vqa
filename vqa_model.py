import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import util
import time
import numpy as np
from time import gmtime, strftime
from random import sample as rand_sample
from PIL import Image
import image_utils

class Encoder(object):
    def __init__(self, config):
        self.config = config

    def encode(self, inputs, masks, encoder_state_input, embeddings, dropout_keep_prob):
        pass        

class Decoder(object):
    def __init__(self, config):
        self.config = config

    def decode(self, knowledge_rep, dropout_keep_prob):
        pass

class VQASystem(object):
    def __init__(self, encoder, decoder, pretrained_embeddings, config):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        
        # ==== set up placeholder tokens ========
        self.add_placeholders()

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings(pretrained_embeddings)
            self.setup_system()
            self.setup_loss()
            
        # ==== set up training/updating procedure ====
            if self.config.train_all:
                self.add_training_op()
            else:
                self.add_filtered_training_op()
            self.add_eval_op()
            self.add_summaries()
            self.saver = tf.train.Saver()
            
        print("Graph setup done!!")
            
    def add_placeholders(self): 
        # Image
        self.image_input_placeholder = tf.placeholder(tf.float32, 
                                                      shape = [None] + self.config.image_size)
        
        # Question and mask
        self.question_input_placeholder = tf.placeholder(tf.int32, 
                                                         shape=[None, self.config.max_question_length],
                                                         name="questions")
        self.question_mask_placeholder = tf.placeholder(tf.int32, shape=[None,], name="question_masks") 
        
        # Answer
        self.answer_placeholder = tf.placeholder(tf.int32, [None,])
        
        # All answers
        self.all_answer_placeholder = tf.placeholder(tf.int32, 
                                                     [None, self.config.num_answers_per_question])
        
        # image mask (because of bidirectinal rnn bug)
        #self.image_mask_placeholder = tf.placeholder(tf.int32, shape=[None,], name="image_masks")
        
        # Dropout rate
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="dropout")
        
        self.const_placeholder = tf.placeholder(tf.float32, shape=(), name="const")
        
        self.is_training_placeholder = tf.placeholder(tf.bool)
        
        print("Done Adding Placeholers!")

    def setup_system(self):
        encoding = self.encoder.encode(inputs=(self.image_input_placeholder,
                                               #self.image_mask_placeholder,
                                               self.question_input_placeholder, 
                                               self.question_mask_placeholder),
                                       encoder_state_input=None,
                                       embeddings=self.embeddings, dropout_keep_prob=self.dropout_placeholder)
        print("encoding", encoding)

        self.scores = self.decoder.decode(encoding, dropout_keep_prob=self.dropout_placeholder)
        print("scores", self.scores)
        print("Done setting up system!")


    def setup_loss(self):
        with vs.variable_scope("loss"):
            cr_ent_loss = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=self.answer_placeholder, 
                                logits=self.scores))
            l2_cost = 0.0
            for var in tf.trainable_variables():
                if 'weight' in var.name or 'kernel' in var.name:
                    l2_cost += tf.nn.l2_loss(var)
                    
            self.loss = cr_ent_loss + self.config.l2_reg * l2_cost
        print("Done setting up loss!")

    def setup_embeddings(self, pretrained_embeddings):
        with vs.variable_scope("embeddings"):
            self.embeddings = tf.Variable(initial_value=pretrained_embeddings, 
                                          dtype=tf.float32, name="L")
        print("Done Adding Embedding!")
    

    def add_training_op(self):
        self.optimizer = self.config.optimizer(learning_rate=self.config.learning_rate)

        grad_list = self.optimizer.compute_gradients(self.loss)
        grads = [g for g, v in grad_list]
        if self.config.clip_gradients:            
            grads, _ = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
            grad_list = [(grads[i], grad_list[i][1]) for i in range(len(grads))]
        self.grad_norm = tf.global_norm(grads)        
        self.train_op = self.optimizer.apply_gradients(grad_list)
        print("Done adding training op!")

    def add_filtered_training_op(self):
        self.optimizer = self.config.optimizer(learning_rate=self.config.learning_rate)

        grad_list = self.optimizer.compute_gradients(self.loss)
        print(len(grad_list))
        grad_list = [(g, v) for (g, v) in grad_list if v not in self.encoder.vgg_net.parameters]
        print(len(grad_list))
        grads = [g for g, v in grad_list]
        if self.config.clip_gradients:            
            grads, _ = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
            grad_list = [(grads[i], grad_list[i][1]) for i in range(len(grads))]
        self.grad_norm = tf.global_norm(grads)        
        self.train_op = self.optimizer.apply_gradients(grad_list)
        print("Done adding training op!")
        
    '''
    def acc_count(self, pred, all_answers):
        pred = tf.reshape(pred, shape=(-1, 1))
        elements_equal_to_value = tf.equal(pred, all_answers)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints, axis=1)
        accuracy = 1.0 * tf.minimum(count / 3, 1.0)
        return accuracy
    
    def add_eval_op(self):
        self.answer = tf.cast(tf.argmax(self.scores, axis=1), tf.int32)
        self.accuracy = tf.reduce_mean(self.acc_count(self.answer, self.all_answer_placeholder))
    '''
    
    def add_eval_op(self):
        self.pred_answer = tf.cast(tf.argmax(self.scores, axis=1), tf.int32)
        self.correct_pred = tf.equal(self.pred_answer, self.answer_placeholder)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
    def add_summaries(self):
        self.train_loss_summary = tf.summary.scalar("train_loss", self.loss) 
        self.val_loss_summary = tf.summary.scalar("val_loss", self.loss) 
        self.grad_norm_summary = tf.summary.scalar("grad_norm", self.grad_norm) 
        self.train_acc_summary = tf.summary.scalar("train_acc", self.accuracy) 
        self.val_acc_summary = tf.summary.scalar("val_acc", self.accuracy) 
        #self.const_sum = tf.summary.scalar("training_accuracy", self.const_placeholder)
                
    def create_feed_dict(self, image_inputs_batch=None, image_mask_batch=None, question_inputs_batch=None,
                         question_masks_batch=None, answers_batch=None, 
                         all_answers_batch=None, dropout_keep_prob=1.0):
        feed_dict = {}
        if image_inputs_batch is not None: 
            feed_dict[self.image_input_placeholder] = image_inputs_batch
            #feed_dict[self.image_mask_placeholder] = self.config.images_input_sequece_len * \
            #                                         np.ones(len(image_inputs_batch), dtype=np.int32)
        if question_inputs_batch is not None: 
            feed_dict[self.question_input_placeholder] = question_inputs_batch
        if question_masks_batch is not None: 
            feed_dict[self.question_mask_placeholder] = question_masks_batch
        if answers_batch is not None: 
            feed_dict[self.answer_placeholder] = answers_batch
        if all_answers_batch is not None: 
            feed_dict[self.all_answer_placeholder] = all_answers_batch
        if dropout_keep_prob is not None: 
            feed_dict[self.dropout_placeholder] = dropout_keep_prob        
        return feed_dict

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        image_batch, question_batch, mask_batch = train_x
        answers_batch = train_y
        input_feed = self.create_feed_dict(image_inputs_batch=image_batch, 
                                           question_inputs_batch=question_batch,
                                           question_masks_batch=mask_batch, 
                                           answers_batch=answers_batch, 
                                           dropout_keep_prob=self.config.dropout_keep_prob)
        output_feed = [self.train_op, self.loss, self.grad_norm, 
                       self.train_loss_summary, self.grad_norm_summary]
        
        _, train_loss, g_norm, train_loss_summ, gnorm_summ = session.run(output_feed, input_feed)
        self.train_writer.add_summary(train_loss_summ, self.step_num)
        self.train_writer.add_summary(gnorm_summ, self.step_num)
                
        #return (train_loss, g_norm)

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        image_batch, question_batch, mask_batch = valid_x
        answers_batch = valid_y
        input_feed = self.create_feed_dict(image_inputs_batch=image_batch, 
                                           question_inputs_batch=question_batch,
                                           question_masks_batch=mask_batch, 
                                           answers_batch=answers_batch, 
                                           dropout_keep_prob=1.0)
        output_feed = [self.loss, self.val_loss_summary]
        
        val_loss, val_loss_summ = session.run(output_feed, input_feed)
        self.train_writer.add_summary(val_loss_summ, self.step_num)
        #return loss

    def decode(self, session, test_x):
        image_batch, question_batch, mask_batch = test_x
        input_feed = self.create_feed_dict(image_inputs_batch=image_batch, 
                                           question_inputs_batch=question_batch,
                                           question_masks_batch=mask_batch, 
                                           answers_batch=answers_batch, 
                                           dropout_keep_prob=1.0)
        output_feed = [self.pred_answer]
        
        answers_id = session.run(output_feed, input_feed)
        #writer.add_summary(val_loss, step)
        return loss

    def answer(self, session, test_x):
        return self.decode(session, test_x)

    def validate(self, sess, valid_dataset):
        pass
    
    def eval_batch(self, session, x, y, datatype):
        image_batch, question_batch, mask_batch = x
        answers_batch = y
        input_feed = self.create_feed_dict(image_inputs_batch=image_batch, 
                                           question_inputs_batch=question_batch,
                                           question_masks_batch=mask_batch, 
                                           answers_batch=answers_batch, 
                                           dropout_keep_prob=1.0)
        
        if datatype == "train":
            output_feed = [self.accuracy, self.train_acc_summary]        
            acc, train_acc_summ = session.run(output_feed, input_feed)
            self.train_writer.add_summary(train_acc_summ, self.step_num)
        elif datatype == "val":
            output_feed = [self.accuracy, self.val_acc_summary]        
            acc, val_acc_summ = session.run(output_feed, input_feed)
            self.train_writer.add_summary(val_acc_summ, self.step_num)
        return acc

    def read_image(self, filename_queue):
        reader = tf.WholeFileReader()
        key,value = reader.read(filename_queue)
        image = tf.image.decode_image(value, channels=3)
        return key,image

    def image_inputs(self, filenames):
        filename_queue = tf.train.string_input_producer(filenames)
        filename,read_img = self.read_image(filename_queue)
        return filename, read_img
    
    def get_image_batch(self, sess, image_ids_batch, datatype):
        #tic = time.time()
        input_files = ['data/preprocessed_images_' + datatype + '/' + str(img_id) + '.jpeg'
                       for img_id in image_ids_batch]
        
        images_batch = np.empty([len(image_ids_batch)] + self.config.image_size)
        
        for i, f in enumerate(input_files):
            im = np.asarray(Image.open(f))
            #print(im.shape)
            images_batch[i] = im
        
        #toc = time.time()
        #print("Took {} seconds".format(toc - tic))
        return images_batch
    
    def evaluate_answer(self, session, sample_size=100, log=False, dataset=None):
        train_indices = rand_sample(list(range(dataset.train.questions.shape[0])), sample_size)
        val_indices = rand_sample(list(range(dataset.val.questions.shape[0])), sample_size)
        
        train_batch = util.get_minibatches_indices(train_indices, self.config.batch_size) 
        val_batch = util.get_minibatches_indices(val_indices, self.config.batch_size)
        
        train_acc, val_acc = 0.0, 0.0
        
        #print("Evaluating")
        prog = util.Progbar(target=1 + int(len(train_indices) / self.config.batch_size))
        for i in range(len(train_batch)):
            # Run optimizer for train data
            batch_indices = train_batch[i]
            image_ids_batch = util.get_batch_from_indices(dataset.train.image_ids, batch_indices) 
            image_batch = self.get_image_batch(session, image_ids_batch, "train")
            question_batch = util.get_batch_from_indices(dataset.train.questions, batch_indices)
            mask_batch = util.get_batch_from_indices(dataset.train.mask, batch_indices)
            answers_batch = util.get_batch_from_indices(dataset.train.answers, batch_indices)
            t_acc = self.eval_batch(session, (image_batch, question_batch, mask_batch), 
                                    answers_batch, "train")
            train_acc += 100.0 * len(batch_indices) * t_acc / sample_size
            
            # Get val loss
            batch_indices = val_batch[i%len(val_batch)]
            image_ids_batch = util.get_batch_from_indices(dataset.val.image_ids, batch_indices) 
            image_batch = self.get_image_batch(session, image_ids_batch, "val")
            question_batch = util.get_batch_from_indices(dataset.val.questions, batch_indices)
            mask_batch = util.get_batch_from_indices(dataset.val.mask, batch_indices)
            answers_batch = util.get_batch_from_indices(dataset.val.answers, batch_indices)
            v_acc = self.eval_batch(session, (image_batch, question_batch, mask_batch), 
                                     answers_batch, "val")
            val_acc += 100.0 * len(batch_indices) * v_acc / sample_size
            
            prog.update(i + 1, [("Evaluating", i)])
        return (train_acc, val_acc)        

    def run_epoch(self, sess, dataset, epoch):
        train_indices = list(range(dataset.train.questions.shape[0]))[:self.config.train_limit]
        val_indices = list(range(dataset.val.questions.shape[0]))[:self.config.train_limit]
        
        train_batch = util.get_minibatches_indices(train_indices, self.config.batch_size) 
        val_batch = util.get_minibatches_indices(val_indices, self.config.batch_size)
        
        #print("Training")
        prog = util.Progbar(target=1 + int(len(train_indices) / self.config.batch_size))
        for i in range(len(train_batch)):
            # Run optimizer for train data
            batch_indices = train_batch[i]
            image_ids_batch = util.get_batch_from_indices(dataset.train.image_ids, batch_indices) 
            image_batch = self.get_image_batch(sess, image_ids_batch, "train")
            question_batch = util.get_batch_from_indices(dataset.train.questions, batch_indices)
            mask_batch = util.get_batch_from_indices(dataset.train.mask, batch_indices)
            answers_batch = util.get_batch_from_indices(dataset.train.answers, batch_indices)
            self.optimize(sess, (image_batch, question_batch, mask_batch), answers_batch)
            
            # Get val loss
            batch_indices = val_batch[i%len(val_batch)]
            image_ids_batch = util.get_batch_from_indices(dataset.val.image_ids, batch_indices) 
            image_batch = self.get_image_batch(sess, image_ids_batch, "val")
            question_batch = util.get_batch_from_indices(dataset.val.questions, batch_indices)
            mask_batch = util.get_batch_from_indices(dataset.val.mask, batch_indices)
            answers_batch = util.get_batch_from_indices(dataset.val.answers, batch_indices)
            self.test(sess, (image_batch, question_batch, mask_batch), answers_batch)
            
            if (i+1) % self.config.eval_every == 0:
                train_acc, val_acc = self.evaluate_answer(sess, sample_size=self.config.num_evaluate, 
                                           log=True, dataset=dataset)
                print("Iter", i+1, "in epoch", epoch+1, train_acc, val_acc)
                if val_acc > self.best_score:
                    self.best_score = val_acc
                    print("New best score! Saving model in {}".format(self.config.model_dir + "/model.ckpt"))
                    self.saver.save(sess, self.config.model_dir + "/model.ckpt")
            
            self.step_num += 1
            prog.update(i + 1, [("Training", i)])
                        
        acc = self.evaluate_answer(sess, sample_size=self.config.num_evaluate, log=True, dataset=dataset) 
        return acc

    def train(self, session, dataset, best_score=0.0):
        """
        Implement main training loop

        :param session: it should be passed in from train.py
        :return:
        """
        self.best_score = best_score
        self.step_num = 0
        #logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        logs_path = "tensorboard/" + \
            "lr_{}_lrd_{}_r_{}_gn_{}_dr_{}_bs_{}".format(self.config.learning_rate,
                                                         self.config.lr_decay,
                                                         self.config.l2_reg,
                                                         self.config.max_gradient_norm,
                                                         self.config.dropout_keep_prob,
                                                         self.config.batch_size) + \
            strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        self.train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
        self.train_writer.add_graph(session.graph)
        #self.val_writer = tf.summary.FileWriter(logs_path + '/val', session.graph)
        
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        print("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        losses, accuracies = [], []
        
        for epoch in range(self.config.epochs):
            print("Epoch {} out of {}".format(epoch + 1, self.config.epochs))
            train_acc, val_acc = self.run_epoch(session, dataset, epoch)
            print(epoch, train_acc, val_acc)
            
            
            '''
            _, s = session.run([self.const_placeholder, self.const_sum], 
                            {self.const_placeholder: 1.0 * epoch})
            train_writer.add_summary(s, epoch*2)
            '''
            if val_acc > self.best_score:
                self.best_score = val_acc
                print("New best score! Saving model in {}".format(self.config.model_dir + "/model.ckpt"))
                self.saver.save(session, self.config.model_dir + "/model.ckpt")
                
            self.config.learning_rate *= self.config.lr_decay

            #losses.append(loss)       
            accuracies.append(val_acc)
        