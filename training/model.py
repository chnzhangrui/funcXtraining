import json
from pdb import set_trace
import numpy as np
import os, sys

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

tf.keras.backend.set_floatx("float32")

class SpectralNorm(Wrapper):

    def __init__(self, layer, iteration=1, **kwargs):
        super(SpectralNorm, self).__init__(layer, **kwargs)
        self.iteration = iteration

    def build(self, input_shape):

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('Invalid layer for SpectralNorm.')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()
            self.u = self.add_weight(shape=(1, self.w_shape[-1]), initializer=tf.random_normal_initializer(), name='sn_u', trainable=False, dtype=tf.float32)

        super(SpectralNorm, self).build()

    @tf.function
    def call(self, inputs, training=None):

        self._compute_weights(training)
        output = self.layer(inputs)

        return output

    def _compute_weights(self, training):
        
        iteration = self.iteration
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = tf.identity(self.u)
        v_hat = None

        for _ in range(self.iteration):
                
            v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w_reshaped)
            u_hat = tf.nn.l2_normalize(u_)

        if training == True: self.u.assign(u_hat)
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        
        w_norm = self.w / sigma

        self.layer.kernel = w_norm
        
    def compute_output_shape(self, input_shape):

        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())


class WGANGP:
    def __init__(self, sample_config, hp_config):
        self.model = hp_config.get('model', 'BNswish') # default to photon GAN BNswish
        self.G_size = hp_config.get('G_size', 1)
        self.D_size = hp_config.get('D_size', 1)
        self.G_lr = hp_config.get('G_lr', 0.0001)
        self.D_lr = hp_config.get('D_lr', 0.0001)
        self.G_beta1 = hp_config.get('G_beta1', 0.5)
        self.D_beta1 = hp_config.get('D_beta1', 0.5)
        self.batchsize = hp_config.get('batchsize', 512)
        self.dgratio = hp_config.get('dgratio', 5)

        self.particle = sample_config.get('particle', 'photons')
        self.eta_min, self.eta_max = tuple(sample_config.get('eta_slice', '20_25').split('_'))

        # Construct D and G models
        self.G = self.make_generator_functional_model()
        self.D = self.make_discriminator_model()

        # Construct D and G optimizers
        self.generator_optimizer = tf.optimizers.Adam(learning_rate=self.G_lr, beta_1=self.G_beta1)
        self.discriminator_optimizer = tf.optimizers.Adam(learning_rate=self.D_lr, beta_1=self.D_beta1)

        # Prepare for check pointing
        self.saver = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer, generator=self.G, discriminator=self.D,)
        self.tf_batchsize = tf.constant(self.batchsize, dtype=tf.int32)
        self.tf_n_disc = tf.constant(self.dgratio, dtype=tf.int32)

    def make_generator_functional_model(self):
        noise = layers.Input(shape=(self.ganParameters.latent_dim,), name="Noise")
        condition = layers.Input(shape=(self.ganParameters.conditional_dim,), name="mycond")
        con = layers.concatenate([noise, condition])
        print('Use BN', self.bn)
        initializer = tf.keras.initializers.he_uniform()

        if self.bn == "BNReLU":
            G = layers.Dense(self.ganParameters.generatorLayers[0], kernel_initializer=initializer, bias_initializer="zeros")(con)
            G = layers.BatchNormalization()(G)
            G = layers.ReLU()(G)
            G = layers.Dense(self.ganParameters.generatorLayers[1], kernel_initializer=initializer, bias_initializer="zeros")(G)
            G = layers.BatchNormalization()(G)
            G = layers.ReLU()(G)
            G = layers.Dense(self.ganParameters.generatorLayers[2],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.BatchNormalization()(G)
            G = layers.ReLU()(G)
            G = layers.Dense(self.ganParameters.nvoxels,use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.ReLU()(G)
        elif self.bn == "BNswish":
            initializer = tf.keras.initializers.glorot_normal()
            G = layers.Dense(self.ganParameters.generatorLayers[0],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(con)
            G = layers.BatchNormalization()(G)
            G = layers.Activation(activations.swish)(G)
            G = layers.Dense(self.ganParameters.generatorLayers[1],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.BatchNormalization()(G)
            G = layers.Activation(activations.swish)(G)
            G = layers.Dense(self.ganParameters.generatorLayers[2],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.BatchNormalization()(G)
            G = layers.Activation(activations.swish)(G)
            G = layers.Dense(self.ganParameters.nvoxels,use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.BatchNormalization()(G)
            G = layers.Activation(activations.swish)(G)
        elif self.bn == "BNLeakyReLU":
            G = layers.Dense(self.ganParameters.generatorLayers[0],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(con)
            G = layers.BatchNormalization()(G)
            G = layers.LeakyReLU(alpha=0)(G)
            G = layers.Dense(self.ganParameters.generatorLayers[1],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.BatchNormalization()(G)
            G = layers.LeakyReLU(alpha=0)(G)
            G = layers.Dense(self.ganParameters.generatorLayers[2],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.BatchNormalization()(G)
            G = layers.LeakyReLU(alpha=0)(G)
            G = layers.Dense(self.ganParameters.nvoxels,use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.BatchNormalization()(G)
            G = layers.LeakyReLU(alpha=0)(G)
        # elif self.bn == "bnF":
        #     G = layers.Dense(self.ganParameters.generatorLayers[0],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(con)
        #     G = layers.BatchNormalization()(G)
        #     G = layers.LeakyReLU(alpha=0.03)(G)
        #     G = layers.Dense(self.ganParameters.generatorLayers[1],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
        #     G = layers.BatchNormalization()(G)
        #     G = layers.LeakyReLU(alpha=0.03)(G)
        #     G = layers.Dense(self.ganParameters.generatorLayers[2],use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
        #     G = layers.BatchNormalization()(G)
        #     G = layers.LeakyReLU(alpha=0.03)(G)
        #     G = layers.Dense(self.ganParameters.nvoxels,use_bias=bias_node,kernel_initializer=initializer,bias_initializer="zeros")(G)
        #     G = layers.BatchNormalization()(G)
        #     G = layers.Activation(activation='sigmoid')(G)
        elif self.bn == "noBN":
            G = layers.Dense(self.ganParameters.generatorLayers[0],use_bias=bias_node,activation="relu",kernel_initializer=initializer,bias_initializer="zeros")(con)
            G = layers.Dense(self.ganParameters.generatorLayers[1],use_bias=bias_node,activation="relu",kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.Dense(self.ganParameters.generatorLayers[2],use_bias=bias_node,activation="relu",kernel_initializer=initializer,bias_initializer="zeros")(G)
            G = layers.Dense(self.ganParameters.nvoxels,use_bias=bias_node,activation="relu",kernel_initializer=initializer,bias_initializer="zeros")(G)
        elif self.bn == "SN":
            G = SpectralNorm(layers.Dense(self.ganParameters.generatorLayers[0],use_bias=bias_node,activation="relu",kernel_initializer=initializer,bias_initializer="zeros"))(con)
            G = SpectralNorm(layers.Dense(self.ganParameters.generatorLayers[1],use_bias=bias_node,activation="relu",kernel_initializer=initializer,bias_initializer="zeros"))(G)
            G = SpectralNorm(layers.Dense(self.ganParameters.generatorLayers[2],use_bias=bias_node,activation="relu",kernel_initializer=initializer,bias_initializer="zeros"))(G)
            G = SpectralNorm(layers.Dense(self.ganParameters.nvoxels,use_bias=bias_node,activation="relu",kernel_initializer=initializer,bias_initializer="zeros"))(G)
        else:
            assert(0)

        generator = Model(inputs=[noise, condition], outputs=G)
        generator.summary()
        return generator

    def make_discriminator_model(self):
        initializer = tf.keras.initializers.he_uniform()
        bias_node = True
        model = tf.keras.Sequential()
        model.add(layers.Dense(int(self.ganParameters.discriminatorLayers[0] * self.D_size),use_bias=bias_node,input_shape=(self.ganParameters.nvoxels + self.ganParameters.conditional_dim,),kernel_initializer=initializer,bias_initializer="zeros"))
        model.add(layers.ReLU())
        model.add(layers.Dense(int(self.ganParameters.discriminatorLayers[1] * self.D_size),use_bias=bias_node,input_shape=(int(self.ganParameters.discriminatorLayers[0] * self.D_size, kernel_initializer=initializer,bias_initializer="zeros"))))
        model.add(layers.ReLU())
        model.add(layers.Dense(int(self.ganParameters.discriminatorLayers[2] * self.D_size),use_bias=bias_node,input_shape=(int(self.ganParameters.discriminatorLayers[1] * self.D_size, kernel_initializer=initializer,bias_initializer="zeros"))))
        model.add(layers.ReLU())
        model.add(layers.Dense(1, use_bias=bias_node,input_shape=(int(self.ganParameters.discriminatorLayers[2] * self.D_size),),kernel_initializer=initializer,bias_initializer="zeros"))

        model.summary()
        return model

    @tf.function
    def gradient_penalty(self, f, x_real, x_fake, cond_label):
        alpha = tf.random.uniform([self.ganParameters.batchsize, 1], minval=0.0, maxval=1.0)

        inter = alpha * x_real + (1 - alpha) * x_fake
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = self.D(tf.concat([inter, cond_label], 1))
        grad = t.gradient(pred, [inter])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
        gp = self.ganParameters.lam * tf.reduce_mean((slopes - 1.0) ** 2)
        return gp

    @tf.function
    def D_loss(self, x_real, cond_label):
        z = tf.random.normal([self.ganParameters.batchsize, self.ganParameters.latent_dim],mean=0.5,stddev=0.5,dtype=tf.dtypes.float32,)
        x_fake = self.G(inputs=[z, cond_label])
        D_fake = self.D(tf.concat([x_fake, cond_label], 1))
        D_real = self.D(tf.concat([x_real, cond_label], 1))
        D_loss = (tf.reduce_mean(D_fake)- tf.reduce_mean(D_real)+ self.gradient_penalty(f=partial(self.D, training=True),x_real=x_real,x_fake=x_fake,cond_label=cond_label,))
        return D_loss, D_fake

    @tf.function
    def G_loss(self, D_fake):
        G_loss = -tf.reduce_mean(D_fake)
        return G_loss

    def getTrainData_ultimate(self, n_epoch):
        true_batchsize = tf.cast(tf.math.multiply(self.tf_batchsize, self.tf_n_disc), tf.int64)
        n_samples = tf.cast(tf.gather(tf.shape(self.X), 0), tf.int64)
        n_batch = tf.cast(tf.math.floordiv(n_samples, true_batchsize), tf.int64)
        n_shuffles = tf.cast(tf.math.ceil(tf.divide(n_epoch, n_batch)), tf.int64)
        ds = tf.data.Dataset.from_tensor_slices((self.X, self.Labels))
        ds = ds.shuffle(buffer_size=n_samples).repeat(n_shuffles).batch(true_batchsize, drop_remainder=True).prefetch(2)
        self.ds = ds
        self.ds_iter = iter(ds)
        X_feature_size = tf.gather(tf.shape(self.X), 1)
        Labels_feature_size = tf.gather(tf.shape(self.Labels), 1)
        self.X_batch_shape = tf.stack((self.tf_n_disc, self.tf_batchsize, X_feature_size), axis=0)
        self.Labels_batch_shape = tf.stack((self.tf_n_disc, self.tf_batchsize, Labels_feature_size), axis=0)

    @tf.function
    def train_loop(self, X_trains, cond_labels):
        for i in tf.range(self.tf_n_disc):
            with tf.GradientTape() as disc_tape:
                (D_loss_curr, D_fake) = self.D_loss(tf.gather(X_trains, i), tf.gather(cond_labels, i))
                gradients_of_discriminator = disc_tape.gradient(D_loss_curr, self.D.trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

        last_index = tf.subtract(self.tf_n_disc, 1)
        with tf.GradientTape() as gen_tape:
            # Need to recompute D_fake, otherwise gen_tape doesn't know the history
            (D_loss_curr, D_fake) = self.D_loss(tf.gather(X_trains, last_index), tf.gather(cond_labels, last_index))
            G_loss_curr = self.G_loss(D_fake)
            gradients_of_generator = gen_tape.gradient(G_loss_curr, self.G.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
            return D_loss_curr, G_loss_curr

    def train(self, trainingInputs, dataParamaters, eta_cond):
        checkpoint_dir = trainingInputs.GAN_dir + "/%s/checkpoints_eta_%s_%s/" % (self.voxInputs.particle,self.voxInputs.eta_min,self.voxInputs.eta_max,)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print("training started")
        print("Memory required before loading data: " + str(getrusage(RUSAGE_SELF).ru_maxrss))
        dl = DataLoader(self.voxInputs, dataParamaters, eta_cond)

        print("Memory required after loading data: " + str(getrusage(RUSAGE_SELF).ru_maxrss))

        D_loss_iter, G_loss_iter, Epochs = [], [], []
        time_iter, mem_iter = [], []

        if trainingInputs.start_epoch > 0:
            try:
                print("Try to load starting model %d" % (trainingInputs.start_epoch))
                iepoch = str(int(trainingInputs.start_epoch / 1000))
                print("convert trainingInputs.start_epoch ",trainingInputs.start_epoch,iepoch,)
                print("before loading checkpoint", self.G.layers[-1].weights)
                self.saver.restore("%s/model-%s" % (checkpoint_dir, iepoch))
                print("after loading checkpoint", self.G.layers[-1].weights)
                D_loss_iter = np.loadtxt(checkpoint_dir + "/d_loss.txt").tolist()[: trainingInputs.start_epoch]
                G_loss_iter = np.loadtxt(checkpoint_dir + "/g_loss.txt").tolist()[: trainingInputs.start_epoch]
                Epochs = list(range(trainingInputs.start_epoch))
                trainingInputs.start_epoch = trainingInputs.start_epoch + 1

            except:
                print("Error while loading checkpoint", sys.exc_info()[0])
                raise

        ind_of_exp = 1

        print("Training is done using all sample together")
        self.exp_max = dataParamaters.max_expE
        self.exp_min = dataParamaters.min_expE

        s_time = time.time()
        dur_train_loop = dur_getTrainData_ultimate = dur_convert = 0
        D_loss_curr, G_loss_curr = 0.0, 0.0
        existing_models = glob.glob(checkpoint_dir + "/model*.index")
        existing_models.sort(key=lambda f: int(re.sub('\D', '', f)))
        existing_models = [m[:-6] for m in existing_models] # remove .index suffix from the name
        for epoch in range(trainingInputs.start_epoch, trainingInputs.max_epochs + 1):

            if epoch == 0:
                print("Model and loss values will be saved every " + str(trainingInputs.sample_interval) + " epochs, the loss will also be plotted.")

            if epoch % trainingInputs.sample_interval == 0:
                if len(existing_models) > 1:
                    self.saver.restore(existing_models[0])
                    print("Iter: {} skip, load {}".format(epoch, existing_models[0]))
                    print("Iter: {}; D loss: {:.4f}; G_loss: {:.4f}; DataPrep: {:.4f}, Convert1: {:.4f} ".format(epoch, D_loss_curr, G_loss_curr, dur_getTrainData_ultimate, dur_convert,))
                    dur_getTrainData_ultimate = dur_convert = 0.0
                    existing_models.remove(existing_models[0])
                else:
                    e_time = time.time()
                    self.saver.save(file_prefix=checkpoint_dir + "/model")
                    save_time = time.time() - e_time

                    e_time = time.time()
                    time_diff = e_time - s_time
                    s_time = e_time
                    memory = getrusage(RUSAGE_SELF).ru_maxrss

                    time_iter.append(time_diff / trainingInputs.sample_interval)
                    mem_iter.append(memory)

                    print("Iter: {}; D loss: {:.4f}; G_loss: {:.4f}; TotalTime: {:.4f}; DataPrep: {:.4f}, Convert1: {:.4f}, TrainLoop: {:.4f}, Save: {:.4}; Mem: {}".format(epoch,D_loss_curr,G_loss_curr,time_diff,dur_getTrainData_ultimate,dur_convert,dur_train_loop,save_time,memory,)
                    )
                    dur_train_loop = dur_getTrainData_ultimate = dur_convert = 0.0


            self.X, self.Labels = dl.getAllTrainData(self.exp_min, self.exp_max)
            print("Memory required after getting data: " + str(getrusage(RUSAGE_SELF).ru_maxrss))
            if isinstance(self.X, list):
                self.X = tf.convert_to_tensor(self.X, dtype=tf.float32)
            if isinstance(self.Labels, list):
                self.Labels = tf.convert_to_tensor(self.Labels, dtype=tf.float32)
            print("Energies shape", self.X.shape)
            print("Labels shape", self.Labels.shape)
            remained_epoch = tf.constant(trainingInputs.max_epochs - epoch, dtype=tf.int64)
            dur_getTrainData_ultimate_start = time.time()
            self.getTrainData_ultimate(remained_epoch)
            dur_getTrainData_ultimate_stop = time.time()
            dur_getTrainData_ultimate += dur_getTrainData_ultimate_stop - dur_getTrainData_ultimate_start
            print("Epoch: " + str(epoch))
            print("Loading new data using indexes from " + str(self.exp_min) + " to " + str(self.exp_max))

            dur_convert_start = time.time()
            X, Labels = self.ds_iter.get_next()
            if len(existing_models) > 1:
                dur_convert_stop = time.time()
                dur_convert += dur_convert_stop - dur_convert_start
                continue
            else:
                X_trains = tf.reshape(X, self.X_batch_shape)
                cond_labels = tf.reshape(Labels, self.Labels_batch_shape)
                dur_convert_stop = time.time()
                dur_convert += dur_convert_stop - dur_convert_start

                train_loop_start = time.time()
                D_loss_curr, G_loss_curr = self.train_loop(X_trains, cond_labels)
                train_loop_stop = time.time()

                dur_train_loop += train_loop_stop - train_loop_start

                G_loss_iter.append(G_loss_curr.numpy())
                D_loss_iter.append(-D_loss_curr.numpy())

                Epochs.append(epoch)

        self.G_losses = G_loss_iter
        self.D_losses = D_loss_iter
        self.Epochs = Epochs

        np.savetxt(checkpoint_dir + "/d_loss.txt", D_loss_iter)
        np.savetxt(checkpoint_dir + "/g_loss.txt", G_loss_iter)
        np.savetxt(checkpoint_dir + "/time_per_epoch.txt", time_iter)
        np.savetxt(checkpoint_dir + "/memory.txt", mem_iter)

        self.plot_loss(trainingInputs)
        return

    def plot_loss(self, trainingInputs):
        import matplotlib.pyplot as plt

        loss_dir = trainingInputs.GAN_dir + "/%s/G_D_loss_iter_eta_%s_%s/" % (
            self.voxInputs.particle,
            self.voxInputs.eta_min,
            self.voxInputs.eta_max,
        )
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
        ax = plt.gca()
        ax.set_xlim(0, 1.1 * self.Epochs[-1])
        ax.cla()
        if trainingInputs.sample_interval < 1000:
            ax.plot(self.Epochs, self.G_losses, label="Generator")
            ax.plot(self.Epochs, self.D_losses, label="Discriminator")
        else:
            ax.plot(self.Epochs[1000:], self.G_losses[1000:], label="Generator")
            ax.plot(self.Epochs[1000:], self.D_losses[1000:], label="Discriminator")
        ax.set_xlabel("Epoch", fontsize=15)
        ax.set_ylabel("Wasserstein Loss", fontsize=15)
        ax.grid(True)
        ax.legend(fontsize=20)
        plt.savefig(loss_dir + "loss.pdf")

    def load(self, epoch, labels, nevents, input_dir_gan, ischeck=False):
        print("%s/model-%s" % (input_dir_gan, epoch))
        self.saver.restore("%s/model-%s" % (input_dir_gan, epoch)).expect_partial()
        if ischeck:
            return 0
        z = tf.random.normal(
            [nevents, self.ganParameters.latent_dim],
            mean=0.5,
            stddev=0.5,
            dtype=tf.dtypes.float32,
        )
        x_fake = self.G(inputs=[z, labels])
        return x_fake
