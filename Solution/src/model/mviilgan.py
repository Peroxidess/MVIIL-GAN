import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras import regularizers


def get_discriminator_model(shape_x, model_dim, coder_stack, output_dim, flag_mask_cat=True, name="discriminator"):
    inputs_x_discr = Input(shape=(shape_x,), dtype='float32')
    masks_dicri_input = Input(shape=(shape_x))
    masks_dicri = K.cast(masks_dicri_input, 'float32')
    if flag_mask_cat:
        input_x = K.concatenate([inputs_x_discr, masks_dicri])
    else:
        input_x = inputs_x_discr
    for i in range(coder_stack):
        input_x = Dense(model_dim, activation=tf.nn.gelu, kernel_regularizer=regularizers.l2(0.01))(input_x)
    if '_m' in name:
        output_discr = Dense(output_dim)(input_x)
    else:
        output_discr = Dense(output_dim)(input_x)
    model_d = Model([inputs_x_discr, masks_dicri_input], output_discr, name=name)
    return model_d


def get_generator_model(shape_x, model_dim, feed_forward_size, coder_stack, flag_mask_cat=True, name="generator"):
    inputs_x_discr = Input(shape=(shape_x,), dtype='float32')
    masks_encoder_input = Input(shape=(shape_x))
    masks_decoder_input = Input(shape=(shape_x))
    masks_dicri = K.cast(masks_encoder_input, 'float32')
    if flag_mask_cat:
        input_x = K.concatenate([inputs_x_discr, masks_dicri])
    else:
        input_x = inputs_x_discr
    for i in range(coder_stack):
        input_x = Dense(model_dim, activation=tf.nn.gelu)(input_x)
        input_x = BatchNormalization()(input_x)
    data_gen = Dense(feed_forward_size)(input_x)
    model_g = Model([inputs_x_discr, masks_encoder_input, masks_decoder_input], [data_gen, data_gen, data_gen], name=name)
    return model_g


class MVIILGAN(Model):
    def __init__(
        self,
        discriminator_m,
        discriminator_s,
        generator,
        latent_dim,
        discriminator_extra_steps_s=1,
        discriminator_extra_steps_m=1,
        generator_extra_steps=1,
        reconstruction_extra_steps=0,
        class_extra_steps=1,
        gp_weight=1,
    ):
        super(MVIILGAN, self).__init__()
        self.discriminator_m = discriminator_m
        self.discriminator_s = discriminator_s
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps_s = discriminator_extra_steps_s
        self.d_steps_m = discriminator_extra_steps_m
        self.g_steps = generator_extra_steps
        self.rec_steps = reconstruction_extra_steps
        self.class_steps = class_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer_m, d_optimizer_s, g_optimizer, rec_optimizer, class_optimizer,
                d_loss_fn, g_loss_fn, reconstruction_loss_fn, class_loss_fn, class_exfeat_fn):
        super(MVIILGAN, self).compile()
        self.d_optimizer_m = d_optimizer_m
        self.d_optimizer_s = d_optimizer_s
        self.g_optimizer = g_optimizer
        self.rec_optimizer = rec_optimizer
        self.class_optimizer = class_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.class_loss_fn = class_loss_fn
        self.class_exfeat_fn = class_exfeat_fn

    def gradient_penalty(self, discriminator, mask, batch_size, data_real, data_fake):
        # Get the interpolated data
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = data_fake - data_real
        interpolated = data_real + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated data.
            pred = discriminator([interpolated, mask], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated data.
        grads = gp_tape.gradient(pred, [interpolated])
        grads_ = grads[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads_), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, list_data):
        if isinstance(list_data, tuple):
            list_data = list_data[0]
        input_x = list_data[0]
        index_nan = list_data[1]
        mask_hint = list_data[2]
        data_fake = list_data[3]
        index_nan_fake = list_data[4]
        mask_hint_fake = list_data[5]

        # Get the batch size
        batch_size = tf.shape(input_x)[0]
        input_dim = tf.shape(input_x)[1]
        masks = K.cast(index_nan, 'float32') #True -> 1 : notna
        masks_fake = K.cast(index_nan_fake, 'float32') #True -> 1 : notna

        d_cost_s = 0
        d_cost_m = 0
        g_loss = 0
        reconstruction_loss = 0

        # Reconstruction task
        for i in range(self.rec_steps):
            with tf.GradientTape() as tape:
                data_generated, mean, logvar = self.generator([input_x, index_nan, index_nan], training=True)
                reconstruction_loss = self.reconstruction_loss_fn(input_x, data_generated, masks)
            rec_gradient = tape.gradient(reconstruction_loss, self.generator.trainable_variables)
            self.rec_optimizer.apply_gradients(zip(rec_gradient, self.generator.trainable_variables))

        # Train the generator
        for i in range(self.g_steps):
            # Get the latent vector
            with tf.GradientTape() as tape:
                # Generate fake data using the generator
                data_generated, mean, logvar = self.generator([input_x, index_nan, index_nan], training=True)
                data_generated_fake, mean_fake, logvar_fake = self.generator([data_fake, index_nan_fake, index_nan_fake], training=True)
                # Get the discriminator logits for fake data
                input_discri = tf.add(tf.multiply((1 - masks), data_generated), tf.multiply(masks, input_x))
                data_gen_logits_m = self.discriminator_m([input_discri, mask_hint], training=True)
                data_gen_logits_m = tf.multiply((1 - masks), data_gen_logits_m)
                data_gen_logits_m_fake = self.discriminator_m([data_generated_fake, mask_hint_fake], training=True)
                data_gen_logits_m_fake = tf.multiply((1 - masks_fake), data_gen_logits_m_fake)
                data_gen_logits_s_fake = self.discriminator_s([data_generated_fake, index_nan_fake], training=True)
                # Calculate the generator loss
                g_loss_m = self.g_loss_fn(data_gen_logits_m, None, index_nan, flag_='m')
                g_loss_m_fake = self.g_loss_fn(data_gen_logits_m_fake, None, index_nan_fake, flag_='m')
                g_loss_s_fake = self.g_loss_fn(data_gen_logits_s_fake)
                g_loss = g_loss_m + g_loss_m_fake + g_loss_s_fake

            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(
                zip(gen_gradient, self.generator.trainable_variables)
            )

        for i in range(self.d_steps_m):
            with tf.GradientTape() as tape:
                # Generate fake from the latent vector
                data_fake_m, _, __ = self.generator([input_x, index_nan, index_nan], training=True)
                input_discri_m = tf.add(tf.multiply((1 - masks), data_fake_m), tf.multiply(masks, input_x))
                # Get the logits for the fake
                fake_logits_m = self.discriminator_m([input_discri_m, mask_hint], training=True)
                fake_logits_m_0 = tf.multiply((1 - masks), fake_logits_m)
                fake_logits_m_1 = tf.multiply(masks, fake_logits_m)
                # Get the logits for the real
                # Calculate the discriminator loss using the fake and real data logits
                d_cost_m_fake = self.d_loss_fn(fake_logits_m_1, fake_logits_m_0, None, index_nan, flag_='single_1')
                d_cost_m_fake2 = self.d_loss_fn(fake_logits_m_1, fake_logits_m_0, None, index_nan, flag_='single_0')
                # Calculate the gradient penalty
                gp_m = self.gradient_penalty(self.discriminator_m, mask_hint, batch_size, list_data[0], input_discri_m)
                # Add the gradient penalty to the original discriminator loss
                d_cost_m = d_cost_m_fake + d_cost_m_fake2 + gp_m * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient_m = tape.gradient(d_cost_m, self.discriminator_m.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer_m.apply_gradients(
                zip(d_gradient_m, self.discriminator_m.trainable_variables)
            )

        for i in range(self.d_steps_s):
            with tf.GradientTape() as tape:
                # Generate fake from the latent vector
                data_fake_s, _, __ = self.generator([input_x, index_nan, index_nan], training=True)
                data_fake_s_fake, _, __ = self.generator([data_fake, index_nan_fake, index_nan_fake], training=True)
                input_discri_s = tf.add(tf.multiply((1 - masks), data_fake_s), tf.multiply(masks, input_x))
                input_discri_s_fake = tf.add(tf.multiply((1 - masks_fake), data_fake_s_fake), tf.multiply(masks_fake, data_fake))
                real_logits_s = self.discriminator_s([input_discri_s, index_nan], training=True)
                fake_logits_s = self.discriminator_s([input_discri_s_fake, index_nan_fake], training=True)
                d_cost_s = self.d_loss_fn(real_logits_s, fake_logits_s, None, index_nan, flag_='')
                # Calculate the gradient penalty
                gp_s = self.gradient_penalty(self.discriminator_s, index_nan,
                                             batch_size, list_data[0], input_discri_s)
                # Add the gradient penalty to the original discriminator loss
                d_cost_s = d_cost_s + gp_s * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient_s = tape.gradient(d_cost_s, self.discriminator_s.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer_s.apply_gradients(
                zip(d_gradient_s, self.discriminator_s.trainable_variables)
            )

        return {
            "d_loss_s": d_cost_s,
            "d_loss_m": d_cost_m,
            "g_loss": g_loss,
            'rec_loss': reconstruction_loss,
            'fake': -g_loss,
            'real': -d_cost_s - g_loss,
            }

    def call(self, list_data):
        input_x = list_data[0]
        index_notna = list_data[1]
        mask_hint = list_data[2]
        data_ger, _, _ = self.generator([input_x, index_notna, index_notna], training=False)
        masks = K.cast(index_notna, 'float32')
        data_ger_ = tf.add(tf.multiply((1 - masks), data_ger), tf.multiply(masks, input_x))
        logits_fake_m = self.discriminator_m([data_ger_, mask_hint], training=False)
        logits_real_m = self.discriminator_m([input_x, mask_hint], training=False)
        logits_fake = self.discriminator_s([data_ger_, index_notna], training=False)
        logits_real = self.discriminator_s([input_x, index_notna], training=False)
        return data_ger_, logits_fake, logits_real, logits_fake_m, logits_real_m, masks


class GAN_MVIIL():
    def __init__(self, shape_x, feed_forward_size, model_dim=64, coder_stack=2, noise_dim=128, ablation_sel=None,
                 seed=2022):
        self.seed = seed
        tf.random.set_seed(seed)
        generator_optimizer = keras.optimizers.RMSprop(learning_rate=1e-4)
        discriminator_optimizer_m = keras.optimizers.RMSprop(learning_rate=1e-6)
        discriminator_optimizer_s = keras.optimizers.RMSprop(learning_rate=1e-5)
        rec_optimizer = keras.optimizers.RMSprop(learning_rate=3e-2)
        class_optimizer = keras.optimizers.RMSprop(learning_rate=5e-4)

        # Define the loss functions for the discriminator,
        def discriminator_loss(real_logits, fake_logits, axis=None, index_nan=None, flag_=''):
            if 'single_1' in flag_:
                mask = tf.cast(index_nan, 'float32')
                mask_col_count = tf.reduce_mean(mask)
                loss_ = -tf.divide(tf.reduce_mean(real_logits), mask_col_count)
            elif 'single_0' in flag_:
                mask = tf.cast(index_nan, 'float32')
                mask_col_count = tf.reduce_mean(mask)
                loss_ = tf.divide(tf.reduce_mean(tf.multiply((1 - mask), fake_logits)), mask_col_count)
            else:
                real_loss = tf.reduce_mean(real_logits, axis=axis)
                fake_loss = tf.reduce_mean(fake_logits, axis=axis)
                loss_ = fake_loss-real_loss
            return loss_

        # Define the loss functions for the generator.
        def generator_loss(logits_fake, axis=None, index_nan=None, flag_=''):
            if 'm' in flag_:
                mask_col_count0 = tf.reduce_sum(K.cast(tf.equal(index_nan, False), 'float32'), axis=axis)
                loss_ = -tf.divide(tf.reduce_sum(logits_fake, axis=axis), mask_col_count0)
            else:
                loss_ = -tf.reduce_mean(logits_fake)
            return loss_

        # Define the loss functions for reconstruction.
        def rec_loss(input_x, data_generated, masks):
            loss_rec = keras.losses.mean_squared_error(input_x * masks, data_generated * masks)
            return tf.reduce_mean(loss_rec)

        def class_loss(true, pred):
            if K.dtype(pred) != 'float32': pred = K.cast(pred, 'float32')
            if K.dtype(true) != 'float32': true = K.cast(true, 'float32')
            loss_class = tf.reduce_mean(keras.losses.binary_crossentropy(true, pred))
            return loss_class

        def class_exfeat_fn(exfeat_real, exfeat_fake):
            loss_ = tf.reduce_mean(keras.losses.mean_squared_error(exfeat_real, exfeat_fake))
            return loss_

        model_g = get_generator_model(shape_x, model_dim, feed_forward_size, coder_stack, flag_mask_cat=True)
        model_g.summary()
        model_dm = get_discriminator_model(shape_x, model_dim, coder_stack, output_dim=shape_x, flag_mask_cat=True,
                                           name='discriminator_m')
        model_dm.summary()
        model_ds = get_discriminator_model(shape_x, model_dim, coder_stack, output_dim=1, flag_mask_cat=False,
                                           name='discriminator_s')
        model_ds.summary()

        if 'disc_s&ae_not' in ablation_sel:
            disc_ex_step_s = 0
            disc_ex_step_m = 1
            ger_ex_step_s = 1
            rec_ex_step_s = 0
        elif 'ae_not' in ablation_sel:
            disc_ex_step_s = 1
            disc_ex_step_m = 1
            ger_ex_step_s = 1
            rec_ex_step_s = 0
        elif 'disc_s_not' in ablation_sel:
            disc_ex_step_s = 0
            disc_ex_step_m = 1
            ger_ex_step_s = 1
            rec_ex_step_s = 1
        elif 'ae_only' in ablation_sel:
            disc_ex_step_s = 0
            disc_ex_step_m = 0
            ger_ex_step_s = 1
            rec_ex_step_s = 1
        else:
            disc_ex_step_s = 1
            disc_ex_step_m = 1
            ger_ex_step_s = 1
            rec_ex_step_s = 1
        # Instantiate the MVIILGAN model.
        self.model = MVIILGAN(
            discriminator_m=model_dm,
            discriminator_s=model_ds,
            generator=model_g,
            latent_dim=noise_dim,
            discriminator_extra_steps_s=disc_ex_step_s,
            discriminator_extra_steps_m=disc_ex_step_m,
            generator_extra_steps=ger_ex_step_s,
            reconstruction_extra_steps=rec_ex_step_s,
            class_extra_steps=0,
        )

        # Compile the MVIILGAN model.
        self.model.compile(
            d_optimizer_m=discriminator_optimizer_m,
            d_optimizer_s=discriminator_optimizer_s,
            g_optimizer=generator_optimizer,
            rec_optimizer=rec_optimizer,
            class_optimizer=class_optimizer,
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss,
            reconstruction_loss_fn=rec_loss,
            class_loss_fn=class_loss,
            class_exfeat_fn=class_exfeat_fn,
        )

    def fit_transform(self, data_raw_nan, data_fake, index_nan, index_nan_fake, epochs=2, batch_size=512,
                     seed=2022, flag_save=True):
        tf.random.set_seed(seed)
        unif_random_matrix = np.random.uniform(0., 1., size=index_nan.shape)
        binary_random_matrix = 1 * (unif_random_matrix < 0.9)
        mask_hint = index_nan * binary_random_matrix
        unif_random_matrix = np.random.uniform(0., 1., size=index_nan_fake.shape)
        binary_random_matrix = 1 * (unif_random_matrix < 0.9)
        mask_hint_fake = index_nan_fake * binary_random_matrix
        model_save_path = "model_file_path"

        def scheduler(epoch):
            if epoch % 100 == 0 and epoch != 0:
                lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, lr * 0.1)
                print("lr changed to {}".format(lr * 0.1))
            return K.get_value(self.model.optimizer.lr)
        reduce_lr = LearningRateScheduler(scheduler)

        if not flag_save:
            self.model.load_weights(model_save_path)
        else:
            history_model = self.model.fit([data_raw_nan.astype('float32'),
                                            index_nan, mask_hint,
                                            data_fake.astype('float32'), index_nan_fake, mask_hint_fake],
                                           batch_size=batch_size, epochs=epochs,
                                           callbacks=[reduce_lr]
                                           )

        data_ger, logits_fake, logits_real, logits_fake_m, logits_real_m, masks = \
            self.model.predict([data_raw_nan, index_nan, mask_hint])
        data_fake_ger, logits_fake, logits_real, logits_fake_m, logits_real_m, masks = \
            self.model.predict([data_fake, index_nan_fake, mask_hint_fake])
        return data_ger, data_fake_ger

    def transform(self, data_raw_nan, index_notna):
        unif_random_matrix = np.random.uniform(0., 1., size=index_notna.shape)
        binary_random_matrix = 1 * (unif_random_matrix < 0.9)
        mask_hint = index_notna * binary_random_matrix
        data_ger, logits_fake, logits_real, logits_fake_m, logits_real_m, masks = self.model.predict([data_raw_nan, index_notna, mask_hint])
        return data_ger
