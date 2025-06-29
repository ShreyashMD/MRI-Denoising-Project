import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    Conv3DTranspose,
    LeakyReLU,
    BatchNormalization,
    Concatenate,
    Layer,
    Flatten,
    Dense,
)
from tensorflow.keras.models import Model
import nibabel as nib
import matplotlib.pyplot as plt


# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


def load_fmri_data(file_path):
    """Load fMRI data from a .nii.gz file."""
    print(f"Loading fMRI data from {file_path}")
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    print(f"fMRI data shape: {data.shape}")
    return data, affine


def add_noise(data, noise_factor=0.1):
    """Add Gaussian noise to the data."""
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
    return data + noise


def preprocess_data(data, patch_size, noise_factor=0.1):
    """Normalize data and extract 3D patches with optional noise."""
    data = data.astype(np.float32)
    mean = np.mean(data)
    std = np.std(data)
    data_norm = (data - mean) / (std + 1e-8)
    noisy_data = add_noise(data_norm, noise_factor=noise_factor)

    clean_patches = []
    noisy_patches = []

    if data_norm.ndim == 4:
        time_dim = data_norm.shape[3]
    else:
        time_dim = 1
        data_norm = data_norm[..., np.newaxis]
        noisy_data = noisy_data[..., np.newaxis]

    for t in range(time_dim):
        clean_vol = data_norm[..., t]
        noisy_vol = noisy_data[..., t]
        for x in range(0, clean_vol.shape[0] - patch_size[0] + 1, patch_size[0]):
            for y in range(0, clean_vol.shape[1] - patch_size[1] + 1, patch_size[1]):
                for z in range(0, clean_vol.shape[2] - patch_size[2] + 1, patch_size[2]):
                    patch_clean = clean_vol[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                    patch_noisy = noisy_vol[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                    clean_patches.append(patch_clean)
                    noisy_patches.append(patch_noisy)

    clean_patches = np.array(clean_patches)[..., np.newaxis]
    noisy_patches = np.array(noisy_patches)[..., np.newaxis]
    norm_params = {"mean": mean, "std": std}
    return clean_patches, noisy_patches, norm_params


# === Model definitions (Generator, Critic, UWGAN) ===

def build_generator(input_shape):
    inputs = Input(shape=input_shape)
    min_dim = min(input_shape[0], input_shape[1], input_shape[2])
    max_depth = 0
    while min_dim > 4:
        min_dim //= 2
        max_depth += 1
    skips = []
    x = inputs
    n_filters = 16
    for i in range(max_depth - 1):
        x = Conv3D(n_filters * (2 ** i), (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        skips.append(x)
    bottleneck_filters = n_filters * (2 ** (max_depth - 1))
    x = Conv3D(bottleneck_filters, (3, 3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    for i in range(max_depth - 1):
        f = n_filters * (2 ** (max_depth - 2 - i))
        x = Conv3DTranspose(f, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = keras.activations.relu(x)
        if i < len(skips):
            x = Concatenate()([x, skips[-(i + 1)]])
    output = Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=output, name="Generator")


def build_critic(input_shape):
    inputs = Input(shape=input_shape)
    min_dim = min(input_shape[0], input_shape[1], input_shape[2])
    max_depth = 0
    while min_dim > 2:
        min_dim //= 2
        max_depth += 1
    n_filters = 16
    x = inputs
    for i in range(max_depth):
        x = Conv3D(
            n_filters * (2 ** i),
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2) if i < max_depth - 1 else (1, 1, 1),
            padding="same",
        )(x)
        if i > 0:
            x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.2)(x)
    x = Flatten()(x)
    output = Dense(1)(x)
    return Model(inputs=inputs, outputs=output, name="Critic")


class GradientPenaltyLayer(Layer):
    def __init__(self, critic, gp_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.critic = critic
        self.gp_weight = gp_weight

    def call(self, inputs):
        real_images, fake_images = inputs
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        gradients = tape.gradient(pred, interpolated)
        gradients_squared = tf.square(gradients)
        gradients_squared_sum = tf.reduce_sum(gradients_squared, axis=[1, 2, 3, 4])
        gradient_l2_norm = tf.sqrt(gradients_squared_sum)
        gradient_penalty = self.gp_weight * tf.square(gradient_l2_norm - 1.0)
        return tf.reduce_mean(gradient_penalty)


class UWGAN(keras.Model):
    def __init__(self, input_shape, critic_extra_steps=3, gp_weight=10.0):
        super().__init__()
        self.generator = build_generator(input_shape)
        self.critic = build_critic(input_shape)
        self.critic_extra_steps = critic_extra_steps
        self.gp_weight = gp_weight
        self.gp_layer = GradientPenaltyLayer(self.critic, gp_weight)
        self.generator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
        self.generator.summary()
        self.critic.summary()

    def compile(self, **kwargs):
        super().compile(**kwargs)

    @tf.function
    def train_critic(self, noisy_images, real_images):
        noisy_images = tf.cast(noisy_images, tf.float32)
        real_images = tf.cast(real_images, tf.float32)
        with tf.GradientTape() as tape:
            fake_images = self.generator(noisy_images, training=True)
            real_output = self.critic(real_images, training=True)
            fake_output = self.critic(fake_images, training=True)
            critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = self.gp_layer([real_images, fake_images])
            critic_loss += gp
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        return critic_loss

    @tf.function
    def train_generator(self, noisy_images):
        noisy_images = tf.cast(noisy_images, tf.float32)
        with tf.GradientTape() as tape:
            fake_images = self.generator(noisy_images, training=True)
            fake_output = self.critic(fake_images, training=True)
            gen_loss = -tf.reduce_mean(fake_output)
            l1_loss = tf.reduce_mean(tf.abs(fake_images - noisy_images)) * 100.0
            total_gen_loss = gen_loss + l1_loss
        gradients = tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return total_gen_loss, gen_loss, l1_loss

    @tf.function
    def train_step(self, data):
        noisy_images, real_images = data
        for _ in range(self.critic_extra_steps):
            c_loss = self.train_critic(noisy_images, real_images)
        g_loss, g_wasserstein, g_l1 = self.train_generator(noisy_images)
        return {
            "critic_loss": c_loss,
            "gen_loss": g_loss,
            "wasserstein_loss": g_wasserstein,
            "l1_loss": g_l1,
        }


# === Training utilities ===

def train_model(model, noisy_patches, clean_patches, batch_size=4, epochs=20):
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((noisy_patches, clean_patches))
        .shuffle(buffer_size=len(noisy_patches))
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    class VisualizeCallback(keras.callbacks.Callback):
        def __init__(self, noisy_patches, clean_patches):
            super().__init__()
            n_samples = min(2, len(noisy_patches))
            vis_idx = np.random.randint(0, len(noisy_patches), n_samples)
            self.vis_noisy = noisy_patches[vis_idx]
            self.vis_clean = clean_patches[vis_idx]

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0 or epoch == 0:
                denoised_vis = self.model.generator.predict(self.vis_noisy)
                plt.figure(figsize=(12, 8))
                for i in range(len(self.vis_noisy)):
                    slice_idx = self.vis_noisy[i].shape[2] // 2
                    plt.subplot(3, len(self.vis_noisy), i + 1)
                    plt.imshow(self.vis_noisy[i, :, :, slice_idx, 0], cmap="gray")
                    plt.title("Noisy")
                    plt.axis("off")
                    plt.subplot(3, len(self.vis_noisy), i + 1 + len(self.vis_noisy))
                    plt.imshow(denoised_vis[i, :, :, slice_idx, 0], cmap="gray")
                    plt.title("Denoised")
                    plt.axis("off")
                    plt.subplot(3, len(self.vis_noisy), i + 1 + 2 * len(self.vis_noisy))
                    plt.imshow(self.vis_clean[i, :, :, slice_idx, 0], cmap="gray")
                    plt.title("Clean")
                    plt.axis("off")
                plt.tight_layout()
                plt.savefig(f"denoised_epoch_{epoch + 1}.png")
                plt.close()

    model.compile()
    vis_callback = VisualizeCallback(noisy_patches, clean_patches)
    history = model.fit(train_dataset, epochs=epochs, callbacks=[vis_callback], verbose=1)
    return history.history


# === Command line interface ===

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the MRI denoising model")
    parser.add_argument("nifti_file", help="Path to a .nii.gz file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=10, help="Training batch size")
    args = parser.parse_args()

    data, _ = load_fmri_data(args.nifti_file)
    patch_size = (32, 32, 8)
    clean_patches, noisy_patches, _ = preprocess_data(data, patch_size)
    if len(clean_patches) == 0:
        print("Error: No patches were created. Cannot train the model.")
        return
    patch_shape = clean_patches[0].shape
    print(f"Training with patch shape: {patch_shape}")
    model = UWGAN(patch_shape, critic_extra_steps=2, gp_weight=10.0)
    history = train_model(model, noisy_patches, clean_patches, batch_size=args.batch_size, epochs=args.epochs)
    model.generator.save("fmri_denoiser_generator.h5")
    model.critic.save("fmri_critic.h5")
    plt.figure(figsize=(12, 8))
    metrics = ["critic_loss", "gen_loss", "wasserstein_loss", "l1_loss"]
    for i, metric in enumerate(metrics):
        if metric in history:
            plt.subplot(2, 2, i + 1)
            plt.plot(history[metric])
            plt.title(metric.replace("_", " ").title())
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("training_losses.png")
    plt.close()
    print("Model training completed and saved!")


if __name__ == "__main__":
    main()
