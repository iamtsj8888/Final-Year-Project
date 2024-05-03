import tensorflow as tf
import cv2

# Load the input image
input_image = cv2.imread("face1/1.png")

# Convert the image to a tensor
input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension

filters=1
kernel_size=(8,8)
strides=(8,8)
padding="valid"
activation=None

def transposed_conv_layer(input_tensor):
    transposed_image =  tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=(8,8),
        strides=(8,8),
        padding="valid",
        activation=None
    )(input_tensor)
    return transposed_image

def upsample_and_conv(input_tensor):
    x = tf.keras.layers.UpSampling2D(size=strides)(input_tensor)
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='valid')(x)
    return x

upsampled_tensor = tf.keras.layers.UpSampling2D(size=(8,8))(input_tensor)

# Apply the transposed convolution layer
upsampled_tensor = transposed_conv_layer(input_tensor)

# Convert the upsampled tensor back to an image
upsampled_image = tf.squeeze(upsampled_tensor, axis=0)
upsampled_image = tf.cast(upsampled_image, tf.uint8)
upsampled_image = upsampled_image.numpy()

# Save the upsampled image
cv2.imwrite("transposedConv2d.jpg", upsampled_image)