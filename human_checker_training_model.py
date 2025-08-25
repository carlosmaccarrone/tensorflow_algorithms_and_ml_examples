from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np

# -------------------------
# 1 Load the data
# -------------------------
circle_data = np.load("circle.npy")    # shape: [num_samples,28,28] or [num_samples,28,28,1]
square_data = np.load("square.npy")
triangle_data = np.load("triangle.npy")

# Concatenate all
X = np.concatenate([circle_data, square_data, triangle_data], axis=0)

# Create labels: 0=circle, 1=square, 2=triangle
y = np.array([0]*len(circle_data) + [1]*len(square_data) + [2]*len(triangle_data))

print("Shapes:", X.shape, y.shape)

# -------------------------
# 2 Preprocessing
# -------------------------
def preprocess(images):
    # Make sure it has channel (H,W,1)
    if len(images.shape) == 3:
        images = images[..., np.newaxis]
    images = images.astype('float32') / 255.0  # normalize 0-1
    images = 1.0 - images                       # invert colors: background=0, figure=1
    # Optional binarization
    images = (images > 0.5).astype('float32')
    return images

X = preprocess(X)

X = X.reshape(-1, 28, 28, 1)

# -------------------------
# 3 Train/Test split
# -------------------------
train_images, test_images, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Train:", train_images.shape, train_labels.shape)
print("Test:", test_images.shape, test_labels.shape)

# -------------------------
# 4 Build CNN model
# -------------------------
def build_model(input_shape=(28,28,1), num_classes=3):
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    
    # Flatten + Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

model = build_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------
# 5️ Callbacks
# -------------------------
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

# -------------------------
# 6 Training
# -------------------------
history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    batch_size=64,
                    epochs=50,
                    callbacks=callbacks)

# -------------------------
# 7 Guardar modelo
# -------------------------
model.save("shapes_model.h5")
print("Model saved as shapes_model.h5 ready to use!")