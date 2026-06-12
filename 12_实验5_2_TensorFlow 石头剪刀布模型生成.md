```python
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
print("Python版本:", sys.version)
print("TensorFlow版本:", tf.__version__)
print("NumPy版本:", np.__version__)
```

    Python版本: 3.10.20 (main, Mar 11 2026, 17:43:48) [Clang 20.1.8 ]
    TensorFlow版本: 2.16.2
    NumPy版本: 1.26.4



```python
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 解压本地数据集
def extract_zip(zip_name):
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall()
    print(f"{zip_name} 解压完成")

extract_zip("rps.zip")
extract_zip("rps-test-set.zip")

# 图像预处理
TRAINING_DIR = "./rps/"
VALIDATION_DIR = "./rps-test-set/"

training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=126
 )

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=126
 )

# 搭建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
 ])

model.summary()

# 编译训练
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=25,
    steps_per_epoch=20,
    validation_data = validation_generator,
    verbose = 1,
    validation_steps=3
 )

# 保存模型
model.save("rps.h5")
print("模型已保存为 rps.h5")

# 绘制准确率、损失曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.tight_layout()
plt.show()
```

    rps.zip 解压完成
    rps-test-set.zip 解压完成
    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_4"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_16 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">148</span>, <span style="color: #00af00; text-decoration-color: #00af00">148</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">1,792</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_16 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">74</span>, <span style="color: #00af00; text-decoration-color: #00af00">74</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_17 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │        <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_17 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_18 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">34</span>, <span style="color: #00af00; text-decoration-color: #00af00">34</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_18 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">17</span>, <span style="color: #00af00; text-decoration-color: #00af00">17</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_19 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">15</span>, <span style="color: #00af00; text-decoration-color: #00af00">15</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_19 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6272</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6272</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │     <span style="color: #00af00; text-decoration-color: #00af00">3,211,776</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">1,539</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,473,475</span> (13.25 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,473,475</span> (13.25 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    Epoch 1/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m17s[0m 600ms/step - accuracy: 0.3171 - loss: 1.5310 - val_accuracy: 0.3333 - val_loss: 1.0974
    Epoch 2/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 450ms/step - accuracy: 0.3571 - loss: 1.1385 - val_accuracy: 0.3333 - val_loss: 1.0954
    Epoch 3/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 425ms/step - accuracy: 0.3921 - loss: 1.0945 - val_accuracy: 0.5457 - val_loss: 1.0607
    Epoch 4/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 436ms/step - accuracy: 0.4373 - loss: 1.0706 - val_accuracy: 0.4301 - val_loss: 1.1108
    Epoch 5/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 444ms/step - accuracy: 0.4583 - loss: 1.0876 - val_accuracy: 0.3333 - val_loss: 1.4820
    Epoch 6/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 452ms/step - accuracy: 0.5298 - loss: 0.9225 - val_accuracy: 0.6344 - val_loss: 0.6001
    Epoch 7/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 474ms/step - accuracy: 0.6234 - loss: 0.8027 - val_accuracy: 0.5968 - val_loss: 0.8652
    Epoch 8/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 450ms/step - accuracy: 0.6484 - loss: 0.7756 - val_accuracy: 0.8118 - val_loss: 0.3413
    Epoch 9/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 442ms/step - accuracy: 0.7139 - loss: 0.7323 - val_accuracy: 0.7957 - val_loss: 0.3225
    Epoch 10/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 438ms/step - accuracy: 0.7329 - loss: 0.6355 - val_accuracy: 0.9570 - val_loss: 0.1762
    Epoch 11/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 481ms/step - accuracy: 0.7508 - loss: 0.6541 - val_accuracy: 1.0000 - val_loss: 0.1557
    Epoch 12/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 444ms/step - accuracy: 0.7540 - loss: 0.5961 - val_accuracy: 0.9032 - val_loss: 0.1774
    Epoch 13/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 438ms/step - accuracy: 0.7798 - loss: 0.6632 - val_accuracy: 0.7634 - val_loss: 0.5954
    Epoch 14/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 432ms/step - accuracy: 0.7389 - loss: 1.0128 - val_accuracy: 0.9462 - val_loss: 0.1958
    Epoch 15/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 442ms/step - accuracy: 0.7829 - loss: 0.6921 - val_accuracy: 0.9489 - val_loss: 0.1447
    Epoch 16/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 444ms/step - accuracy: 0.7968 - loss: 0.8522 - val_accuracy: 0.9677 - val_loss: 0.1237
    Epoch 17/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 481ms/step - accuracy: 0.7825 - loss: 0.9913 - val_accuracy: 0.9704 - val_loss: 0.0667
    Epoch 18/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 488ms/step - accuracy: 0.8290 - loss: 0.6596 - val_accuracy: 0.4677 - val_loss: 6.2793
    Epoch 19/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 604ms/step - accuracy: 0.7401 - loss: 1.6766 - val_accuracy: 0.9167 - val_loss: 0.2023
    Epoch 20/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 458ms/step - accuracy: 0.7722 - loss: 2.5066 - val_accuracy: 0.6667 - val_loss: 4.1090
    Epoch 21/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 451ms/step - accuracy: 0.7333 - loss: 3.2242 - val_accuracy: 0.7258 - val_loss: 2.3443
    Epoch 22/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 439ms/step - accuracy: 0.7254 - loss: 5.7837 - val_accuracy: 0.5242 - val_loss: 14.1668
    Epoch 23/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 515ms/step - accuracy: 0.7313 - loss: 10.6102 - val_accuracy: 0.8737 - val_loss: 1.3121
    Epoch 24/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 480ms/step - accuracy: 0.7063 - loss: 14.4960 - val_accuracy: 0.6667 - val_loss: 28.8476
    Epoch 25/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 447ms/step - accuracy: 0.7004 - loss: 31.9144 - val_accuracy: 0.6882 - val_loss: 10.5309


    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 


    模型已保存为 rps.h5



    
![png](12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_files/12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_1_9.png)
    



```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# 路径配置
TRAINING_DIR = "./rps/"
VALIDATION_DIR = "./rps-test-set/"

# 数据增强
training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)

# 加载数据
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=126
 )

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=126
 )

# 搭建和原教程一致的网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
 ])

# 替换优化器为Adam（解决训练发散）
model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 训练
history = model.fit(
    train_generator,
    epochs=25,
    steps_per_epoch=20,
    validation_data = validation_generator,
    verbose = 1,
    validation_steps=3
 )

# 保存标准keras格式（消除警告）
model.save("rps.keras")
print("模型已保存为 rps.keras")

# 绘制曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.tight_layout()
plt.show()
```

    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.
    Epoch 1/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 616ms/step - accuracy: 0.3817 - loss: 1.0991 - val_accuracy: 0.4892 - val_loss: 0.9537
    Epoch 2/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 504ms/step - accuracy: 0.5381 - loss: 0.9295 - val_accuracy: 0.8978 - val_loss: 0.4984
    Epoch 3/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 503ms/step - accuracy: 0.6770 - loss: 0.7171 - val_accuracy: 0.8844 - val_loss: 0.3502
    Epoch 4/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 479ms/step - accuracy: 0.7663 - loss: 0.5750 - val_accuracy: 0.9194 - val_loss: 0.2654
    Epoch 5/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 505ms/step - accuracy: 0.8389 - loss: 0.4233 - val_accuracy: 0.9785 - val_loss: 0.1090
    Epoch 6/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 475ms/step - accuracy: 0.8702 - loss: 0.3566 - val_accuracy: 0.9677 - val_loss: 0.1641
    Epoch 7/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 488ms/step - accuracy: 0.8861 - loss: 0.2972 - val_accuracy: 0.9812 - val_loss: 0.0675
    Epoch 8/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 460ms/step - accuracy: 0.8948 - loss: 0.2861 - val_accuracy: 0.9758 - val_loss: 0.0922
    Epoch 9/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 461ms/step - accuracy: 0.9198 - loss: 0.2351 - val_accuracy: 0.9570 - val_loss: 0.1514
    Epoch 10/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 462ms/step - accuracy: 0.9187 - loss: 0.2105 - val_accuracy: 0.9597 - val_loss: 0.0966
    Epoch 11/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 492ms/step - accuracy: 0.9310 - loss: 0.1931 - val_accuracy: 0.9758 - val_loss: 0.0907
    Epoch 12/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 456ms/step - accuracy: 0.9385 - loss: 0.1668 - val_accuracy: 0.9704 - val_loss: 0.1663
    Epoch 13/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 503ms/step - accuracy: 0.9373 - loss: 0.2040 - val_accuracy: 0.9651 - val_loss: 0.0907
    Epoch 14/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 461ms/step - accuracy: 0.9302 - loss: 0.1905 - val_accuracy: 0.9677 - val_loss: 0.0705
    Epoch 15/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 478ms/step - accuracy: 0.9361 - loss: 0.2079 - val_accuracy: 0.9597 - val_loss: 0.1183
    Epoch 16/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 484ms/step - accuracy: 0.9298 - loss: 0.2092 - val_accuracy: 0.9651 - val_loss: 0.0852
    Epoch 17/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 490ms/step - accuracy: 0.9298 - loss: 0.2255 - val_accuracy: 0.8253 - val_loss: 0.5530
    Epoch 18/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 468ms/step - accuracy: 0.8925 - loss: 0.3628 - val_accuracy: 0.9543 - val_loss: 0.1030
    Epoch 19/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 567ms/step - accuracy: 0.8813 - loss: 0.5765 - val_accuracy: 0.9516 - val_loss: 0.2818
    Epoch 20/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 533ms/step - accuracy: 0.8738 - loss: 0.5352 - val_accuracy: 0.9543 - val_loss: 0.1059
    Epoch 21/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 580ms/step - accuracy: 0.8210 - loss: 1.0731 - val_accuracy: 0.9704 - val_loss: 0.1175
    Epoch 22/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 468ms/step - accuracy: 0.7806 - loss: 1.2740 - val_accuracy: 0.7419 - val_loss: 1.0953
    Epoch 23/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 476ms/step - accuracy: 0.7075 - loss: 2.6194 - val_accuracy: 0.6532 - val_loss: 1.4310
    Epoch 24/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 468ms/step - accuracy: 0.7091 - loss: 3.5235 - val_accuracy: 0.8656 - val_loss: 0.4766
    Epoch 25/25
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 571ms/step - accuracy: 0.8254 - loss: 1.3779 - val_accuracy: 0.9409 - val_loss: 0.1541
    模型已保存为 rps.keras



    
![png](12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_files/12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_2_1.png)
    



```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 定义三类图片路径
rock_dir = os.path.join('./rps/rock')
paper_dir = os.path.join('./rps/paper')
scissors_dir = os.path.join('./rps/scissors')

# 过滤函数：只保留png图片，排除目录、隐藏文件
def get_image_files(folder):
    files = []
    for f in os.listdir(folder):
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path) and f.lower().endswith('.png'):
            files.append(f)
    return files

# 分别读取三个文件夹图片（修正变量错误）
rock_files = get_image_files(rock_dir)
paper_files = get_image_files(paper_dir)
scissors_files = get_image_files(scissors_dir)

# 打印统计信息（实验要求）
print("石头图片总数:", len(rock_files))
print("布图片总数:", len(paper_files))
print("剪刀图片总数:", len(scissors_files))
print("石头前10个文件名:", rock_files[:10])
print("布前10个文件名:", paper_files[:10])
print("剪刀前10个文件名:", scissors_files[:10])

# 每类选取2张图片展示
pic_index = 2
next_rock = [os.path.join(rock_dir, fname) for fname in rock_files[:pic_index]]
next_paper = [os.path.join(paper_dir, fname) for fname in paper_files[:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) for fname in scissors_files[:pic_index]]

# 循环展示图片
for img_path in next_rock + next_paper + next_scissors:
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
```

    石头图片总数: 840
    布图片总数: 840
    剪刀图片总数: 840
    石头前10个文件名: ['rock04-059.png', 'rock01-108.png', 'rock04-065.png', 'rock05ck01-067.png', 'rock05ck01-073.png', 'rock04-071.png', 'rock05ck01-098.png', 'rock02-008.png', 'rock07-k03-013.png', 'rock02-034.png']
    布前10个文件名: ['paper03-088.png', 'paper05-026.png', 'paper05-032.png', 'paper03-077.png', 'paper03-063.png', 'paper02-099.png', 'paper04-037.png', 'paper04-023.png', 'paper02-066.png', 'paper02-072.png']
    剪刀前10个文件名: ['testscissors03-040.png', 'testscissors03-054.png', 'testscissors03-068.png', 'testscissors03-083.png', 'testscissors03-097.png', 'scissors03-113.png', 'scissors03-107.png', 'testscissors02-051.png', 'testscissors02-045.png', 'scissors01-002.png']



    
![png](12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_files/12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_3_1.png)
    



    
![png](12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_files/12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_3_2.png)
    



    
![png](12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_files/12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_3_3.png)
    



    
![png](12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_files/12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_3_4.png)
    



    
![png](12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_files/12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_3_5.png)
    



    
![png](12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_files/12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_3_6.png)
    



```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# 加载模型
model = tf.keras.models.load_model("rps.keras")

# 标签顺序：paper(布)、rock(石头)、scissors(剪刀)
class_names = ["布", "石头", "剪刀"]

# 选用列表里真实的文件名 testrock01-00.png
img_path = "./rps-test-set/rock/testrock01-00.png"

# 图片预处理
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# 预测
predictions = model.predict(img_array)
pred_idx = np.argmax(predictions[0])

# 输出结果
print(f"预测结果：{class_names[pred_idx]}")
print(f"置信度：{np.max(predictions[0]) * 100:.2f} %")

# 展示图片
plt.imshow(img)
plt.axis("off")
plt.show()
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1s/step
    预测结果：石头
    置信度：100.00 %



    
![png](12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_files/12_%E5%AE%9E%E9%AA%8C5_2_TensorFlow%20%E7%9F%B3%E5%A4%B4%E5%89%AA%E5%88%80%E5%B8%83%E6%A8%A1%E5%9E%8B%E7%94%9F%E6%88%90_4_1.png)
    



```python
import tensorflow as tf

# 加载模型并转换为 TensorFlow Lite 格式
model = tf.keras.models.load_model("rps.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存文件
with open("rps_lite.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TensorFlow Lite 模型转换完成，生成 rps_lite.tflite")
```

    INFO:tensorflow:Assets written to: /var/folders/np/tw641bs56mx95tw421k_fhyc0000gn/T/tmpd_yx0g_r/assets


    INFO:tensorflow:Assets written to: /var/folders/np/tw641bs56mx95tw421k_fhyc0000gn/T/tmpd_yx0g_r/assets



```python
# 标准 TensorFlow Lite 模型转换代码（实验要求知识点）
import os
# 切换到模型所在目录
os.chdir("/Users/unnn")

# 禁用GPU尝试解决崩溃（已尝试，本机硬件限制仍会重启）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# 加载训练完成的模型
model = tf.keras.models.load_model("rps.keras")

# 初始化转换器并执行转换
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存轻量化模型
with open("rps_lite.tflite", "wb") as f:
    f.write(tflite_model)

print("模型转换完成")
```

    2026-06-12 14:14:30.384789: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2
    2026-06-12 14:14:30.384964: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 8.00 GB
    2026-06-12 14:14:30.384970: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 2.67 GB
    2026-06-12 14:14:30.385346: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2026-06-12 14:14:30.385362: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)


    INFO:tensorflow:Assets written to: /var/folders/np/tw641bs56mx95tw421k_fhyc0000gn/T/tmpftt2obgl/assets


    INFO:tensorflow:Assets written to: /var/folders/np/tw641bs56mx95tw421k_fhyc0000gn/T/tmpftt2obgl/assets



```python

```
