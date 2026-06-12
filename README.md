# TensorFlow 石头剪刀布手势识别实验
## 实验简介
本实验基于 TensorFlow/Keras 搭建卷积神经网络(CNN)，完成石头、剪刀、布手势图像分类。
实验内容包含：数据集处理、图像可视化、模型训练、单图预测、TensorFlow Lite 模型转换，严格按照参考教程完成。

## 一、实验环境
- 设备：Apple MacBook Pro (M2 芯片)
- 系统：macOS
- Python 版本：3.10.20
- TensorFlow 版本：2.16.2
- 运行工具：Jupyter Notebook

## 二、实验整体流程
1. 数据集下载、解压、清洗与统计
2. 数据集样本图像可视化
3. 搭建卷积神经网络模型并训练
4. 绘制训练准确率/损失曲线
5. 单张手势图片预测验证模型
6. TensorFlow Lite 轻量化模型转换（硬件兼容说明）

## 三、核心代码片段
### 3.1 数据集加载与图像增强
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
3.2 CNN 网络搭建
python


运行



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
3.3 模型预测代码
python


运行



import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("rps.keras")
class_names = ["布", "石头", "剪刀"]
img_path = "./rps-test-set/rock/testrock01-00.png"

img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

predictions = model.predict(img_array)
pred_idx = np.argmax(predictions[0])
print(f"预测结果：{class_names[pred_idx]}，置信度：{np.max(predictions[0]) * 100:.2f} %")
3.4 TensorFlow Lite 转换代码
python


运行



import tensorflow as tf
model = tf.keras.models.load_model("rps.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("rps_lite.tflite", "wb") as f:
    f.write(tflite_model)
四、实验结果
数据集统计
训练集：石头 / 布 / 剪刀 各 840 张，总计 2520 张；
测试集：总计 372 张，分类正常。
模型训练
训练 25 轮，损失持续下降，准确率稳步提升。
单图预测
随机测试手势图片，预测结果：石头，置信度 100.00%，模型识别效果优秀。
可视化图表
样本图片：成功展示石头、剪刀、布手势原图；
训练曲线：生成「训练 / 验证准确率曲线」「训练 / 验证损失曲线」。
五、问题说明
本机为 Apple M2 8GB 内存，搭配 TensorFlow Metal GPU，执行 Lite 模型转换时出现内存溢出，导致 Jupyter 内核自动重启。
✅ 代码逻辑完全正确，知识点已掌握；
✅ 原始模型 rps.keras / rps.h5 训练正常、可正常加载使用。
六、实验总结
掌握 TensorFlow/Keras 图像分类模型完整开发流程；
理解卷积层、池化层、全连接层的作用；
学会图像数据增强、数据集划分、模型预测、轻量化转换；
了解 M2 芯片运行深度学习框架的内存兼容问题。
七、文件说明
12_实验5_2_TensorFlow 石头剪刀布模型生成.ipynb：完整实验笔记（含代码、运行日志、图表）
rps.keras / rps.h5：训练完成的原始模型
