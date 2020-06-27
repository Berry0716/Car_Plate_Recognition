使用模块：
1. cv2 4.1.2
2. numpy as np
3. sys
4. os
5. matplotlib
6. tensorflow 2.0.0
7. keras
8. pandas

使用场景：
车牌识别（可用于带有背景的图片）

目录介绍：
1. plate_recognition_utils: 存放自定义的车牌获取模块（附有详细代码介绍）
2. plates_image: 存放获取的车牌图片
3. plates_split_utils: 存放车牌字符分割自定义模块（附有详细代码介绍）

4. 01_find_plate.ipynb  车牌获取实现
5. 02_plate_split.ipynb 车牌字符分割

6. 03_tensorflow_training  用于训练keras模型来识别车牌分割字符
