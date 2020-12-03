# mask_detection
Mask detection algorithm using TensorFlow and OpenCV libraries. To start place your images into 'new_images' folder and start mask_detector.ipynb notebook.


Решение задачи состоит из двух основных частей:

1. Обучение классификатора лиц в масках и без масок на базе искуственного датасета с использованием библиотеки TensorFlow Keras и 'fine tuning' (mask_detection_model.ipynb):
   1.1 Препроцессинг данных выполняется при помощи библиотеки scikit-learn (приведение к одному размеру, масштабирование, создание меток классов).
   1.2 Модель строится на базе предобученной нейросети MobileNetV2 (без полносвязных слоев, с весами 'imagenet') с полносвязной надстройкой. 
       При обучении полученной модели слои предобученной MobileNetV2 'замораживаются', обновления их весов не происходит, обучается только полносвязная надстройка, 
       что позволяет значительно сократить время обучения модели.
   1.3 При обучении также используется 'data augmentation' - поворот, увеличение, сдвиг и т.д. изображения для улучшения обобщающей способности модели.
   1.4 После обучения модель оценивается на тестовых данных и сохраняется в файл для последующего использования.
   
2. Распознавание лиц на новых фотографиях при помощи OpenCV и их классификация предварительно обученным классификатором (mask_detector.ipynb):
   2.1 Новые изображения помещаются в папку 'new_images'.
   2.2 При помощи HAAR CascadeClassifier (frontal_alt2 и profile) из библиотеки OpenCV на изображениях выделяются лица.
   2.3 Выполняется препроцессинг данных.   
   2.4 Данные анализируются ранее обученной моделью.
   2.5 Алгоритм выводит изображения с выделенными лицами и маркерами 'Mask' или 'No mask'.
   
Для классификации новых изображений нужно поместить их в папку 'new_images'.

Источники:
1. Dataset: https://www.kaggle.com/chandrasekarank/prajna-bhandary-face-mask-detection-dataset
2. Tutorial 1: https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/ 
3. Tutorial 2: https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
