# Introduction: 
LSTM-CNN model for Human Activity Recognition
The  first wearable dataset is [Human Activity Recognition database](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), which consists of recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors. Each person performed six activities by wearing a smartphone (Samsung Galaxy S II) on the waist. From the embedded accelerometer and gyroscope, 3-axial linear acceleration and 3-axial angular velocity were captured at a constant rate of 50Hz. The labels were recorded by video. The sensor signals were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). From each window, a vector of 561 features was obtained by calculating variables from the time and frequency domain.
Another wearable dataset is the [Mobile Health (MH)](http://archive.ics.uci.edu/ml/datasets/mhealth+dataset) dataset , which comprises body motion and vital signs recordings for ten volunteers while performing 12 common activities. Sensors placed on the subject’s chest, right wrist and left ankle were used to measure the motion experienced by diverse body parts, namely, acceleration, rate of turn and magnetic field orientation. The sensor positioned on the chest also provides 2-lead ECG measurements, which can be potentially used for basic heart monitoring, checking for various arrhythmias or looking at the effects of exercise on the ECG. All sensing modalities were recorded using a video camera at a sampling rate of 50 Hz, and sampled in fixed-width sliding windows of 128 readings/window.
Both HAR and MH training data have been randomly partitioned into 80% training set, and 20% validation set.

# Requirements:
- tensorflow
- python3

How to run:
Run LSTM-CNN model for different dataset as below:
```
HAR_lstm_cnn.py --data_type HAR
HAR_lstm_cnn.py --data_type MH
```

Remember to cite the following papers if you use any of the code:
```
@inproceedings{lyu2017privacy,
  title={Privacy-Preserving Collaborative Deep Learning with Application to Human Activity Recognition},
  author={Lyu, Lingjuan and He, Xuanli and Law, Yee Wei and Palaniswami, Marimuthu},
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
  pages={1219--1228},
  year={2017},
  organization={ACM}
}
```
