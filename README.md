# PeopleDetector
Implementation of a machine learning technique for people
detection.

The algorithm is based on multiscale window processing, sliding on
each scale a window whose size is 64x128 pixels, enough to fit the
human body aspect ratio.

The histogram of oriented gradients is computed on each sliding window 
using the novel technique presented in [1]. This processing leads to
a descriptor of 3780 components, which is projected on a 3780-dimensional
space in order to use Support Vector Machines to classify it.

The linear SVM boundary region has been trained with over 100k samples,
33% of them containing positive data i.e. images of actual people cropped
on a 64x128 window. For this purpose a number of datasets have been compiled
in order to enhance as much as posible the accuracy of the detection system
(currently over 99.85%).
