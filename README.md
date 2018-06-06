# PeopleDetector
Implementation of a machine learning technique for people
detection.

The algorithm is based on multiscale window processing, sliding on
each scale a window whose size is 64x128 pixels, enough to fit the
human body aspect ratio.

The histogram of oriented gradients is computed on each sliding window 
using the novel technique presented in [1]. This processing leads to
a descriptor of 3780 components, which is projected into the feature
space for its binary classification.

Linear SVM is used in order to accomplish the later classification process
as much accurate as posible. In order to avoid training overfit, a penalty
factor of over 0.03 has been used since it gives the best results regarding
the testing dataset. This means that some missclasification is performed during
the training process in order to achieve a higher boundary margin therefore
accomplishing a better model generalization.

The linear SVM boundary region has been trained with over 100k samples,
33% of them containing positive data i.e. images of actual people cropped
on a 64x128 window. For this purpose a number of datasets have been compiled
in order to enhance as much as posible the accuracy of the detection system, 
which is currently over 99.85%.
