import numpy as np;
import cv2;
import sklearn as sk;

class Detector:
  #Fill the following parameters according
  #to previous research
  
  _SVM_THRESHOLD = ;
  _BLOCK_SIZE = ;
  _ANGLE_BINS = ;
  
  
  @staticmethod
  def train():
    #train the SVM to get the (N-1) dimension
    #boundary function which separates the
    #decision regions of the binary classes
    #which decide if there's a person or not
    #within the analised ROI
    
  @staticmethod
  def detect():
    #Project the feature vector into the SVM
    #space in order to decide wether the
    #current ROI encapsulates a person or nor
    
    
   
   
if (__name__=="__main__"):
  Detector.detect();
















