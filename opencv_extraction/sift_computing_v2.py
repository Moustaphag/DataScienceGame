##############################################################################################################
### Script that computes all the sift descriptors values of a picture concatenate everything in a dataframe ###
##############################################################################################################

import cv2
import pandas as pd
import os

#path of the folder that contains all the train picture
train_folder = '/Users/pierreeliashaidara/Google Drive/Projets/Data Science Game/Qualifications/Python_scripts/test_imgs'

dense = cv2.FeatureDetector_create("Dense")
sift = cv2.SIFT()
cpt = 0
#Looping through the train directory
for root, directories, files in os.walk(train_folder):
    for filename in files:
        cpt = cpt + 1
        if cpt % 100 == 0:
            print cpt
        #Reading the image
        img = cv2.imread(os.path.join(root,filename))
        #Denoizing it
        #img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        #Resizing it
        img_resized = cv2.resize(img, (100,100))
        #Transforming to gray scale
        img_resized_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        #Detecting the sift keypoints on the gray scale picture
        kp = dense.detect(img_resized_gray)
        #Computing the sift descriptors values
        kp, des = sift.compute(img_resized_gray, kp)
        if len(kp) == 0:
            print('PAS DE SIFT POUR IMG: ' + filename)
        #Concatenating all the sift descriptors in a dataframe train_sifts
        if cpt == 1:
            train_sifts = pd.DataFrame(des)
            train_sifts['Id'] = filename[:-4] # Id of the picture
        else:
            sifts = pd.DataFrame(des)
            sifts['Id'] = filename[:-4]
            train_sifts = pd.concat([train_sifts, sifts])

#Writing the dataframe to a csv file
train_sifts.to_csv("test_sifts_dense.csv")

