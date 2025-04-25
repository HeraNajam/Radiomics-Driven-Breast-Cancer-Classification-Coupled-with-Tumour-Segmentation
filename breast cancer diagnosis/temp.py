# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from get_data import feature_extraction_complete_function
from ml_models import testing_with_ml_best_model
from dl_model import testing_with_nn_best_model

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from PIL import Image
#import time

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, getTestCase

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer


def fillmask_func( canvas_image, mask_file_name ):
    image = cv2.imread( 'images/' + canvas_image)  # Load the image

    if image is None:
        raise ValueError("Image not loaded properly. Check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # Apply threshold to get a binary image

    # # Save and display the binary image for debugging
    # binary_filename = "binary_image.png"
    # cv2.imwrite(binary_filename, binary)
    # # st.image(binary_filename, caption='Binary Image')

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:  # Check if any contours are found
        raise ValueError("No contours found in the image")

    # Select the largest contour (you can choose based on your requirement)
    contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(contour)  # Find a point inside the contour
    if M["m00"] != 0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
    else:
        raise ValueError("Contour area is zero, can't find the center")

    height, width = image.shape[:2]  # Create a mask for flood fill
    mask = np.zeros((height + 2, width + 2), np.uint8)

    color = (255, 255, 255)  # Set the fill color to white
    cv2.floodFill(image, mask, (x, y), color)  # Flood fill the shape with the white color

    # Save the resultant image
    cv2.imwrite( 'images/' + mask_file_name, image)  # saving the final mask
    return mask_file_name

####################################################################################
############ Upload Image draw canvas and save the resultant mask ############
st.title("Upload a test Image")
uploaded_file = st.file_uploader("Choose an image", type = ["png", "jpg", "jpeg"] )  # Upload image

# Ensure session state is initialized
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Open image
    # st.image(image, caption='Uploaded Image.', use_column_width=True) ## display image
    img_array = np.array(image)  # Convert the image to numpy array

    canvas_result = st_canvas(  # Create a canvas component
        fill_color="white",
        stroke_width=2,
        stroke_color="white",
        background_image=Image.fromarray(img_array),
        update_streamlit=True,
        height=img_array.shape[0],
        width=img_array.shape[1],
        drawing_mode="freedraw",
        key="canvas",
    )
    original_image_name = uploaded_file.name
    image_object = Image.fromarray(img_array)
    width_ = img_array.shape[1]
    # st.write(width_)
    # st.write(original_image_name)
    mask_file_name = original_image_name.split('.')[0] + '_mask.png'

    image_object.save( 'images/' + original_image_name )  # save original image for getting radiomic features

    # Processing with the drawn canvas
    if canvas_result.image_data is not None:
        output_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        output_image.save( 'images/' + "canvas_image.png")  # Save the original image to a file

    # Add a sleep of 2 seconds before saving the image
    # time.sleep(1)

    ####### Fill the canvas shape --- function call #######
    masked_image = fillmask_func( "canvas_image.png", mask_file_name )

    ####### Display the final masked image #######
    masked_image_name   = 'images/' + masked_image  
    original_image_name = 'images/' + original_image_name
    final_image = Image.open( masked_image_name )  # Open image
    st.image(final_image, caption='Mask', use_column_width=width_)  # display final mask
    
    # st.write(original_image_name)
    # st.write(masked_image_name)

    ####################################################################################
    ########################### Extract Radiomic Features ##############################
    ####################################################################################

    if st.button( 'Extract Radiomic Features', type="primary" ):
        st.session_state.df = feature_extraction_complete_function( original_image_name, masked_image_name  )
        st.write('extracting radiomics features ...') 
        st.write( st.session_state.df )
        
        # # Display the chart in Streamlit
        # st.write("Bar chart using Altair:")
        
        
####################################################################################
########################### Testing the sample data   ##############################
####################################################################################   

if st.session_state.df is not None:
    
    if st.button( 'Test the sample with Best Models', type="primary" ):
        
        col1, col2, col3 = st.columns(3) # Columns for different models
        with col1:
            st.header("SVM")
            svm_model = 'models/svm_2df_without_outliers_50_bal_linear_model.pkl'
            print(st.session_state.df.shape)
            pred_svm = testing_with_ml_best_model( st.session_state.df, svm_model  )
            if pred_svm == 1:
                st.write('Class 1 - MALIGNANT')
            elif pred_svm == 0:
                st.write('CLASS 0 - BENIGN')
                
            
        with col2:
            st.header("RF")
            rf_model = 'models/rf_2df_without_outliers_50_bal_model.pkl'
            print(st.session_state.df.shape)
            pred_rf = testing_with_ml_best_model( st.session_state.df, rf_model  )
            if pred_rf == 1:
                st.write('Class 1 - MALIGNANT')
            elif pred_rf == 0:
                st.write('CLASS 0 - BENIGN')
            
        with col3:
            st.header("FF-NN")
            ff_nn_model = 'models/nn_best_model.pth'
            pred_nn = testing_with_nn_best_model( st.session_state.df, ff_nn_model  )
            if pred_nn == 1:
                st.write('Class 1 - MALIGNANT')
            elif pred_nn == 0:
                st.write('CLASS 0 - BENIGN')
            
        
