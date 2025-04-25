# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:28:43 2024

@author: FATTANI COMPUTERS
"""
import radiomics
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor, getTestCase

def feature_extraction_complete_function( image_path, mask_path ):
    print(image_path)
    print(mask_path)
    
    def process_single_image_255( image_path, mask_path ):
      image = sitk.ReadImage( image_path, sitk.sitkInt32 )
      mask = sitk.ReadImage( mask_path, sitk.sitkInt32 )
    
      if image is None or mask is None:  # Something went wrong, in this case PyRadiomics will also log an error
        raise Exception('Error getting image!')  # Raise exception to prevent cells below from running in case of "run all"
      else:
        # Instantiate the extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        result = extractor.execute( image, mask, label=255 )
        return result

    def process_single_image_65025( image_path, mask_path ):
      image = sitk.ReadImage( image_path, sitk.sitkInt32 )
      mask  = sitk.ReadImage( mask_path, sitk.sitkInt32 )
    
      if image is None or mask is None:  # Something went wrong, in this case PyRadiomics will also log an error
        raise Exception('Error getting image!')  # Raise exception to prevent cells below from running in case of "run all"
      else:
        # Instantiate the extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        result = extractor.execute( image, mask, label=65025 )
        return result
    
    def extract_original_features( result ):
      # Extract data with keys starting from 'original'
      firstorder_data = {key: value for key, value in result.items() if any(key.startswith(prefix) for prefix in features_prefix)}
      return firstorder_data
    
    def convert_dict_to_df_single_image( dict_dict ):
      columns = []
      values = [[]]
      for key, value in dict_dict.items():
        columns.append( key )
        values[0].append( float(value) )
    
      df = pd.DataFrame( values, columns=columns)
      return df
    
    #######################################################
    
    # image_path = 'images/malignant (1).png'
    # mask_path  = 'images/malignant (1)_mask.png'
    
    print(image_path)
    print(mask_path)
    
    ########################################################
    ############## EXTRACT RADIOMICS FEATURES ##############
    ########################################################
    try:
        result = process_single_image_255( image_path, mask_path )
        print(len(result))
    except:
        result = process_single_image_65025( image_path, mask_path )
        print(len(result))
        

    ## EXTRACT ORIGINAL FEATURES ONLY
    features_prefix = ['original_firstorder', 'original_glcm', 'original_gldm', 'original_glrlm', 'original_glszm', 'original_ngtdm' ]
    
    firstorder_data_dict = extract_original_features( result )
    print(len(firstorder_data_dict))
    
    
    counts = {prefix: sum(1 for key in firstorder_data_dict.keys() if key.startswith(prefix)) for prefix in features_prefix}
    print(counts)
    
    df = convert_dict_to_df_single_image( firstorder_data_dict )
    print(df.shape)
    print(df.iloc[:, :5]) ## printing only first 5 columns
    return df
