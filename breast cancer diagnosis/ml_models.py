

import pickle

def testing_with_ml_best_model( df, model  ):
    
    # Load the model
    with open( model , 'rb') as f:
        loaded_model = pickle.load(f)
        
    out_cols = ['original_firstorder_10Percentile', 'original_firstorder_Energy', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum',
                 'original_firstorder_TotalEnergy', 'original_glcm_Imc1', 'original_gldm_DependenceNonUniformity',
                 'original_gldm_GrayLevelNonUniformity',
                 'original_gldm_LargeDependenceHighGrayLevelEmphasis',
                 'original_glrlm_GrayLevelNonUniformity',
                 'original_glrlm_LongRunEmphasis',
                 'original_glrlm_LongRunHighGrayLevelEmphasis',
                 'original_glrlm_LongRunLowGrayLevelEmphasis',
                 'original_glrlm_RunVariance',
                 'original_glszm_LargeAreaEmphasis',
                 'original_glszm_LargeAreaHighGrayLevelEmphasis',
                 'original_glszm_LargeAreaLowGrayLevelEmphasis',
                 'original_glszm_ZoneVariance',
                 'original_ngtdm_Busyness']
    
    filtered_list = [item for item in df.columns if item not in out_cols]

    df = df[filtered_list]
    print('INSIDE THE MODEL FUNCTION')
    print( df.isnull().sum( axis=1 ) )
    print('**********************', df.shape  ,'**********************' )

        
    # Predict the class of the sample
    prediction = loaded_model.predict( df )
    print('THIS IS THE PREDICTION = ', prediction)
    return prediction
