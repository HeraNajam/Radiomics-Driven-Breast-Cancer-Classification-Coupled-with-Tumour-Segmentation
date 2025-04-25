# import pandas as pd
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from IPython.core.debugger import Tracer

class Dataset(torch.utils.data.Dataset): # class for data loading

    def __init__(self, df ):
      self.df = df
      # self.input_cols  = list( set(self.df.columns) - set(['y'])) # save cols
      # self.output_cols = ['y']
    
    def __len__(self):
      return len(self.df) # TODO: here i will return the number of samples in the dataset
    
    def __getitem__(self, idx):
      cur_sample = self.df.iloc[ idx ] # read row, split in input and output and convert in tensors
    
      # cur_sample_x = cur_sample[ self.input_cols]  # split in input / ground-truth
      # cur_sample_y = cur_sample[ self.output_cols] # split in input / ground-truth
    
      cur_sample_x = torch.tensor( cur_sample.tolist() )                     # convert the quantities into the tensor (torch format)
      # cur_sample_y = torch.tensor( cur_sample_y.tolist(), dtype = torch.long ) # convert the quantities into the tensor (torch format)
    
      return cur_sample

def testing_with_nn_best_model( df, model ):
    
    
    class Net(nn.Module):
      def __init__(self):
        super(Net,self).__init__()
        #define layer
        self.layer1 = nn.Linear( 73, 256 )
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear( 256, 128 )
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear( 128, 64 )
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear( 64, 32 )
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Linear( 32, 16 )
        self.layer10 = nn.ReLU()
        self.layer11 = nn.Linear( 16, 1)
    
      def forward(self,x):
        # print(f'shape in start -- {x.shape}')
        x=self.layer1(x)
        # print(f'shape after 1st layer -- {x.shape}')
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.layer6(x)
        x=self.layer7(x)
        x=self.layer8(x)
        x=self.layer9(x)
        x=self.layer10(x)
        x=self.layer11(x)
        # print(f'shape in end -- {x.shape}')
        return x
    
    net = Net()
    # TODO: load best network
    state = torch.load( model )
    net.load_state_dict( state['net'])
      
    # create testing function
    def testing_func_one_sample( net, ds_temp, loss_func ): 
        
      net.eval() # set network in eval mode
      inp = ds_temp
      
      with torch.no_grad(): # get output
        out = net(inp)
      ##########################################################################
      probs = torch.sigmoid(out)  # Apply sigmoid to get probabilities
      preds = (probs > 0.5).float()   # testing to see the predicted values
      ##########################################################################
      return preds 
      
    ###########################################################################
    ############ CREATE AN OBJECT FOR DATASET CLASS
    ###########################################################################
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

    df = df[ filtered_list ]
    ds = torch.tensor( df.to_numpy(), dtype=torch.float32 )
    
    loss_func = nn.BCEWithLogitsLoss()

    ############# TESTING
    pred_nn = testing_func_one_sample( net, ds, loss_func )

    
    return pred_nn
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    