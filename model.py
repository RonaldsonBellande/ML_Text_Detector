import torch.nn as nn
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


"""
1. collect and prepare data
2. make data make sense
3. use data to answer questions
4. create prediction application
"""

class Model(nn.Module):

    def __init__(self, optimization):
        super(Model, self).__init__()
        self.optimization = optimization
        self.stages = {'Trans': optimization.Transformation, 'Feat': optimization.FeatureExtraction,
                       'Seq': optimization.SequenceModeling, 'Pred': optimization.Prediction}

        """ Thin Plate Spline Transformation (TPS) """
        if optimization.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=optimization.num_fiducial, I_size=(optimization.imgH, optimization.imgW), I_r_size=(optimization.imgH, optimization.imgW), I_channel_num=optimization.input_channel)
        else:
            print('No Transformation module specified')



        """ FeatureExtraction """
        if optimization.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(optimization.input_channel, optimization.output_channel)
        elif optimization.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(optimization.input_channel, optimization.output_channel)
        elif optimization.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(optimization.input_channel, optimization.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        
        self.FeatureExtraction_output = optimization.output_channel  
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))



        """ Sequence modeling"""
        if optimization.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, optimization.hidden_size, optimization.hidden_size),
                BidirectionalLSTM(optimization.hidden_size, optimization.hidden_size, optimization.hidden_size))
            self.SequenceModeling_output = optimization.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output



        """ Prediction """
        if optimization.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, optimization.num_class)
        elif optimization.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, optimization.hidden_size, optimization.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')




    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.optimization.batch_max_length)

        return prediction

