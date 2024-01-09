from torch_geometric.nn.conv import EdgeConv
from torch import nn
from torch_geometric.nn import aggr


import torch


class MyGNN(nn.Module):

    """
    One more EdgeConv, more features in last layer and more complex MLP at the end
    """

    def __init__(self,output_classes=10):

        super().__init__()
        
        self.edge_conv_1 = EdgeConv(
                # here we define h_theta
                nn=nn.Sequential(
                    nn.Linear(2*3,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )
        
        self.edge_conv_2 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        self.edge_conv_3 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,128),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        self.edge_conv_4 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*128,256),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        # self.edge_conv_5 = EdgeConv(
        #         nn=nn.Sequential(
        #             nn.Linear(2*128,256),
        #             nn.Sigmoid()
        #         ),
        #         aggr='max'
        #     )
        
        self.max_pooling = aggr.MaxAggregation()
        self.mean_pooling = aggr.MeanAggregation()

        self.output_layer = nn.Sequential(
            nn.Linear(256,128),
            nn.Sigmoid(),
            nn.Linear(128,output_classes)

        )

    def forward(self,x,edge_index,batch):

        x = self.edge_conv_1(x,edge_index)
        x = self.edge_conv_2(x,edge_index)
        x = self.edge_conv_3(x,edge_index)
        x = self.edge_conv_4(x,edge_index)
        # x = self.edge_conv_5(x,edge_index)
        x = self.max_pooling(x,batch)

        return self.output_layer(x)

class BestGNNsoFAR(nn.Module):

    """
    One more EdgeConv
    """

    def __init__(self,output_classes=10):

        super().__init__()
        
        self.edge_conv_1 = EdgeConv(
                # here we define h_theta
                nn=nn.Sequential(
                    nn.Linear(2*3,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )
        
        self.edge_conv_2 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        self.edge_conv_3 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        self.edge_conv_4 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,128),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        self.edge_conv_5 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*128,128),
                    nn.Sigmoid()
                ),
                aggr='max'
            )
        
        self.max_pooling = aggr.MaxAggregation()
        self.mean_pooling = aggr.MeanAggregation()

        self.output_layer = nn.Sequential(
            nn.Linear(128,output_classes)

        )

    def forward(self,x,edge_index,batch):

        x = self.edge_conv_1(x,edge_index)
        x = self.edge_conv_2(x,edge_index)
        x = self.edge_conv_3(x,edge_index)
        x = self.edge_conv_4(x,edge_index)
        x = self.edge_conv_5(x,edge_index)
        x = self.max_pooling(x,batch)

        return self.output_layer(x)

class ComplexMLP_GNN(nn.Module):
    
    """
    A slightly more complicated GNN using bigger MLP in the EdgeConv

    DOES NOT WORK
    """


    def __init__(self,output_classes=10):

        super().__init__()
        
        self.edge_conv_1 = EdgeConv(
                # here we define h_theta
                nn=nn.Sequential(
                    nn.Linear(2*3,64),
                    nn.Sigmoid(),
                    nn.Linear(64,32),
                    nn.Sigmoid()
                ),
                aggr='max'
            )
        
        self.edge_conv_2 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*32,128),
                    nn.Sigmoid(),
                    nn.Linear(128,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        self.edge_conv_3 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        self.edge_conv_4 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,128),
                    nn.Sigmoid()
                ),
                aggr='max'
            )
        
        self.max_pooling = aggr.MaxAggregation()
        self.mean_pooling = aggr.MeanAggregation()

        self.output_layer = nn.Sequential(
            nn.Linear(128,256),
            nn.Sigmoid(),
            nn.Linear(256,output_classes)
        )

    def forward(self,x,edge_index,batch):

        x = self.edge_conv_1(x,edge_index)
        x = self.edge_conv_2(x,edge_index)
        x = self.edge_conv_3(x,edge_index)
        x = self.edge_conv_4(x,edge_index)
        x = self.max_pooling(x,batch)

        return self.output_layer(x)


class Baseline_GNN(nn.Module):

    """
    This model got 83% in test set of modelnet10
    """

    def __init__(self,output_classes=10):

        super().__init__()
        
        self.edge_conv_1 = EdgeConv(
                # here we define h_theta
                nn=nn.Sequential(
                    nn.Linear(2*3,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )
        
        self.edge_conv_2 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        self.edge_conv_3 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,64),
                    nn.Sigmoid()
                ),
                aggr='max'
            )

        self.edge_conv_4 = EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*64,128),
                    nn.Sigmoid()
                ),
                aggr='max'
            )
        
        self.max_pooling = aggr.MaxAggregation()
        self.mean_pooling = aggr.MeanAggregation()

        self.output_layer = nn.Sequential(
            nn.Linear(128,output_classes)
        )

    def forward(self,x,edge_index,batch):

        x = self.edge_conv_1(x,edge_index)
        x = self.edge_conv_2(x,edge_index)
        x = self.edge_conv_3(x,edge_index)
        x = self.edge_conv_4(x,edge_index)
        x = self.max_pooling(x,batch)

        return self.output_layer(x)