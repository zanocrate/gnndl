from torch_geometric.nn.conv import EdgeConv
from torch import nn
# from torch_geometric.transforms import FaceToEdge
from transform import Decimation_FaceToEdge
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import aggr
import os

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='with_decimation')

TARGET_REDUCTION = 0.7
transform = Decimation_FaceToEdge(remove_faces=True,target_reduction=TARGET_REDUCTION)

train_dataset = ModelNet(root=os.getcwd(),name='10',train=True,transform=transform)
train_loader = DataLoader(dataset=train_dataset,batch_size=8,shuffle=True) 

test_dataset = ModelNet(root=os.getcwd(),name='10',train=False,transform=transform)
test_loader = DataLoader(dataset=test_dataset,batch_size=8,shuffle=True) 



class MyGNN(nn.Module):

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



model = MyGNN(10)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

pbar = tqdm.tqdm(range(1,101))
for epoch in pbar:

    # training mode
    model.train()

    # initialize epoch counters and metrics
    total_training_loss = 0
    total_train_accuracy = 0
    total_val_loss = 0
    total_train_accuracy = 0
    total_val_accuracy = 0
    train_batch = 0
    test_batch = 0
    
    ######################################## TRAINING LOOP
    for batch in train_loader:
        

        optimizer.zero_grad()
        logits = model(batch.pos,batch.edge_index,batch.batch)
        loss = criterion(logits,batch.y)
        loss.backward()
        optimizer.step()
        total_training_loss += loss.item()*batch.num_graphs
        total_train_accuracy += sum(logits.argmax(axis=1) == batch.y)

        train_batch+=1

    # divide by the total number of graphs
    total_train_accuracy = total_train_accuracy.item() / train_dataset.len()
    total_training_loss  = total_training_loss / train_batch


    ######################################## TESTING LOOP

    # evaluation mode
    model.eval()

    for batch in test_loader:
        
        logits = model(batch.pos,batch.edge_index,batch.batch)
        val_loss = criterion(logits,batch.y)

        total_val_loss += val_loss.item()*batch.num_graphs
        total_val_accuracy += sum(logits.argmax(axis=1) == batch.y)

        test_batch+=1

    # divide by the total number of graphs
    total_val_accuracy = total_val_accuracy.item() / test_dataset.len()
    total_val_loss = total_val_loss / test_batch



    ######################################## REPORTING

    writer.add_scalar('Loss/train',total_training_loss,epoch)
    writer.add_scalar('Accuracy/train',total_train_accuracy,epoch)

    writer.add_scalar('Loss/test',total_val_loss,epoch)
    writer.add_scalar('Accuracy/test',total_val_accuracy,epoch)

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': total_training_loss,
                }, os.path.join(writer.log_dir,'model.pt'))

    # pbar.set_description(f"Epoch loss {total_loss}")
