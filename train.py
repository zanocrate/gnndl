# from torch_geometric.transforms import FaceToEdge
from transform import Decimation_FaceToEdge
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import os

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

# open configuration file
import json
with open("config.json") as f: config = json.load(f)

writer = SummaryWriter(log_dir=config['log_dir'],comment='extra_edgeconv_more_final_features_MODELNET10')
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################# DATASET


transform = Decimation_FaceToEdge(remove_faces=True,target_reduction=config['target_reduction'])

assert config['dataset'] == 10 or config['dataset'] == 40

train_dataset = ModelNet(root=os.getcwd(),name=str(config['dataset']),train=True,transform=transform)
train_loader = DataLoader(dataset=train_dataset,batch_size=config['batch_size'],shuffle=True) 

test_dataset = ModelNet(root=os.getcwd(),name=str(config['dataset']),train=False,transform=transform)
test_loader = DataLoader(dataset=test_dataset,batch_size=config['batch_size'],shuffle=True) 


####################### MODEL

from model import MyGNN
model = MyGNN(config['dataset'])
model.to(device)

####################### OPTIMIZER AND LOSS

# GAMMA = 0.5

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=GAMMA)


######################### TRAINING LOOP

val_best_accuracy = 0
val_best_loss = 1e10

pbar = tqdm.tqdm(range(1,151))
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

    if total_val_accuracy > val_best_accuracy: val_best_accuracy = total_val_accuracy
    if total_val_loss < val_best_loss: val_best_loss = total_val_loss


    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': total_training_loss,
                }, os.path.join(writer.log_dir,'model.pt'))

    # pbar.set_description(f"Epoch loss {total_loss}")

writer.add_hparams(
    hparam_dict= {
        'lr' : LR,
        # 'gamma' : GAMMA,
        'batch_size' : BATCH_SIZE,
        'target_reduction' : TARGET_REDUCTION
    },

    metric_dict = {
        'hparams/val_accuracy' : val_best_accuracy,
        'hparams/val_loss' : val_best_loss

    }
)