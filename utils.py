import time
import copy
import numpy as np
import torch
from sklearn.metrics import f1_score

def train_model(model, model_name, criterion, optimizer, scheduler, dataloaders_dict, dataset_sizes, device, num_epochs=25):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100

    val_loss = []
    train_loss = []

    raw_preds = np.array([])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            preds = np.array([])
            labels = np.array([])     
            
            # Iterate over data.
            for inputs, sentiment in dataloaders_dict[phase]:

                inputs = inputs.to(device) 
                sentiment = sentiment.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)             

                    loss = criterion(outputs, sentiment)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                preds = np.append(preds, torch.argmax(outputs, axis=1).cpu())
                labels = np.append(labels, sentiment.cpu())

                if phase == 'val' and epoch == num_epochs-1:
                    raw_preds = collect(raw_preds, outputs)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            f1 = f1_score(labels, preds, average='micro')

            if phase == 'train':
                train_loss.append(epoch_loss)
            elif phase == 'val':
                val_loss.append(epoch_loss)

            print('{} total loss: {:.4f} '.format(phase,epoch_loss))
            print('{} F-1 score : {:.4f} '.format(phase,f1))

            if phase == 'val' and epoch_loss < best_loss:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), '/content/drive/My Drive/test_task/'+model_name+'.pth')

        print()
  
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(float(best_loss)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, raw_preds

def collect(raw_preds, outputs):
  if len(raw_preds)==0:
    raw_preds = outputs.cpu().detach().numpy()
  else:
    raw_preds = np.vstack((raw_preds, outputs.cpu().detach().numpy()))
  return raw_preds