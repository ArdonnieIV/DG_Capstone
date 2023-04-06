from IPython.display import clear_output
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import copy

#######################################################################
################################ Model ################################
#######################################################################

class PoseFFNN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):

        super().__init__()

        hidden_dims = [74, 79]

        self.linear1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(hidden_dims[1], output_dim)


    ###################################################################
    
    def forward(self, featureVector):

        output = None
        
        output = self.linear1(featureVector)
        output = self.relu1(output)
        output = self.linear2(output)
        output = self.tanh(output)
        output = self.linear3(output)

        return output
    
    ###################################################################

def get_accuracy(model, data, device):

    totalVals = len(data)
    perPoseCorrect = defaultdict(int)
    perPoseTotal = defaultdict(int)

    for i in range(totalVals):

        correct = data[i]['label'].argmax(dim=0, keepdim=True).to(device).item()
        output = model(data[i]['input'].to(device))
        prediciton = output.argmax(dim=0, keepdim=True).item()

        perPoseTotal[correct] += 1
        perPoseCorrect[correct] += (prediciton == correct)

    accSum = 0
    totalPoses = len(perPoseCorrect)
    for i in range(totalPoses):
        accSum += (perPoseCorrect[i] / perPoseTotal[i])
    accSum /= totalPoses

    return accSum

    ###################################################################

def train(model, train_loader, val_data, optimizer, criterion, epochs, batch_size):

    device = None
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    # Set the device for the model
    model.to(device)

    totalVals = len(train_loader) * batch_size
    totalCorrect = 0
    train_accuracy = []
    val_accuracy = []

    best_model, best_val_acc = None, 0

    # Loop through the epochs
    for epoch in tqdm(range(epochs)):

        # Set the model to train mode
        model.train()
        
        thisTotalCorrect = 0

        # Loop through the training data
        for batch_data in train_loader:

            # Get the inputs and labels for this batch
            inputs = batch_data['input'].to(device)
            labels = batch_data['label'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Calculate the accuracy
            pred_values = outputs.argmax(dim=1, keepdim=True)
            true_values = labels.argmax(dim=1, keepdim=True)
            thisTotalCorrect += sum([pred_values[i] == true_values[i] for i in range(len(true_values))])

        clear_output(wait=True)

        with torch.no_grad():

            model.eval()

            # Track total and get batch accuracy
            totalCorrect += thisTotalCorrect
            this_train_acc = thisTotalCorrect / totalVals
            this_val_acc = get_accuracy(model, val_data, device)

            train_accuracy.append(this_train_acc)
            val_accuracy.append(this_val_acc) 

            # plot the accuracies

            # set the seaborn style to 'ticks'
            plt.clf()
            plt.plot(range(epoch+1), [acc.cpu().numpy() for acc in train_accuracy], color='b')
            plt.plot(range(epoch+1), val_accuracy, color='g')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy')
            plt.style.use('dark_background')
            plt.show()
            print(f'Epoch {epoch + 1}/{epochs} | Training accuracy: {this_train_acc.item():.4f} | Validation accuracy: {this_val_acc:.4f}')

            if this_val_acc > best_val_acc:
                best_val_acc = this_val_acc
                best_model = copy.deepcopy(model)

    return best_model, best_val_acc

    ###################################################################