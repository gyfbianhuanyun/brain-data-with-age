import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import time
from torch.autograd import Variable

# Parameters
time_s = 100  # The time of data
region_n = 94  # The brain region number
HIDDEN_SIZE = region_n  # RNN hidden size = brain region number
N_LAYERS = 1  # RNN layer number
N_EPOCHS = 3000  # Epoch number
N_AGE = 7
input_size = region_n  # RNN input size = brain region number
other_size = time_s  # Input matrix other size = time of data
N_train = 795  # The number of train
r_step_list = [0.0001, 0.00001, 0.000001, 0.0000001]  # Rate change list

# Read the brain data for all subjects
brain_read = pd.read_csv('C:/Users/USER/Desktop/rest_csv_data/brain_entropy 2.6.csv')

# Read the project name and put it in the list
project_namelist = np.unique(brain_read.project)

# Put data in GPU
def create_variable(tensor):
    # Determine the GPU exists
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        print('NO GPU')

# Read the subject name under the same project and put it in the list
def subject_namelist(project_name):
    project = brain_read[brain_read.project == project_name]
    return project.subject

# Read the project name and subject based on rows
def project_subject_name(row_number):
    the_brain_data = brain_read[row_number - 1: row_number]
    the_project_name = np.array(the_brain_data.project)[0]
    the_subject_name = np.array(the_brain_data.subject)[0]
    return the_project_name, the_subject_name

# Read the data for a subject
def brain_data_load(project_name, subject_name):
    all_data_load = np.genfromtxt('C:/Users/USER/Desktop/rest_csv_data/{}/{}/rest_image.csv'.format(project_name, subject_name), delimiter=',')
    return all_data_load[1: time_s + 1, : region_n] / 10000

# Read the age for a subject
def age_data_load(project_name, subject_name):
    project_data = brain_read[brain_read.project == project_name]
    subject_data = project_data[project_data.subject == subject_name]
    subject_data_array = np.array(subject_data.new)
    data_in_torch = Variable(torch.from_numpy(subject_data_array)) / 10 - 1
    target_age = torch.LongTensor([data_in_torch])
    return create_variable(target_age)

# Modify the data shape
def data_input(project_name, subject_name):
    data_load = brain_data_load(project_name, subject_name)
    # datax = datax.swapaxes(0, 1)
    data_reshape = np.reshape(data_load, (other_size, 1, input_size))
    data_in_torch = create_variable(torch.from_numpy(data_reshape))
    return create_variable(torch.cuda.DoubleTensor(data_in_torch))

# Our model
class RNNClassifier(nn.Module):

    def __init__(self, ml_input_size, hidden_size, output_size, n_layers):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(ml_input_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, ml_input):
        # Make a hidden
        batch_size = 1
        hidden = self._init_hidden(batch_size)
        ml_input = ml_input.type(torch.cuda.FloatTensor)
        output_var, hidden = self.gru(ml_input, hidden)
        # Use the last layer output as FC's input
        fc_output = self.fc(hidden)

        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return create_variable(hidden)

if __name__ == '__main__':

    classifier = RNNClassifier(input_size, HIDDEN_SIZE, N_AGE, N_LAYERS)

    # Put model in GPU
    if torch.cuda.is_available():
        classifier.cuda()

    criterion = nn.CrossEntropyLoss()

    r_step = r_step_list[0]
    start = time.time()

    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(1, N_EPOCHS + 1):
        optimizer = torch.optim.Adam(classifier.parameters(), lr=r_step)

    # Train cycle
        print('Epochs:{}, Train start'.format(epoch))

        total_loss = 0
        ml_num = 0
        train_correct = 0

        for i in range(1, N_train + 1):

            train_project_name, train_subject_name = project_subject_name(i)
            input_data = data_input(train_project_name, train_subject_name)

            output = classifier(input_data)
            output.squeeze_(0)

            target = age_data_load(train_project_name, train_subject_name)

            train_pre = output.data.max(1, keepdim=True)[1]
            train_correct += train_pre.eq(target.data.view_as(train_pre)).cpu().sum()

            loss = criterion(output, target)
            total_loss = total_loss + loss

            classifier.zero_grad()
            loss.backward()
            optimizer.step()

            ml_num = ml_num + 1

            #if ml_num >= N_train:
                #break

        print('Train Epoch: {}, Loss: {}, R_step: {}'.format(epoch, total_loss, r_step))
        print('Train set: Accuracy: {}/{} ({:.0f}%)'.format(train_correct, ml_num, 100. * train_correct / ml_num))

        # Change rate
        if total_loss < 580:
            r_step = r_step_list[1]
            if total_loss < 300:
                r_step = r_step_list[2]
                if total_loss < 250:
                    r_step = r_step_list[3]

    # Testing
        print("evaluating trained model ...")

        test_correct = 0

        #for j in range(N_train + 1, brain_read.shape[0] + 1):
        for j in range(1, brain_read.shape[0] + 1):

            test_project_name, test_subject_name = project_subject_name(j)
            test_input_data = data_input(test_project_name, test_subject_name)

            output_test = classifier(test_input_data)
            output_test.squeeze_(0)

            test_target = age_data_load(test_project_name, test_subject_name)

            test_pre = output_test.data.max(1, keepdim=True)[1]
            test_correct += test_pre.eq(test_target.data.view_as(test_pre)).cpu().sum()

            ml_num = ml_num + 1

        print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(test_correct, ml_num / 2, 100. * test_correct * 2 / ml_num))