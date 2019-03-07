import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import time
from torch.autograd import Variable
import random

# Parameters
time_s = 100  # The time of data
region_n = 94  # The brain region number
N_LAYERS = 2  # RNN layer number
Layer1_HIDDEN_SIZE = 200  # RNN hidden size2 = The number you want
Layer2_HIDDEN_SIZE = 7  # RNN hidden size1 = The number you want
N_EPOCHS = 3000  # Epoch number
N_AGE = 7  # Age Range
input_size = region_n  # RNN input size = brain region number
other_size = time_s  # Input matrix other size = time of data
N_train = 596  # The number of train
N_valid = 199  # The number of validation
r_step_list = [0.0001, 0.00001, 0.000001, 0.0000001]  # Rate list
state_name = 'model_valid_min201903071854.pt'  # ML state saved name

# Read the brain data for all subjects & Change words in brain_read and function brain_data_load
brain_read = pd.read_csv('C:/Users/USER/Desktop/rest_csv_data/brain_entropy 2.6.csv')

# Put data in GPU or CPU
def create_variable(tensor):
    # Determine the GPU exists
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    # If no GPU, put it in CPU
    else:
        #print('NO GPU, USE CPU')
        return Variable(tensor)

# Read the project name and subject based on rows
def project_subject_name(row_number):
    the_brain_data = brain_read[row_number: row_number + 1]
    the_project_name = np.array(the_brain_data.project)[0]
    the_subject_name = np.array(the_brain_data.subject)[0]
    return the_project_name, the_subject_name

# Read the data for a subject
def brain_data_load(project_name, subject_name):
    all_data_load = np.genfromtxt('C:/Users/USER/Desktop/rest_csv_data/{}/{}/rest_image.csv'
                                  .format(project_name, subject_name), delimiter=',')
    # The first row data is words, so start with the second row
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
    data_reshape = np.reshape(data_load, (other_size, 1, input_size))
    data_in_torch = create_variable(torch.from_numpy(data_reshape))
    if torch.cuda.is_available():
        return create_variable(torch.cuda.DoubleTensor(data_in_torch))
    else:
        return create_variable(torch.DoubleTensor(data_in_torch))

# Randomly drawing training set & validation set & test set, and put number in list
def get_train_valid_test_set(number_train, number_valid):

    num_all_subject = list(range(brain_read.shape[0]))

    num_train_subject = random.sample(range(brain_read.shape[0]), k=number_train)

    out_of_train = list(set(num_all_subject).difference(set(num_train_subject)))
    num_valid_subject = random.sample(range(brain_read.shape[0] - number_train), k=number_valid)

    num_test_subject = list(set(out_of_train).difference(set(num_valid_subject)))

    return num_train_subject, num_valid_subject, num_test_subject

# Our model
class RNNClassifier(nn.Module):

    def __init__(self, ml_input_size, layer1_hidden_size, layer2_hidden_size, output_size, n_layers):
        super(RNNClassifier, self).__init__()
        self.hidden_size1 = layer1_hidden_size
        self.hidden_size2 = layer2_hidden_size
        self.n_layers = n_layers
        self.gru1 = nn.GRU(ml_input_size, layer1_hidden_size, n_layers)
        self.gru2 = nn.GRU(layer1_hidden_size, layer2_hidden_size, n_layers)
        self.fc = nn.Linear(layer2_hidden_size, output_size)

    def forward(self, ml_input):
        # Make a hidden
        batch_size = 1
        hidden1 = self._init_hidden1(batch_size)
        hidden2 = self._init_hidden2(batch_size)
        if torch.cuda.is_available():
            ml_input = ml_input.type(torch.cuda.FloatTensor)
        else:
            ml_input = ml_input.type(torch.FloatTensor)
        output_var1, hidden1 = self.gru1(ml_input, hidden1)
        output_var, hidden2 = self.gru2(output_var1, hidden2)
        # Use the last layer output as FC's input
        fc_output = self.fc(hidden2[-1])

        return fc_output

    def _init_hidden1(self, batch_size):
        hidden1 = torch.zeros(self.n_layers, batch_size, self.hidden_size1)
        return create_variable(hidden1)

    def _init_hidden2(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size2)
        return create_variable(hidden)

def train(number_train_set_list, rate):
    print('\nEpochs:{}, Train start'.format(epoch))
    classifier.train()

    total_loss_train = 0
    train_num = 0
    train_correct = 0

    for i in number_train_set_list:
        train_project_name, train_subject_name = project_subject_name(i)
        input_data = data_input(train_project_name, train_subject_name)

        output = classifier(input_data)

        target = age_data_load(train_project_name, train_subject_name)

        train_pre = output.data.max(1, keepdim=True)[1]
        train_correct += train_pre.eq(target.data.view_as(train_pre)).cpu().sum()

        loss = criterion(output, target)
        total_loss_train = total_loss_train + loss

        classifier.zero_grad()
        loss.backward()
        optimizer.step()

        train_num = train_num + 1

    print('Train Epoch: {}, Total loss: {}, R_step: {}'.format(epoch, total_loss_train, rate))
    print('Train set: Accuracy: {}/{} ({:.0f}%)'.format(train_correct, train_num, 100. * train_correct / train_num))

    return total_loss_train

def validate(number_valid_set_list):
    print("Validating trained model ...")
    classifier.eval()

    valid_correct = 0
    total_loss_valid = 0
    valid_num = 0

    for j in number_valid_set_list:
        valid_project_name, valid_subject_name = project_subject_name(j)
        valid_input_data = data_input(valid_project_name, valid_subject_name)

        output_valid = classifier(valid_input_data)

        valid_target = age_data_load(valid_project_name, valid_subject_name)

        valid_pre = output_valid.data.max(1, keepdim=True)[1]
        valid_correct += valid_pre.eq(valid_target.data.view_as(valid_pre)).cpu().sum()

        loss_valid = criterion(output_valid, valid_target)
        total_loss_valid = total_loss_valid + loss_valid

        valid_num = valid_num + 1

    print('Train Epoch: {}, Validation loss: {}'.format(epoch, total_loss_valid))
    print('Validation set: Accuracy: {}/{} ({:.0f}%)'.format(valid_correct, valid_num,
                                                             100. * valid_correct / valid_num))

    return total_loss_valid

def test(number_test_set_list, load_model_name):

    print("\nEvaluating trained model ...")
    classifier.load_state_dict(torch.load(load_model_name))
    classifier.eval()
    test_num = 0
    test_correct = 0

    for k in number_test_set_list:

        test_project_name, test_subject_name = project_subject_name(k)
        test_input_data = data_input(test_project_name, test_subject_name)

        output_test = classifier(test_input_data)

        test_target = age_data_load(test_project_name, test_subject_name)

        test_pre = output_test.data.max(1, keepdim=True)[1]
        test_correct += test_pre.eq(test_target.data.view_as(test_pre)).cpu().sum()

        test_num = test_num + 1

    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(test_correct, test_num,
                                                             100. * test_correct / test_num))

    return test_correct

if __name__ == '__main__':

    classifier = RNNClassifier(input_size, Layer1_HIDDEN_SIZE, Layer2_HIDDEN_SIZE, N_AGE, N_LAYERS)

    # Put model in GPU
    if torch.cuda.is_available():
        classifier.cuda()

    criterion = nn.CrossEntropyLoss()

    r_step = r_step_list[0]
    total_loss_valid_min = np.Inf

    list_train, list_valid, list_test = get_train_valid_test_set(N_train, N_valid)

    start = time.time()

    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(1, N_EPOCHS + 1):
        optimizer = torch.optim.Adam(classifier.parameters(), lr=r_step)

    # Train cycle
        sum_loss_train = train(list_train, r_step)

        # Change rate
        if sum_loss_train < 500:
            r_step = r_step_list[1]
            if sum_loss_train < 250:
                r_step = r_step_list[2]
                if sum_loss_train < 200:
                    r_step = r_step_list[3]

    # Validation cycle
        sum_loss_valid = validate(list_valid)

        if sum_loss_valid <= total_loss_valid_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(total_loss_valid_min, sum_loss_valid))
            torch.save(classifier.state_dict(), state_name)
            total_loss_valid_min = sum_loss_valid

    # Testing
    name = state_name
    test(list_test, name)
    print('END')
