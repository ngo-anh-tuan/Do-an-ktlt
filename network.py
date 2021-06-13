import numpy
import scipy.special
import matplotlib.pyplot
import imageio
import glob

class neuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
    
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.whh = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.who = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))

        self.lr = learningrate
        
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        
        inputs = numpy.array(inputs_list, ndmin=2).T 
        targets = numpy.array(targets_list, ndmin=2).T 
        
        
        hidden1_inputs = numpy.dot(self.wih, inputs)
        
        hidden1_outputs = self.activation_function(hidden1_inputs)
        hidden2_inputs = numpy.dot(self.whh, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)
        
        
        final_inputs = numpy.dot(self.who, hidden2_outputs)
       
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden2_errors = numpy.dot(self.who.T, output_errors)
        hidden1_errors = numpy.dot(self.whh.T, hidden2_errors)
        
        
        self.wih += self.lr * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)), numpy.transpose(inputs))
        self.whh += self.lr * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)),numpy.transpose(hidden1_outputs))
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden2_outputs))

        pass
        
    def save(self):
        numpy.save('saved_whh_2.npy', self.whh)
        numpy.save('saved_wih_2.npy', self.wih)
        numpy.save('saved_who_2.npy', self.who)
        pass
    def load(self):
        self.wih = numpy.loadtxt('weights_wih.txt')
        self.who = numpy.loadtxt('weights_who.txt')
        self.whh = numpy.loadtxt('weights_whh.txt')
        pass
    
    def query(self, inputs_list):
        
        inputs = numpy.array(inputs_list, ndmin=2).T 
        hidden1_inputs = numpy.dot(self.wih, inputs)       
        hidden1_outputs = self.activation_function(hidden1_inputs)
        hidden2_inputs = numpy.dot(self.whh, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)
        
        final_inputs = numpy.dot(self.who, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def return_weights(self):
        wih = self.wih
        whh = self.whh
        who = self.who
        return wih, whh, who

