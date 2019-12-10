import numpy as np

def sigmoid(x):
    ''' Sigmoid activation function: f(x) = 1 / (1 + e^(-x))'''
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    ''' Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))'''
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    ''' y_true and y_pred are numpy arrays of the same length.'''
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    def __init__(self):
        # Weights
        self.wforh1 = np.random.normal(0,1,3)
        self.wforh2 = np.random.normal(0,1,3)
        self.wforh3 = np.random.normal(0,1,3)
        self.wforh4 = np.random.normal(0,1,3)
        self.wforh5 = np.random.normal(0,1,3)
        self.wforhi1 = np.random.normal(0,1,5)
        self.wforhi2 = np.random.normal(0,1,5)
        self.wforhi3 = np.random.normal(0,1,5)
        self.wforhi4 = np.random.normal(0,1,5)
        self.wforhi5 = np.random.normal(0,1,5)
        self.wforo1 = np.random.normal(0,1,5)
        
        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()
        self.b5 = np.random.normal()
        self.b6 = np.random.normal()
        self.b7 = np.random.normal()
        self.b8 = np.random.normal()
        self.b9 = np.random.normal()
        self.b10 = np.random.normal()
        self.b11 = np.random.normal()
        
        
    
    def feedforward(self, x):
        '''This is the first feedforward; it sets the values of the 5 neurons in the first hidden layer of the network by using the three inputs in x.  It returns these values in numpy array y'''
        # x is a numpy array with 3 elements.
        
        h1 = sigmoid(np.dot(x, self.wforh1) + self.b1)
        h2 = sigmoid(np.dot(x, self.wforh2) + self.b2)
        h3 = sigmoid(np.dot(x, self.wforh3) + self.b3)
        h4 = sigmoid(np.dot(x, self.wforh4) + self.b4)
        h5 = sigmoid(np.dot(x, self.wforh5) + self.b5)
        
        y = np.array([h1,h2,h3,h4,h5])
        
        return y
    def feedforward2(self, y):
        '''This is the second feedforward; it sets the values of the 5 neurons in the second hidden layer as well as the singular final output which the method also returns. '''
        hi1 = sigmoid(np.dot(y, self.wforhi1) + self.b6)
        hi2 = sigmoid(np.dot(y, self.wforhi2) + self.b7)
        hi3 = sigmoid(np.dot(y, self.wforhi3) + self.b8)
        hi4 = sigmoid(np.dot(y, self.wforhi4) + self.b9)
        hi5 = sigmoid(np.dot(y, self.wforhi5) + self.b10)
        
        z = np.array([hi1,hi2,hi3,hi4,hi5])
        
        o1 = sigmoid(np.dot(z, self.wforo1) + self.b11)
        
        return o1

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 3) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
          This method changes the weights and biases appied to each neuron to train the network
        '''
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through the entire dataset
        
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)

                h1 = sigmoid(np.dot(x, self.wforh1) + self.b1)
                h2 = sigmoid(np.dot(x, self.wforh2) + self.b2)
                h3 = sigmoid(np.dot(x, self.wforh3) + self.b3)
                h4 = sigmoid(np.dot(x, self.wforh4) + self.b4)
                h5 = sigmoid(np.dot(x, self.wforh5) + self.b5)

                sum_h1 = np.dot(x, self.wforh1) + self.b1
                sum_h2 = np.dot(x, self.wforh2) + self.b2
                sum_h3 = np.dot(x, self.wforh3) + self.b3
                sum_h4 = np.dot(x, self.wforh4) + self.b4
                sum_h5 = np.dot(x, self.wforh5) + self.b5
                
                y = self.feedforward(x)

                hi1 = sigmoid(np.dot(y, self.wforhi1) + self.b6)
                hi2 = sigmoid(np.dot(y, self.wforhi2) + self.b7)
                hi3 = sigmoid(np.dot(y, self.wforhi3) + self.b8)
                hi4 = sigmoid(np.dot(y, self.wforhi4) + self.b9)
                hi5 = sigmoid(np.dot(y, self.wforhi5) + self.b10)
                
                sum_hi1 = np.dot(y, self.wforhi1) + self.b6
                sum_hi2 = np.dot(y, self.wforhi2) + self.b7
                sum_hi3 = np.dot(y, self.wforhi3) + self.b8
                sum_hi4 = np.dot(y, self.wforhi4) + self.b9
                sum_hi5 = np.dot(y, self.wforhi5) + self.b10
                z = np.array([hi1,hi2,hi3,hi4,hi5])
                h = np.array([h1,h2,h3,h4,h5])
                
                
                o1 = self.feedforward2(y)
                sum_o1 = np.dot(z, self.wforo1) + self.b11 
                y_pred = o1
                
                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                

                # Neuron o1 NEW
                d_ypred_d_w = []
                for i in range(5):
                    d_ypred_d_w.append(z[i] * deriv_sigmoid(sum_o1))
                d_ypred_d_b11 = deriv_sigmoid(sum_o1)
                
                d_ypred_d_hi = []
                for i in range(5):
                    d_ypred_d_hi.append(self.wforo1[i] * deriv_sigmoid(sum_o1))
                d_ypred_d_h = []
                for i in range(5):
                     d_ypred_d_h.append(deriv_sigmoid(sum_o1) * (self.wforo1[0] * deriv_sigmoid(sum_hi1) * self.wforhi1[i] + self.wforo1[1] * deriv_sigmoid(sum_hi2) * self.wforhi2[i] + self.wforo1[2] * deriv_sigmoid(sum_hi3) * self.wforhi3[i] + self.wforo1[3] * deriv_sigmoid(sum_hi4) * self.wforhi4[i] + self.wforo1[4] * deriv_sigmoid(sum_hi5) * self.wforhi5[i]))
                #d_ypred_d_h1 = deriv_sigmoid(sum_o1) * (self.wforo1[0] * deriv_sigmoid(sum_hi1) * self.wforhi1[0] + self.wforo1[1] * deriv_sigmoid(sum_hi2) * self.wforhi2[0] + self.wforo1[2] * deriv_sigmoid(sum_hi3) * self.wforhi3[0] + self.wforo1[3] * deriv_sigmoid(sum_hi4) * self.wforhi4[0] + self.wforo1[4] * deriv_sigmoid(sum_hi5) * self.wforhi5[0])               
                #d_ypred_d_h2 = deriv_sigmoid(sum_o1) * (self.wforo1[0] * deriv_sigmoid(sum_hi1) * self.wforhi1[1] + self.wforo1[1] * deriv_sigmoid(sum_hi2) * self.wforhi2[1] + self.wforo1[2] * deriv_sigmoid(sum_hi3) * self.wforhi3[1] + self.wforo1[3] * deriv_sigmoid(sum_hi4) * self.wforhi4[1] + self.wforo1[4] * deriv_sigmoid(sum_hi5) * self.wforhi5[1])
                #d_ypred_d_h3 = deriv_sigmoid(sum_o1) * (self.wforo1[0] * deriv_sigmoid(sum_hi1) * self.wforhi1[2] + self.wforo1[1] * deriv_sigmoid(sum_hi2) * self.wforhi2[2] + self.wforo1[2] * deriv_sigmoid(sum_hi3) * self.wforhi3[2] + self.wforo1[3] * deriv_sigmoid(sum_hi4) * self.wforhi4[2] + self.wforo1[4] * deriv_sigmoid(sum_hi5) * self.wforhi5[2])
                #d_ypred_d_h4 = deriv_sigmoid(sum_o1) * (self.wforo1[0] * deriv_sigmoid(sum_hi1) * self.wforhi1[3] + self.wforo1[1] * deriv_sigmoid(sum_hi2) * self.wforhi2[3] + self.wforo1[2] * deriv_sigmoid(sum_hi3) * self.wforhi3[3] + self.wforo1[3] * deriv_sigmoid(sum_hi4) * self.wforhi4[3] + self.wforo1[4] * deriv_sigmoid(sum_hi5) * self.wforhi5[3])
                #d_ypred_d_h5 = deriv_sigmoid(sum_o1) * (self.wforo1[0] * deriv_sigmoid(sum_hi1) * self.wforhi1[4] + self.wforo1[1] * deriv_sigmoid(sum_hi2) * self.wforhi2[4] + self.wforo1[2] * deriv_sigmoid(sum_hi3) * self.wforhi3[4] + self.wforo1[3] * deriv_sigmoid(sum_hi4) * self.wforhi4[4] + self.wforo1[4] * deriv_sigmoid(sum_hi5) * self.wforhi5[4])
                
                
                
                # Neuron hi1 NEW
                
                d_hi1_d_w = []
                for i in range(5):
                    d_hi1_d_w.append(h[i] * deriv_sigmoid(sum_hi1))
                d_hi1_d_b6 = deriv_sigmoid(sum_hi1)

                # Neuron hi2 NEW
                d_hi2_d_w = []
                for i in range(5):
                    d_hi2_d_w.append(h[i] * deriv_sigmoid(sum_hi2))
                d_hi2_d_b7 = deriv_sigmoid(sum_hi2)

                # Neuron hi3 NEW
                d_hi3_d_w = []
                for i in range(5):
                    d_hi3_d_w.append(h[i] * deriv_sigmoid(sum_hi3))
                d_hi3_d_b8 = deriv_sigmoid(sum_hi3)

                # Neuron hi4 NEW
                d_hi4_d_w = []
                for i in range(5):
                    d_hi4_d_w.append(h[i] * deriv_sigmoid(sum_hi4))
                d_hi4_d_b9 = deriv_sigmoid(sum_hi4)

                #Neuron hi5 NEW
                d_hi5_d_w = []
                for i in range(5):
                    d_hi5_d_w.append(h[i] * deriv_sigmoid(sum_hi5))
                d_hi5_d_b10 = deriv_sigmoid(sum_hi5)

                #Neuron h1 NEW
                d_h1_d_w = []
                for i in range(3):
                    d_h1_d_w.append(x[i] * deriv_sigmoid(sum_h1))
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                #Neuron h2 NEW
                d_h2_d_w = []
                for i in range(3):
                    d_h2_d_w.append(x[i] * deriv_sigmoid(sum_h2))
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                #Neuron h3 NEW
                d_h3_d_w = []
                for i in range(3):
                    d_h3_d_w.append(x[i] * deriv_sigmoid(sum_h3))
                d_h3_d_b3 = deriv_sigmoid(sum_h3)

                #Neuron h4 NEW
                d_h4_d_w = []
                for i in range(3):
                    d_h4_d_w.append(x[i] * deriv_sigmoid(sum_h4))
                d_h4_d_b4 = deriv_sigmoid(sum_h4)

                #Neuron h5 NEW
                d_h5_d_w = []
                for i in range(3):
                    d_h5_d_w.append(x[i] * deriv_sigmoid(sum_h5))
                d_h5_d_b5 = deriv_sigmoid(sum_h5)
                
                #lets create a variable to reduce typing
                variable = learn_rate * d_L_d_ypred

                # --- Update weights and biases
                # Neuron h1
                for i in range(3):
                    self.wforh1[i] -= variable * d_ypred_d_h[0] * d_h1_d_w[i]
                self.b1 -= variable * d_ypred_d_h[0] * d_h1_d_b1

                # Neuron h2
                for i in range(3):
                    self.wforh2[i] -= variable * d_ypred_d_h[1] * d_h2_d_w[i]
                self.b2 -= variable * d_ypred_d_h[1] * d_h2_d_b2

                

                # Neuron h3
                for i in range(3):
                    self.wforh3[i] -= variable * d_ypred_d_h[2] * d_h3_d_w[i]
                self.b3 -= variable * d_ypred_d_h[2] * d_h3_d_b3

                # Neuron h4
                for i in range (3):
                    self.wforh4[i] -= variable * d_ypred_d_h[3] * d_h4_d_w[i]
                self.b4 -= variable * d_ypred_d_h[3] * d_h4_d_b4

                # Neuron h5
                for i in range (3):
                    self.wforh5[i] -= variable * d_ypred_d_h[4] * d_h5_d_w[i]
                self.b5 -= variable * d_ypred_d_h[4] * d_h5_d_b5

                # Neuron hi1
                for i in range (5):
                    self.wforhi1[i] -= variable * d_ypred_d_hi[0] * d_hi1_d_w[i]
                self.b6 -= variable * d_ypred_d_hi[0] * d_hi1_d_b6

                # Neuron hi2
                for i in range (5):
                    self.wforhi2[i] -= variable * d_ypred_d_hi[1] * d_hi2_d_w[i]
                self.b7 -= variable * d_ypred_d_hi[1] * d_hi2_d_b7

                # Neuron hi3
                for i in range(5):
                    self.wforhi3[i] -= variable * d_ypred_d_hi[2] * d_hi3_d_w[i]
                self.b8 -= variable * d_ypred_d_hi[2] * d_hi3_d_b8

                # Neuron hi4
                for i in range(5):
                    self.wforhi4[i] -= variable * d_ypred_d_hi[3] * d_hi4_d_w[i]
                self.b9 -= variable * d_ypred_d_hi[3] * d_hi4_d_b9

                # Neuron hi5
                for i in range(5):
                    self.wforhi5[i] -= variable * d_ypred_d_hi[4] * d_hi5_d_w[i]
                self.b10 -= variable * d_ypred_d_hi[4] * d_hi5_d_b10

                # Neuron o1 NEW
                for i in range(5):
                    self.wforo1[i] -= variable *d_ypred_d_w[i]
                self.b11 -= variable * d_ypred_d_b11

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                firstlayer = np.apply_along_axis(self.feedforward, 1, data)
                y_preds = np.apply_along_axis(self.feedforward2, 1, firstlayer)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
  [-2, -1, 16],  # Cameron
  [25, 6, -5],   # Bill
  [17, 4, 20],   # Cole
  [-15, -6 ,7], # Drake
])
all_y_trues = np.array([
  0.6, # Cameron
  0, # Bill
  1, # Cole
  0.255, # Drake
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)