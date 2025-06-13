import numpy as np
class NeuralNetwork:
    def __init__(self,  input_size, output_size, hidden_layers):
        # Initialize weights and biases
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        
        # initialise weights
        self.params = {}
        self.params = {
            "W1" : np.random.rand(self.input_size, self.hidden_layers) * np.sqrt(1/self.input_size),
            "b1" : np.zeros((1, self.hidden_layers)) * np.sqrt(1/self.input_size),
            "W2" : np.random.rand(self.hidden_layers, self.output_size) * np.sqrt(1/self.hidden_layers),
            "b2" : np.zeros((1, self.output_size)) * np.sqrt(1/self.hidden_layers)
        }

    def sigmoid(self, z):
        # Activation function
        # sigmoid = 1/(1+e^x)
        z = np.asarray(z)
        
        # handle overflow of numbers with e
        positives = (z >= 0)
        negatives = ~(positives)
        out = np.empty_like(z)
        out[positives] = 1/(1+np.exp(-(z[positives])))
        out[negatives] = np.exp(z[negatives])/(1+np.exp(z[negatives]))
        return out

    def sigmoid_derivative(self, z):
         # Derivative of sigmoid
        return z*(1-z)
    
    def softmax(self, z):
        # to avoid overflow in expotential subtract with max 
        exp_values = np.exp(z - np.max(z, axis = 1, keepdims = True))
        normalise_values = exp_values/ np.sum(exp_values, axis = 1, keepdims = True )
        
        return normalise_values
       

    def forward(self, X, y):
        # Forward pass through the network
        # Save intermediate values for backprop
        self.z1 = X.dot(self.params["W1"]) + self.params["b1"]
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1.dot(self.params["W2"]) + self.params["b2"]
        self.a2 = self.softmax(self.z2)
        return self.a2    

    
    def backward(self, X, y):
        # Backpropagation to calculate gradients
        current_batch_size = y.shape[0]
        dZ2 =  self.a2 - y
        dw2 = (1/current_batch_size)*(np.matmul(self.a1.T,dZ2 )  )
        db2 = (1/current_batch_size) * np.sum(dZ2, axis = 0, keepdims = True)
        dA1 = np.matmul( dZ2, self.params["W2"].T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dw1 = (1/current_batch_size) * np.matmul( X.T,dZ1)
        db1 = (1/current_batch_size) * np.sum(dZ1, axis = 0, keepdims = True)
        self.grads = {"W1" : dw1, "W2" : dw2 , "b1" : db1, "b2" : db2}
    
    def optimise(self, optimiser = 'sgd', learning_rate = 0.01, beta = 0.9):
        
        #optimiser  only SGD and momentum 
        if optimiser == "sgd" :
            for key in self.params:
                self.params[key] = self.params[key] - learning_rate * self.grads[key]
        elif optimiser == 'momentum':
            for key in self.params :
                self.momemtum_opt[key] = beta * self.momemtum_opt[key] + (1-beta) * self.grads[key]
                self.params[key] = self.params[key] - learning_rate * self.momemtum_opt[key]
        else:
            raise ValueError("Optimiser is currently not available please use SGD or Momentum")
            
    def cross_entropy(self, y, p):
        # Loss function 
        p = np.clip(p, 1e-12, 1-1e-12)
        return -np.sum(y * np.log(p)) / y.shape[0]
  
        
    def train(self, X, y, epoch = 5, learning_rate = 0.01, batch_size = 64, optimiser = 'sgd') :
        
        l = []
        self.batch_size = batch_size
        total_size = y.shape[0]
        num_batchs =total_size // self.batch_size
        
        if optimiser == 'momentum':
            self.momemtum_opt = {
                "W1": np.zeros(self.params["W1"].shape),
                "b1": np.zeros(self.params["b1"].shape),
                "W2": np.zeros(self.params["W2"].shape),
                "b2": np.zeros(self.params["b2"].shape),
            }
                
        for i in range(epoch):
            ## add permutation to shuffle data
            # perms = np.random.permutation(X.shape[0])
            # X = X[perms]
            # y = y[perms]
            for j in range(num_batchs):
                start = j*batch_size 
                end = min(start + batch_size, total_size)
                X_batch = X[start : end]
                y_batch = y[start : end]
                output = self.forward(X_batch, y_batch)
                loss_sub = self.cross_entropy(y_batch, output)
                self.backward(X_batch, y_batch )
                self.optimise(optimiser = optimiser, learning_rate = learning_rate )
            if i % 1000 == 0:
                print (' At the end Epoch : ', i, ' Loss : ', loss_sub )
            # print (' At the end Epoch : ', i, ' Loss : ', loss_sub )
                

    def predict_prob(self, X_test):
        output = self.forward(X_test, None)
        return output
    
    def predict(self, X_test):
        predict_probabilities  = self.predict_prob(X_test)
        return np.argmax(predict_probabilities, axis = -1)
    
    def accuracy(self, y, y_pred):  
        return np.sum( np.argmax(y , axis = -1) == y_pred)/ y.shape[0]
                
        