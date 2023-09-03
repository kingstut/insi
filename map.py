import torch 
import torch.nn.functional as F 

def map(sensor1, sensor2):
    # what does a  sensor 1 stream look like for sensor 2 and vice versa? 
    ch1 = sensor1.n_channels()
    ch2 = sensor2.n_channels()

    # approach 1 : directly map configs onto each other (2 way NN)
    model = bi_nn(ch1, ch2)
    model.train(sensor1.stream, sensor2.stream, 10)

    # approach 2 : map sensor_models 


class bi_nn():
    def __init__(self, ch1, ch2, hidden=100):
        self.ch1 = ch1
        self.ch2 = ch2 
        self.layer1 = torch.randn(ch1, hidden)
        self.layer2 = torch.randn(hidden, ch2)
    
    def forward(self, x, y=None):
        assert x.shape[0] in [self.ch1, self.ch2], "input should be in " + str([self.ch1, self.ch2])
        if x.shape[0] == self.ch1:
            W1 = self.layer1
            W2 = self.layer2
        else:
            W1 = self.layer2.T
            W2 = self.layer1.T

        preact = x @ W1
        h = torch.relu(preact)
        out = h @ W2

        loss = None
        if y: 
            # get gradients
            loss = F.cross_entropy(out, y) 
            dlogits = F.softmax(out, 1)
            dlogits -= y
            dW2 = h.T @ dlogits
            dpreact = (h>0) * (dlogits @ self.layer2.T)
            dW1 = x.T @ dpreact

            # update 
            if x.shape[0] == self.ch1:
                self.layer1 -=0.001 * dW1
                self.layer2 -=0.001 * dW2
            else: 
                self.layer1 -=0.001 * dW2.T
                self.layer2 -=0.001 * dW1.T                

        return out, loss
    
    def train(self, stm1, stm2, epochs):
        for _ in range(epochs):
            ctx = 100
            r = torch.randint(0, min(stm1.shape[0], stm2.shape[0]) -100)
            x = stm1[r: r+ctx]
            y = stm2[r: r+ctx]
            out,loss = self.forward(x, y)
            print(loss)
                


        

