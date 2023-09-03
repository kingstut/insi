import torch 
import torch.nn as nn 

class Sensor:

    def __init__(self, company, sensor, version, n_channels, prev_state_dict=None):
        self.company = company 
        self.sensor = sensor 
        self.version = version 
        self.n_channels = n_channels
        self.name = "_".join([self.company, self.sensor, str(self.version), str(self.n_channels)])
        self.sensor_model = self.init_sensor_model(prev_state_dict)
        self.stream = None

    def parse_config(self, config):
        #parser code here -> torch.tensor
        file = open(config)
        stream = torch.tensor(file)
        return stream
    
    def init_sensor_model(self, prev_state_dict=None):
        new_model = Sensor_NN(self)
        if prev_state_dict:
            new_model.load_state_dict(torch.load(prev_state_dict))
        return new_model 
    
    def configure(self, config): 
        # config contains time-series output of the senor 
        stream = self.parse_config(config)
        self.stream = stream 
        model = self.sensor_model()
        train_sensor_NN(model, stream)
        self.sensor_model = model
        torch.save(model, "registered/" + self.name+".ptm")


class Sensor_NN(nn.Module):
    def __init__(self, sensor):
        super.__init__(self)
        self.n_channels = sensor.n_channels
        self.transformer = torch.nn.Transformer
    
    def forward(self, x):
        self.transformer(x)
        return x

def train_sensor_NN(model, stream, epochs, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #FIXME
    for _ in range(epochs):
        for (x, y) in stream: 
            optimizer.zero_grad()
            pred = model(x)
            loss = objective(pred, y)
            loss.backward()
            optimizer.step()
    return model

    




