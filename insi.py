import torch
import random

class Probe: 
    """
    A class representing a data collection probe.
    
    Attributes:
        value (float): Value collected by the probe (if used as a 'write' probe).
        measure (float): Measurement collected by the probe (if used as a 'read' probe).
        discrete (bool): Whether probe can only have int values. 
    """
    def __init__(self, value=0, measure=0, discrete=False):
        self.discrete = discrete
        if self.discrete:
            dtype = torch.int
        else:
            dtype=torch.float
        self.value = torch.tensor(value, dtype=dtype) # if probe is W
        self.measure = torch.tensor(measure, dtype=dtype) # if probe is R

    def add(self, h):
        """
        Add a value to the probe.
        
        Args:
            h (float): Value to be added to the probe's value.
        """
        if self.discrete:
            h = int(h)
        self.value +=h 

class Probes: 
    """
    A class representing a collection of probes.

    Attributes:
        probes (dict): A dictionary of probe IDs and their corresponding Probe objects.
        label (float): A label for the measurements of all the probes in this object. 
    """
    def __init__(self, probes={}, label=0):
        self.probes = probes  # dictionary of int key and Probe value 
        self.label = label

    def add_probe(self, probes):
        """
        Add new probes to the Probes collection.

        Args:
            probes (dict or Probe): Either a dictionary of probe IDs and Probe objects,
                                   or a single Probe object to be added.
        """
        id = len(self.probes)
        if isinstance(probes, Probe):
            probes = {id: probes}
        for i in range(len(probes)):
            self.probes[i+id]=probes[i]

    def delete_probe(self, probe_ids):
        """
        Delete probes from the Probes collection.

        Args:
            probe_ids (int or list): ID(s) of the probes to be deleted.
        """
        if not isinstance(probe_ids, list):
            probe_ids = [probe_ids]
        for i in probe_ids:
            del self.probes[i]
    
    def get_values(self):
        """
        Get the collected values from all probes.

        Returns:
            torch.Tensor: A tensor containing the collected values from all probes.
        """
        x = torch.stack([p.value for p in self.probes.values()])
        return x 
    
    def get_measures(self):
        """
        Get the collected measures from all probes.

        Returns:
            torch.Tensor: A tensor containing the collected measures from all probes.
        """
        x = torch.stack([p.measure for p in self.probes.values()])
        return x 
    
    def set_values(self, add_tensor):
        """
        Set the values of the probes using a tensor.

        Args:
            add_tensor (torch.Tensor): A tensor containing values to be added to the probes.
        """
        for i, p in enumerate(self.probes.values()):
            p.add(add_tensor[i])

    def set_measures(self, add_tensor):
        """
        Set the measures of the probes using a tensor.

        Args:
            add_tensor (torch.Tensor): A tensor containing measures to be set to the probes.
        """
        for i, p in enumerate(self.probes.values()):
            p.measure = add_tensor[i]

class Cortex: 
    """
    A class representing a neural network tuning process using probes.

    Attributes:
        model (torch.nn.Module): The neural network model to be tuned.
        objective (function): A function that computes the loss given predictions and labels. 
        probes (Probes): An object of the Probes class.
    """
     
    def __init__(self, probes, model, objective):
        self.model = model  # torch nn model 
        self.objective = objective  # this is a function that returns loss given a prediction
        self.probes = probes  # object of class Probes

    def tune(self, epochs=10, lr=0.001, input=None, first_layer="linear", autograd=False):
        """
        Tune the neural network model using the probes.

        Args:
            epochs (int): Number of tuning epochs.
            lr (float): Learning rate for the tuning process.
            first_layer (str): Type of first layer - "linear" or "emb"
            input (list): List containing inputs that need to be added to probe values. (optional)
        """
        for _ in range(epochs):
            self.model.train()
            self.model.zero_grad()

            if input:
                x = input[random.randint(0, len(input))]
                self.probes.set_measures(x)
            x = self.probes.get_values() + self.probes.get_measures()
            if autograd:
                x.requires_grad=True       

            pred = self.model(x.int())
            loss = self.objective(pred)
            loss.backward()

            if autograd:
                dx = x.grad 
                self.probes.set_values(- lr * dx)
                x.grad = torch.zeros_like(x.grad)
            else:
                #extract layer 1 weights 
                with torch.no_grad():
                    W = list(self.model.parameters())[0]
                    dW = W.grad
                    if first_layer=="linear":
                        dx = ((x.T)/torch.norm(x)**2) @ dW.T @ W
                        self.probes.set_values(- lr * dx)
                    elif first_layer=="emb": 
                        x = torch.clip(x.long(), 0, dW.shape[0]-1)
                        dx = torch.zeros_like(x)
                        for i in range(x.shape[0]):
                            if x[i]>=dW.shape[0]-1:
                                next = x[i]-1
                            else:
                                next = x[i]+1
                            dx[i] = torch.sum((dW[next] - dW[x[i]]) * dW[x[i]] )
                        dx = torch.nan_to_num(dx/dx.abs())
                        self.probes.set_values(-dx)

            print("done, loss:", loss)


def build_model(units, model, objective, epochs, lr):
    """
    Build and train a neural network model using the probe measurements.

    Args:
        units (list): List of Probes objects.
        model (torch.nn.Module): The neural network model to be trained.
        objective (function): A function that computes the loss given predictions.
        epochs (int): Number of training epochs.
        lr (float): learning rate. 

    Returns:
        torch.nn.Module: The trained neural network model.
    """
    train_data = [(p.get_measures, p.label) for p in units]

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for (x, y) in train_data: 
            optimizer.zero_grad()
            pred = model(x)
            loss = objective(pred, y)
            loss.backward()
            optimizer.step()

    return model