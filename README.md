# insi

**insi** is short for in-silico. 

It is Python code designed to:
- Read in values from probes of any complex system.
- Perform machine learning computation on the probe measurements to model an objective.
- Suggest values for probes to output onto the system to achieve the previously mentioned objective.


Example use cases:

- **Your complex system is a neural network that predicts an integer from a hand-written image (MNIST dataset) and you want to manipulate the network to predict more 9s.**
1. You probe your network to read any 20 activation values when the network predicts 9. The label for these probe measurements is 9. 
3. You probe your network to read any 20 activation values when the network predicts some other number. The label for these probe measurements is that other number. 
4.  Do this multiple times to form a training dataset for a model that predicts whether the network's prediction is 9 given 20 probe measurements. 
5. **insi** reverse engineers the values you would need at the 20 probes to increase your chances of getting a 9 prediction. 
6. You can now add these values to the probed activations to create a network that has a bias towards 9.

- **Your complex system is a brain and you want more high serotonin level areas.**
1. You find a region where serotonin levels are high (100). 
2. You probe it at 10 points to get the electrical signal measurements for those points. The label for these probe measurements is the serotonin level. 
3. You find a region where serotonin levels are moderate (36).
4. Repeat step 2. 
5. Do this multiple times to form a training dataset for a model that predicts the serotonin level given 10 probe measurements. 
6. **insi** reverse engineers the values you would need at the 10 probes to get high serotonin levels. 
7. You can now induce these values at the probes to create a new high serotonin region of the brain. 

## Features

- Create and manage data collection probes.
- Tune neural network models using data collected by probes.
- Analyze and manipulate data for a desired outcome.

##  Usage
Here's a brief overview of how to use insi:

Creating and Managing Probes

```python
from insi import Probe, Probes

# Create probes
probe1 = Probe()
probe2 = Probe()

# Create a Probes collection and add probes
probes_collection = Probes()
probes_collection.add_probe({0: probe1, 1: probe2})

# Set label for probe collection
probes_collection.label = 3

# Delete a probe
probes_collection.delete_probe(1)
```
Training Models from probe measurements and labels

```python
from insi import build_model

# List of Probes objects
units = [probe_unit1, probe_unit2, ...]

# Initialize your neural network model and objective function
model = YourTorchModel()
objective = YourObjectiveFunction()

# Build and train the neural network model
trained_model = build_model(units, model, objective, epochs=20)
```

Finding probe values to be induced for the desired objective

```python
import torch
from cortex_probes import Probes, Cortex

# Create a probe collection
probes = Probes()

# Initialize your neural network model and objective function
model = YourNeuralNetworkModel()
objective = YourObjectiveFunction()

# Initialize Cortex instance
cortex = Cortex(probes, model, objective)

# Tune the neural network using the probes
cortex.tune(epochs=10, lr=0.001)
```
