import random

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
# Data science
import pandas as pd
import numpy as np
import re
import collections
import yaml


## Load in the objects

# Load in thew training data  
train_data = pd.read_pickle("../objects/train.pkl")

# Load in the entities 
class Actions: 
    memory = {"ROBOT_AI": [], "HUMAN": [], "PACKAGE": []}
    
    def __init__(self, startup): 
        # The initial prompt 
        self.startup = startup
        
    # If greet 
    def utter_greet(self):
        # Storing the bank of responses 
        return random.choice(
            [
                "Hello! I am the Saggy AI robot. How may I assist you today?", 
                "Hello, How may I be of help?", 
            ]
        )
        
        # If goodbye 
        def utter_goodbye(self):
            reaffirm = ["Is there anything else I can help you with?"]
            goodbye = [
                "Thank you for this your time. Have a nice day!!!", 
                "Glad I could be of help, have a nice day!!!", 
            ]
            return random.choice(goodbye)

