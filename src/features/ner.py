# Data science
import pandas as pd
import numpy as np
import sklearn

# NER
import spacy
from spacy import displacy
import random
from spacy.matcher import PhraseMatcher
from pathlib import Path

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks", color_codes=True)
import collections
import yaml
import pickle
import streamlit as st
# import imgkit

# Read in the training data 
train_data = pd.read_pickle("../../objects/train.pkl")

# Wrapper to load in the results 
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

# Test out the NER
test_text_human = '''
                        I talked to the Amazon representative earlier in the day but no one responded. The operator isn't human I guess. 
                        The chat ended soon and the support was just awful. I love California. And I love people. 
                  '''

test_text_robot = "Is it a bot ? I thought I have been talking to a fellow human being. The AI system is so freaking good!!!" 

def extract_human(user_input, visualize=False): 
    """
        Takes as input user input, and outputs all the entities extracted. Also made a toggler for visualizing with displacy.''' 
    """
    
    # Load in the trained model
    human_nlp = spacy.load("../../models/human_nlp")
    doc = human_nlp(user_input)
    
    extracted_entities = []
    
    # Extract the intents from the user input
    for ent in doc.ents: 
        extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))
        
    # If I want to visualize 
    if visualize:
        # Visualizing with displacy how the document had its entity tagged 
        colors = {"HUMAN": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"ents": ["HUMAN"], "colors": colors}
        
        # Saves to HTML string 
        html = displacy.render(doc, style="ent", options=options)
        html = html.replace("\n\n", "\n")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
        
    return extracted_entities

def extract_robot(user_input, visualize=False):
    """Takes as input the user input, and outputs all the entities ext"""
    
    # Loading it in
    robot_nlp = spacy.load("../../models/robot_nlp")
    doc = robot_nlp(user_input)

    extracted_entities = []

    # These are the objects you can take out
    for ent in doc.ents:
        extracted_entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))

    # If you want to visualize
    if visualize == True:
        # Visualizing with displaCy how the document had it's entity tagged (runs a server)
        colors = {"ROBOT_AI": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"ents": ["ROBOT_AI"], "colors": colors}
        html = displacy.render(doc, style="ent", options=options)
        # with open("displacy/hardware.html", "a") as out:
        #     out.write(html + "\n")
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    return extracted_entities

def extract_default(user_input): 
    pass 

# Test out the NER
print(extract_human(test_text_human, visualize=True))
print()
print(extract_robot(test_text_robot, visualize=True))


        
                  
