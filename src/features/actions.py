import random

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import spacy 

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

# Entity_dictionary 
# Read in the entity dictionary
with open('../objects/entities_non_span.yml') as file:
    entity_dict_non_span = yaml.load(file, Loader=yaml.FullLoader)

# Load in the entities 
human_nlp = spacy.load("../objects/human_nlp")
robot_nlp = spacy.load("../objects/robot_nlp")
package_nlp = spacy.load("../objects/package_nlp")

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
                "Hello, How may I be of help?" 
            ]
        )
    
    # If goodbye 
    def utter_goodbye(self):
        reaffirm = ["Is there anything else I can help you with?"]
        goodbye = [
            "Thank you for this your time. Have a nice day!!!", 
            "Glad I could be of help, have a nice day!!!"
        ]
        return random.choice(goodbye)
        
    
    # Speak to a representative
    def link_to_human(self):
        return random.choice(
            [
                "Alright. Let me direct you to a representative. "
            ]
        )
        
    # Check order status
    def check_order_status(self):
        return random.choice(
            [
                "Please provide me with your order number", 
                "Please provide me with your order number so I can check the status of your shipment" 
            ]
        )
        
    # Returns and refunds 
    def return_refunds(self): 
        return random.choice(
            [
              "The refunds should ideally be processed within 3-5 business days. For more details, refer to the following link: https://www.amazon.com/gp/help/customer/display.html?nodeId=GNW5VKFXMF72FFMR",
              "Thank you for reaching out. I understand that you're concerned about the refund process. Typically, refunds are processed within 3-5 business days. For more detailed information, you can visit the following link: https://www.amazon.com/gp/help/customer/display.html?nodeId=GNW5VKFXMF72FFMR"
            ]
        )
        
    # Faulty_product
    def faulty_product(self):
        return random.choice(
            [
                "I am sorry to hear that you received a faulty product. For details on how to return the product, please refer to the following link: https://www.amazon.com/gp/help/customer/display.html?nodeId=GP7Z9RS868ZP5J9F",
                "I'm truly sorry to hear that you've received a faulty product. To make things right, you can follow our return instructions by visiting the link below:  https://www.amazon.com/gp/help/customer/display.html?nodeId=GP7Z9RS868ZP5J9F"
            ]
        )
        
    # Account issues 
    def account_issues(self):
        return random.choice(
            [
                "In case you have forgotten your password, please click on the forgot password link. You can refer the instructions here: https://www.amazon.com/gp/help/customer/display.html?nodeId=GH3NM2YWEFEL2CQ4", 
                "If your account was locked due to unusual payment activity, please follow the instructions over here: https://www.amazon.com/gp/help/customer/display.html?nodeId=ThMznYkNjxdOL3GTah"
            ]
        )
        
    # Poor customer support 
    def customer_support(self):
        return random.choice(
            [
                "I'm sorry to hear that you've had a poor experience with our customer support. You can email your experience to experience@amazon.com", 
                "I apologize for the poor customer support you've received. I urge you to email your experience at: experience@amazon.com "
            ]
        )
        
    # Issues with payment 
    def payment_issues(self):
        return random.choice(
            [
                "Ensure your card details (number, expiration date, and CVV) are entered correctly, your payment method is up-to-date, and there are no issues with your card by contacting your bank"
            ]
        )
    
    def talk_entity(self, entity_label): 
        # I think I might need to improvise my chatbot along these lines 
        # if entity == "check_order_status": 
        #     return self.check_order_status()
        # elif entity == "return_refunds": 
        #     return self.return_refunds()
        # elif entity == "faulty_product": 
        #     return self.faulty_product()
        # elif entity == "account_issues": 
        #     return self.account_issues()
        # elif entity == "customer_support": 
        #     return self.customer_support()
        # elif entity == "payment_issues": 
        #     return self.payment_issues()
        # else: 
        #     return "I am sorry, I am not sure how to help with that."
        # entity_lst = []
        # Load the human_nlp model 
        # human_nlp = spacy.load("../objects/human_nlp")
        
        # # Load the robot_nlp model
        # robot_nlp = spacy.load("../objects/robot_nlp")
        
        if entity_label=="ROBOT_AI":
            return random.choice("Yeah I am saggy AI bot. I am here to help!!!")
        
        elif entity_label=="HUMAN":
            return random.choice("Please hold on. If you haven't already shared your issue with me, kindly do so now. Otherwise, you will need to wait for a representative to assist you at the following contact number: 1-602-670-3742.")
            
       # So I am going to add all of these 
       ## check_order_status AND wrong_items 
       ## account_issues - unable  to login and all - untimely delivery 
       ## unauthorized access - faulty_product - customer_service 
       ## customer_support not working 
       ## faulty payment system 
       
    def fallback(self):
        return random.choice(
            [
                "I apologize, I am not sure how to help with that. For more instructions, refer this link: https://www.amazon.com/hz/contact-us/"
            ]
        )
       
       
       

