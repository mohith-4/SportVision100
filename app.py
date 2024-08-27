### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names =  ['air hockey','ampute football','archery','arm wrestling','axe throwing','balance beam','barell racing',
 'baseball','basketball','baton twirling','bike polo','billiards','bmx',
 'bobsled','bowling','boxing','bull riding','bungee jumping','canoe slamon','cheerleading','chuckwagon racing','cricket','croquet','curling','disc golf','fencing','field hockey','figure skating men','figure skating pairs','figure skating women','fly fishing','football',
 'formula 1 racing','frisbee','gaga','giant slalom','golf','hammer throw','hang gliding','harness racing','high jump','hockey','horse jumping','horse racing',
 'horseshoe pitching','hurdles','hydroplane racing','ice climbing',
 'ice yachting','jai alai','javelin','jousting','judo','lacrosse','log rolling','luge','motorcycle racing','mushing','nascar racing','olympic wrestling','parallel bar',
 'pole climbing','pole dancing','pole vault',
 'polo','pommel horse','rings','rock climbing','roller derby','rollerblade racing','rowing','rugby','sailboat racing','shot put','shuffleboard','sidecar racing','ski jumping',
 'sky surfing','skydiving','snow boarding','snowmobile racing','speed skating',
 'steer wrestling','sumo wrestling','surfing','swimming','table tennis','tennis','track bicycle','trapeze','tug of war','ultimate','uneven bars','volleyball','water cycling',
 'water polo','weightlifting','wheelchair basketball','wheelchair racing','wingsuit flying']


# Create EffNetB2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes= 100,  
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_feature_extractor_sports_100.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
 
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "SportVision 100 Mini"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of Sports."

example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description )

# Launch the demo!
demo.launch()
