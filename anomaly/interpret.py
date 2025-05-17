import torch
from transformers import AutoTokenizer, AutoModel
from captum.attr import LayerIntegratedGradients, visualization
import numpy as np
import os
import time
import sys
from os import name
import json
import pandas as pd
import pdfkit

from train import raw2label, label2idx, idx2label
from model import AnomalySeverity


def get_config():
    config_file = open(os.path.join('..', 'config.json'))
    config_file = json.load(config_file)
    # now = datetime.now()
    # now = now.strftime("%Y%m%d_%H%M%S")
    # output_dir = os.path.join(config_file['output_dir'], now)
    # output_dir = os.path.join(config_file['output_dir'], '27112022_190057')
    # previous_step = 0
    # previous_status = False
    use_cuda = True if torch.cuda.is_available() == True else False
    

    wkhtml_path = ""
    if name == 'nt':
        wkhtml_path = config_file['wkhtml_path']['windows']
    # for mac and linux(here, os.name is 'posix')
    else:
        wkhtml_path = config_file['wkhtml_path']['linux']

    return {
        "output_dir": 'output',
        "model_dir": 'pytorch_model.pt',
        # "previous_step": previous_step,
        # "previous_status": previous_status,
        "wkhtml_path": wkhtml_path,
        "app_version": config_file['app_version'],
        "use_cuda": use_cuda,
        "evidence_dir": os.path.join('..', 'data'),
    }

class2color = {
    'normal': '#4CAF50',
    'low': '#FFC107',
    'medium': '#FF5722',
    'high': '#FF5722', 
}


def reconstruct_tokens(tokens, attributions):
    words = []
    attribution_score = []
    current_word = ""
    current_attr = 0
    for token, attribution in zip(tokens, attributions):
        if token.startswith("##"):
            current_word += token[2:]  # Remove "##" and append to the current word
            current_attr += attribution
        else:
            if current_word:
                words.append(current_word)
                attribution_score.append(current_attr)
            current_word = token
            current_attr = attribution
    # Append the last word
    if current_word:
        words.append(current_word)
        attribution_score.append(current_attr)

    return words, attribution_score

def infer_pred(model, input_ids, attention_mask):
    model.eval()
    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=-1)
    label = torch.argmax(probs, dim=-1).cpu().detach().numpy()
    return idx2label.get(label), probs.cpu().detach().numpy()

# def infer_pred(model, input_ids, attention_mask):
#     logits = model(input_ids, attention_mask)
#     # return torch.argmax(torch.softmax(model(input)))
#     # for logits in logitss:
#     # after_sigmoid = [1 if (torch.nn.Sigmoid(logits) >= 0.5) else 0]
#     [after_sigmoid] = torch.sigmoid(logits).cpu().detach().numpy()
#     # print(f'after_sigmoid: {after_sigmoid}')
#     # after_sigmoid = [1 if (torch.sigmoid(element) >= 0.5) else 0 for element in after_sigmoid]
#     # print(f'after_sigmoid: {after_sigmoid}')
#     vector_label = [1 if element >= 0.5 else 0 for element in after_sigmoid]
#     # print(f'vector_label: {vector_label}')
#     labelidx, label = label2class(vector_label)
#     label_prob = after_sigmoid[labelidx]
#     if label == 'normal':
#         label_prob = 1 - after_sigmoid[labelidx]
#     return label, label_prob


def scale_attribution(distribution):
    """
    Scales the input distribution to the range [-1, 1].

    Parameters:
    distribution (numpy.ndarray): The input distribution of values to be scaled.

    Returns:
    numpy.ndarray: The scaled distribution with values in the range [-1, 1].
    """
    distribution = np.asarray(distribution)
    min_val = np.min(distribution)
    max_val = np.max(distribution)
    scaled_distribution = 2 * (distribution - min_val) / (max_val - min_val) - 1
    return scaled_distribution


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    # attributions = attributions.sum(dim=2).squeeze(0)
    # attributions = attributions / torch.norm(attributions)
    # attributions = attributions.cpu().detach().numpy()
    attributions = np.array(attributions)
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            'high',
                            attributions.sum(),
                            text,
                            delta))


vis_data_records_ig = []
def interpret(model, tokenizer, max_seq_length, text, label):
    # device = "cpu"
    # bert_model_name = "bert-base-cased"
    # tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    # bert_model = BertModel.from_pretrained(bert_model_name).to(device)
    # max_seq_length = 64
    # # Load your model
    # model_path = os.path.join('best_models/investigate_15/filtered/gru/pytorch_model.pt')
    # model = DroneLogInter(bert_model, 'gru',
                                    # 1, 3, False, True, 384, 'avg', False, False, 4, None, False, 'cross_entropy').to(device)
    # model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Example text
    # text = "Fly with caution and ensure the aircraft remains within your line of sight."
    # label = 'normal'
    labelidx = label2idx.get(label)
    # print(f'labelidx: {labelidx}')

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    lig = LayerIntegratedGradients(model, model.embedding_model.embeddings)
    pred_label, pred_prob = infer_pred(model, input_ids, attention_mask)

    target_class = label2idx.get('high') # Example target class
    attributions, delta = lig.attribute(inputs=input_ids, 
                                        baselines=input_ids*0, 
                                        additional_forward_args=(attention_mask,),
                                        target=target_class,
                                        return_convergence_delta=True)
    # Sum the attributions across embedding dimensions
    attributions = attributions.sum(dim=-1).squeeze(0)
    # print(f'sum_attr: {attributions}')
    # Normalize the attributions for better visualization
    attributions = attributions / torch.norm(attributions)
    # print(f'normalized: {attributions}')
    # attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    # Convert attributions to numpy
    attributions = attributions.cpu().detach().numpy()

    # Get the tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens, attributions = reconstruct_tokens(tokens, attributions)
    add_attributions_to_visualizer(attributions, tokens, pred_prob, pred_label, label, delta, vis_data_records_ig)
    return attributions, tokens, label, pred_label, pred_prob


# Function to plot attributions
# def plot_attributions(tokens, attributions, label, filename):
#     fig, ax = plt.subplots(figsize=(12, 2))
#     ax.axis('off')
#     color = class2color.get(label)
#     x = 0.05  # starting x coordinate for the first word
#     y = 0.5   # y coordinate, same for all words to display in one line

#     for token, attribution in zip(tokens, attributions):
#         # print(f'attributions: {attributions}')
#         if attribution < 0:
#             attribution = abs(attribution)
#             color = class2color.get('high')
#         rgba_color = to_rgba(color, alpha=attribution)
#         bbox_props = dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor=rgba_color)
#         ax.text(x, y, token, ha='center', va='center', rotation=0, size=12, bbox=bbox_props)
#         x += len(token) * 0.02 + 0.05  # adjust x position for the next word

#     # plt.show()
#     plt.savefig(os.path.join('visualization', 'interpretation', f'{filename}.png'))
#     plt.close()


# Function to plot attributions with text wrapping
# def plot_attributions(tokens, attributions, label, filename):
#     fig, ax = plt.subplots(figsize=(12, 2))
#     ax.axis('off')
#     # color = class2color.get(label)
    
#     # Start coordinates
#     x = 0.05
#     y = 0.9
#     line_height = 0.20  # Adjust the line height for wrapping
#     attributions = scale_attribution(attributions)
#     # print(f'scaled: {attributions}')
#     for token, attribution in zip(tokens, attributions):
#         if attribution < 0:
#             # print(f'negative: {attribution}')
#             attribution = abs(attribution)
#             color = '#FF5722'
#             # color = '#4CAF50'
#             rgba_color = to_rgba(color, alpha=attribution)
#             bbox_props = dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor=rgba_color)
#         else:
#             color = '#4CAF50'
#             rgba_color = to_rgba(color, alpha=attribution)
#             bbox_props = dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor=rgba_color)
#             # print(f'positive: {attribution}')
#         # print(f'token: {token}')
#         # print(f'color fuck you?: {color}')
        
#         # Check if the token fits in the current line
#         if x + len(token) * 0.02 > 1.0:
#             x = 0.05  # Reset x to the start of the line
#             y -= line_height  # Move to the next line

#         ax.text(x, y, token, ha='left', va='center', rotation=0, size=12, bbox=bbox_props)
#         x += len(token) * 0.02 + 0.03  # Adjust x position for the next word, add space
    
#     plt.savefig(os.path.join('visualization', 'interpretation-euclidean-MiniLM-L6', f'{filename}.png'))
#     plt.savefig(os.path.join('visualization', 'interpretation-euclidean-MiniLM-L6', f'{filename}.pdf'))
#     plt.close()


def main():
    # Get the configuration
    config = get_config()

    os.makedirs(config['output_dir'], exist_ok=True)

    print("Starting interpretability report generation...\n")
    time.sleep(2)
    # Load the model

    print("Loading the model...")
    # model_dir = os.path.join(config['model_dir'], 'pytorch_model.pt')
    model_path = os.path.join(config['model_dir'], 'pytorch_model.pt')
    if not os.path.exists('pytorch_model.pt'):
        print("The model not found!")
        sys.exit(0)

    device = torch.device('cpu')
    # tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    # bert_model = BertModel.from_pretrained(bert_model_name).to(device)
    model_name_path = "swardiantara/drone-ordinal-all"
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    embedding_model = AutoModel.from_pretrained(model_name_path).to(device)

    max_seq_length = 64
    # # Load your model
    model = AnomalySeverity(embedding_model, tokenizer).to(device)
    model.load_state_dict(torch.load('pytorch_model.pt'))
    print("Model loaded successfully...")

    # Load the test set
    print("Loading the test set...")
    evidence_file = os.path.join('..', 'data', 'parsed_DJIFlightRecord_2025-05-12_[08-20-56].csv.xlsx')
    if not os.path.isfile(evidence_file):
        print("The test set not found!")
        sys.exit(0)

    test_set = pd.read_excel(evidence_file)
    # test_set = pd.read_csv(os.path.join(config['evidence_dir'], config['evidence_filename']))
    print("Test set loaded successfully...")

    print("Start interpreting...")
    attribution_list = []
    for index, row in test_set.iterrows():
        # print(f'row: {row}')
        # print(f'row[0]: {row[0]}')
        # print(f'row[1]: {row[1]}')
        # return 0
        attributions, tokens, label, pred_label, pred_prob = interpret(model, tokenizer, max_seq_length, row['message'], 'Unknown')
        attribution_list.append([attributions, tokens, label, pred_label, pred_prob])
        # plot_attributions(tokens, attributions, 'high', f'test_{index}')
        # print(attributions, tokens, label, pred_label, pred_prob)
        # if index >= 10:
        #     break
    with open(os.path.join('output', 'raw_attribution.json'), 'w') as json_file:
        json.dump(attribution_list, json_file, indent=4)
    html_output = visualization.visualize_text(vis_data_records_ig)
    with open(os.path.join('output', 'interpret_report_high_drone-severity.html'), 'w') as f:
        f.write(html_output.data)
    path_to_wkhtmltopdf = config['wkhtml_path']
    config_wkhtml = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
    pdfkit.from_string(html_output.data, os.path.join('output', 'interpret_report_high_drone-severity.pdf'), configuration=config_wkhtml)
    print("Finish interpreting...")

if __name__ == "__main__":
    main()