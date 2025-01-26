# ##1## Install dependencies and import them
import re
import csv
import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import urllib.request
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from PyPDF2 import PdfReader
from collections import Counter
import numpy as np

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Configure pandas display options
pd.set_option('display.max_colwidth', 200)


# ##2## Function to parse text from URL
def parse_text_from_url(url):
    html = urllib.request.urlopen(url)
    htmlParse = BeautifulSoup(html, 'html.parser')
    parsed_text = ""
    for para in htmlParse.find_all("p"):
        parsed_text += " " + str(para.get_text())
    return parsed_text


# ##3## Function to parse text from PDF
def parse_text_from_pdf(pdf_path):
    parsed_text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            parsed_text += page.extract_text()
    return parsed_text


# ##4## Convert text into sentences & save in CSV
def save_sentences_to_csv(text, filename='article_text.csv'):
    sentences = [[i] for i in nlp(text).sents]
    with open(filename, 'w', newline='', encoding='utf-8', errors="replace") as myfile:
        writer = csv.writer(myfile)
        writer.writerow(['sentence'])
        writer.writerows(sentences)
    return pd.read_csv(filename, encoding='utf-8')


# ##5## Extract entity pairs
def get_entities(sent):
    ent1 = ""
    ent2 = ""
    prv_tok_dep = ""    # Dependency tag of previous token in the sentence
    prv_tok_text = ""   # Previous token in the sentence
    prefix = ""
    modifier = ""

    # Process each token in the sentence
    for tok in nlp(sent):
        if tok.dep_ != "punct":
            if tok.dep_ == "compound":
                prefix = tok.text
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text
            if tok.dep_.endswith("mod"):
                modifier = tok.text
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text
            if tok.dep_.find("subj") != -1:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""
            if tok.dep_.find("obj") != -1:
                ent2 = modifier + " " + prefix + " " + tok.text

        # Update variables
        prv_tok_dep = tok.dep_
        prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]


# ##6## Get relations for the entities
def get_relation(sent):
    doc = nlp(sent)
    matcher = Matcher(nlp.vocab)
    
    # Define the pattern
    pattern = [{'DEP': 'ROOT'}, 
               {'DEP': 'prep', 'OP': '?'}, 
               {'DEP': 'agent', 'OP': '?'},  
               {'POS': 'ADJ', 'OP': '?'}]
    
    matcher.add("matching_1", [pattern])
    matches = matcher(doc)
    
    # If any match is found, extract the relation
    if matches:
        k = len(matches) - 1
        span = doc[matches[k][1]:matches[k][2]]
        return span.text
    return ""

# New function to score entities and relations
def score_entity_relation(entities, relation):
    # Implement your scoring logic here
    # This is a simple example based on entity types and relation length
    entity_score = sum(2 if ent[1] in ['PERSON', 'ORG', 'GPE'] else 1 for ent in entities)
    relation_score = len(relation) if isinstance(relation, str) else 0
    return entity_score * relation_score


# ##7## Create entity pairs and relations
def process_sentences_to_graph(csv_sentences):
    entity_pairs = []
    relations = []
    scores = []

    for sent in tqdm(csv_sentences['sentence']):
        entities = get_entities(sent)
        relation = get_relation(sent)
        if all(entities):  # Only add if both entities are found
            entity_pairs.append(entities)
            relations.append(relation)
            scores.append(score_entity_relation(entities, relation))

    # Filter based on scores
    threshold = np.percentile(scores, 75)  # Keep top 25% of entity-relation pairs
    filtered_pairs = [(pair, rel, score) for pair, rel, score in zip(entity_pairs, relations, scores) if score >= threshold]

    # Create a DataFrame for the graph
    source = [i[0][0] for i in filtered_pairs]
    target = [i[0][1] for i in filtered_pairs]
    edge = [i[1] for i in filtered_pairs]
    score = [i[2] for i in filtered_pairs]
    kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': edge, 'score': score})


    # Create a directed graph from the DataFrame
    G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.DiGraph())
    
    # Plot the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', font_size=10, edge_color='gray', node_size=3000)
    edge_labels = nx.get_edge_attributes(G, 'edge')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Entity Relation Graph", fontsize=15)
    plt.show()


# ##8## Main function
if __name__ == "__main__":
    # Choose input type
    input_type = input("Enter 'url' to process a web article or 'pdf' to process a PDF file: ").strip().lower()

    if input_type == 'url':
        url = input("Enter the URL: ").strip()
        parsed_text = parse_text_from_url(url)
    elif input_type == 'pdf':
        pdf_path = input("Enter the PDF file path: ").strip()
        parsed_text = parse_text_from_pdf(pdf_path)
    else:
        print("Invalid input type. Please enter either 'url' or 'pdf'.")
        exit()

    # Save sentences to CSV and re-import
    csv_sentences = save_sentences_to_csv(parsed_text)

    # Process and visualize the graph
    process_sentences_to_graph(csv_sentences)
