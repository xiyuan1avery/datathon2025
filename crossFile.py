import PyPDF2
import spacy
import re
import networkx as nx
import matplotlib.pyplot as plt
import os

# Step 1: Extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Step 2: Clean and preprocess text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^\w\s.,]', '', text)  # Remove special characters (excluding .,)
    return text

# Step 3: Extract entities and relations from text
def extract_entity_relations(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    entities = []
    relations = []
    sentences = list(doc.sents)
    
    for sent in sentences:
        entities_in_sent = [(ent.text, ent.label_) for ent in sent.ents]
        entities.extend(entities_in_sent)
        
        # Relation extraction heuristic
        if 'involving' in sent.text or 'between' in sent.text:
            orgs = [ent.text for ent in sent.ents if ent.label_ in ["ORG", "PERSON"]]
            if len(orgs) >= 2:
                for i in range(len(orgs) - 1):
                    relations.append((orgs[i], orgs[i + 1]))
    
    return set(entities), set(relations)

# Step 4: Process multiple files and merge results
def process_multiple_files(file_paths):
    all_entities = set()
    all_relations = set()
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        raw_text = extract_text_from_pdf(file_path)
        cleaned_text = clean_text(raw_text)
        entities, relations = extract_entity_relations(cleaned_text)
        all_entities.update(entities)
        all_relations.update(relations)
    
    return all_entities, all_relations

# Step 5: Visualize relations as a graph
def visualize_graph(entities, relations):
    G = nx.Graph()
    
    for entity, label in entities:
        G.add_node(entity, label=label)
    
    for source, target in relations:
        G.add_edge(source, target)
    
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000, font_size=10, font_weight="bold")
    plt.title("Entity Relation Graph (Cross-File)")
    plt.show()

# Main function
if __name__ == "__main__":
    # Directory containing multiple PDF files
    directory_path = r"E:\vscode\datathon\pdfs"  # Update to your directory path
    
    # Collect all PDF file paths
    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".pdf")]
    
    # Extract and merge results from all files
    all_entities, all_relations = process_multiple_files(file_paths)
    
    # Print results
    print("Entities:", all_entities)
    print("Relations:", all_relations)
    
    # Visualize the unified graph
    visualize_graph(all_entities, all_relations)
