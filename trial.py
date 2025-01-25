import PyPDF2
import spacy
import re
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Extract text from PDF
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

# Step 3: Extract entities and relations
def extract_entity_relations(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    entities = []
    relations = []
    sentences = list(doc.sents)
    
    for sent in sentences:
        entities_in_sent = [(ent.text, ent.label_) for ent in sent.ents]
        entities.extend(entities_in_sent)
        
        # Heuristic-based relation extraction (example: based on 'involving', 'between', etc.)
        if 'involving' in sent.text or 'between' in sent.text:
            tokens = [tok.text for tok in sent]
            orgs = [ent.text for ent in sent.ents if ent.label_ in ["ORG", "PERSON"]]
            if len(orgs) >= 2:
                for i in range(len(orgs) - 1):
                    relations.append((orgs[i], orgs[i + 1]))
    
    return set(entities), set(relations)

# Step 4: Visualize relations as a graph
def visualize_graph(entities, relations):
    G = nx.Graph()
    
    for entity, label in entities:
        G.add_node(entity, label=label)
    
    for source, target in relations:
        G.add_edge(source, target)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=3000, font_size=10, font_weight="bold")
    plt.title("Entity Relation Graph")
    plt.show()

# Main function
if __name__ == "__main__":
    pdf_path = r"E:\vscode\datathon\1.pdf"
    
    # Extract and process text
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    
    # Extract entities and relations
    entities, relations = extract_entity_relations(cleaned_text)
    
    # Print results
    print("Entities:", entities)
    print("Relations:", relations)
    
    # Visualize relations
    visualize_graph(entities, relations)
