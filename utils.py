import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_chunk_to_doc(texts, max_chunk_size=512):
    chunk_to_doc = []
    cid = 0
    for d in range(len(texts)):
        words = texts[d].split()
        num_chunks = (len(words) + max_chunk_size - 1) // max_chunk_size
        chunk_to_doc += [d] * num_chunks
        cid += num_chunks
    return chunk_to_doc

def calculate_classification_metrics(y_true, y_pred, is_multi_label):
    if is_multi_label:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='micro')
        prec = precision_score(y_true, y_pred, average='micro')
        rec = recall_score(y_true, y_pred, average='micro')
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
    return acc, f1, prec, rec

def load_previous_results(file_path='benchmark_results.json'):
    import json
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_results(results, file_path='benchmark_results.json'):
    import json
    with open(file_path, 'w') as f:
        json.dump(results, f, default=lambda x: float(x) if isinstance(x, np.float32) else x)