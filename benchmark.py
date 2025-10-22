import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from sentence_transformers import SentenceTransformer, util
import torch
from src.trainer import CustomTrainer
from src.aske import ASKE
from src.utils import get_chunk_to_doc, calculate_classification_metrics, load_previous_results, save_results

def benchmark_model(model_name, dataset_name, embedder):
    try:
        dataset = load_dataset("lex_glue", dataset_name)
        text_field = next(f for f in ['text', 'facts'] if f in dataset['train'].features)
        label_field = next(f for f in ['labels', 'label'] if f in dataset['train'].features)
        is_multi_label = isinstance(dataset['train'][0][label_field], list)
        num_labels = len(set(l for example in dataset['train'] for l in (example[label_field] if is_multi_label else [example[label_field]])))
        max_length = 4096 if 'longformer' in model_name.lower() else 512
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification" if is_multi_label else "single_label_classification"
        )

        def preprocess(examples):
            texts = examples[text_field]
            processed_texts = [' '.join(t) if isinstance(t, list) else str(t) for t in texts]
            tokenized = tokenizer(processed_texts, truncation=True, max_length=max_length, padding=True)
            if is_multi_label:
                labels = np.zeros((len(examples[label_field]), num_labels), dtype=np.float32)
                for i, ls in enumerate(examples[label_field]):
                    for l in ls:
                        labels[i, l] = 1
                tokenized['labels'] = torch.FloatTensor(labels)
            else:
                tokenized['labels'] = examples[label_field]
            return tokenized

        train_ds = dataset['train'].map(preprocess, batched=True)
        test_ds = dataset['test'].map(preprocess, batched=True)
        batch_size = 1 if 'longformer' in model_name.lower() else 8
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,
            gradient_checkpointing='longformer' in model_name.lower()
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=lambda eval_pred: {
                'accuracy': accuracy_score(eval_pred.label_ids, (eval_pred.predictions > 0).astype(int) if is_multi_label else np.argmax(eval_pred.predictions, axis=1)),
                'f1': f1_score(eval_pred.label_ids, (eval_pred.predictions > 0).astype(int) if is_multi_label else np.argmax(eval_pred.predictions, axis=1), average='micro' if is_multi_label else 'macro'),
                'precision': precision_score(eval_pred.label_ids, (eval_pred.predictions > 0).astype(int) if is_multi_label else np.argmax(eval_pred.predictions, axis=1), average='micro' if is_multi_label else 'macro'),
                'recall': recall_score(eval_pred.label_ids, (eval_pred.predictions > 0).astype(int) if is_multi_label else np.argmax(eval_pred.predictions, axis=1), average='micro' if is_multi_label else 'macro')
            },
            is_multi_label=is_multi_label
        )
        start_time = time.time()
        trainer.train()
        preds = trainer.predict(test_ds)
        time_taken = time.time() - start_time
        if is_multi_label:
            y_pred = (preds.predictions > 0).astype(int)
            y_true = preds.label_ids
        else:
            y_pred = np.argmax(preds.predictions, axis=1)
            y_true = preds.label_ids
        acc, f1, prec, rec = calculate_classification_metrics(y_true, y_pred, is_multi_label)
        label_names = dataset['train'].features[label_field].names if hasattr(dataset['train'].features[label_field], 'names') else list(range(num_labels))
        sim_scores = []
        for i in range(len(y_true)):
            true_idx = np.where(y_true[i] == 1)[0] if is_multi_label else [y_true[i]]
            pred_idx = np.where(y_pred[i] == 1)[0] if is_multi_label else [y_pred[i]]
            true_labels = [str(label_names[idx]) for idx in true_idx]
            pred_labels = [str(label_names[idx]) for idx in pred_idx]
            true_emb = embedder.encode(true_labels)
            pred_emb = embedder.encode(pred_labels)
            for t in true_emb:
                if len(pred_emb) > 0:
                    max_sim = max(util.cos_sim(t, p)[0][0].item() for p in pred_emb)
                    sim_scores.append(max_sim)
        mean_sim = np.mean(sim_scores) if sim_scores else 0
        std_sim = np.std(sim_scores) if sim_scores else 0
        ci_sim = 1.96 * std_sim / np.sqrt(len(sim_scores)) if sim_scores else 0
        pairwise_sim = []
        for i in range(len(y_pred)):
            pred_idx = np.where(y_pred[i] == 1)[0] if is_multi_label else [y_pred[i]]
            pred_labels = [str(label_names[idx]) for idx in pred_idx]
            if len(pred_labels) > 1:
                emb = embedder.encode(pred_labels)
                for j in range(len(emb)):
                    for k in range(j + 1, len(emb)):
                        pairwise_sim.append(util.cos_sim(emb[j], emb[k])[0][0].item())
        mean_pairwise = np.mean(pairwise_sim) if pairwise_sim else 0
        return {'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec, 'mean_sim': mean_sim, 'std_sim': std_sim, 'ci_sim': ci_sim, 'mean_pairwise': mean_pairwise, 'time': time_taken}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"CUDA OutOfMemoryError for {model_name} on {dataset_name}: {str(e)}")
        return {'error': str(e), 'acc': np.nan, 'f1': np.nan, 'prec': np.nan, 'rec': np.nan, 'mean_sim': np.nan, 'std_sim': np.nan, 'ci_sim': np.nan, 'mean_pairwise': np.nan, 'time': np.nan}
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return {'error': str(e), 'acc': np.nan, 'f1': np.nan, 'prec': np.nan, 'rec': np.nan, 'mean_sim': np.nan, 'std_sim': np.nan, 'ci_sim': np.nan, 'mean_pairwise': np.nan, 'time': np.nan}

def evaluate_aske(dataset_name):
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    dataset = load_dataset("lex_glue", dataset_name)
    text_field = next(f for f in ['text', 'facts'] if f in dataset['test'].features)
    label_field = next(f for f in ['labels', 'label'] if f in dataset['test'].features)
    texts = [' '.join(ex[text_field]) if isinstance(ex[text_field], list) else str(ex[text_field]) for ex in dataset['test']]
    labels_list = [ex[label_field] for ex in dataset['test']]
    is_multi_label = isinstance(labels_list[0], list)
    label_names = dataset['train'].features[label_field].names if hasattr(dataset['train'].features[label_field], 'names') else list(set(l for sublist in labels_list for l in (sublist if is_multi_label else [sublist])))
    num_labels = len(label_names)
    gt_concepts = []
    for labels in labels_list:
        gt_concepts.append([label_names[l] for l in (labels if is_multi_label else [labels])])
    gt_emb = [embedder.encode(concepts) for concepts in gt_concepts]
    seed_concepts = [(l, l) for l in label_names]
    aske = ASKE()
    chunk_to_doc = get_chunk_to_doc(texts, aske.max_chunk_size)
    start_time = time.time()
    aske.prepare_data(texts, seed_concepts)
    phase_metrics = []
    phases = ['classification', 'enrichment', 'derivation']

    def compute_metrics():
        extracted_concepts = [set() for _ in texts]
        for cid, labels in aske.chunk_classifications.items():
            d = chunk_to_doc[cid]
            extracted_concepts[d].update(labels)
        sim_scores = []
        for d in range(len(texts)):
            ex_con = list(extracted_concepts[d])
            if not ex_con:
                continue
            ex_emb = embedder.encode(ex_con)
            gt_e = gt_emb[d]
            for g in gt_e:
                max_sim = max(util.cos_sim(g, e)[0][0].item() for e in ex_emb) if len(ex_emb) > 0 else 0
                sim_scores.append(max_sim)
        mean_sim = np.mean(sim_scores) if sim_scores else 0
        std_sim = np.std(sim_scores) if sim_scores else 0
        ci_sim = 1.96 * std_sim / np.sqrt(len(sim_scores)) if sim_scores else 0
        y_true = np.zeros((len(texts), num_labels), dtype=int)
        for d, labels in enumerate(labels_list):
            if is_multi_label:
                for l in labels:
                    y_true[d, l] = 1
            else:
                y_true[d, labels] = 1
        y_pred = np.zeros((len(texts), num_labels), dtype=int)
        label_to_idx = {l: i for i, l in enumerate(label_names)}
        for d, cons in enumerate(extracted_concepts):
            for c in cons:
                parts = c.split('_')
                for part in parts:
                    if part in label_to_idx:
                        y_pred[d, label_to_idx[part]] = 1
        acc, f1, prec, rec = calculate_classification_metrics(y_true, y_pred, is_multi_label)
        return mean_sim, std_sim, ci_sim, acc, f1, prec, rec

    aske.chunk_classification()
    mean_sim, std_sim, ci_sim, acc, f1, prec, rec = compute_metrics()
    phase_metrics.append({'phase': phases[0], 'mean_sim': mean_sim, 'std_sim': std_sim, 'ci_sim': ci_sim, 'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec})

    aske.terminological_enrichment()
    aske.chunk_classification()
    mean_sim, std_sim, ci_sim, acc, f1, prec, rec = compute_metrics()
    phase_metrics.append({'phase': phases[1], 'mean_sim': mean_sim, 'std_sim': std_sim, 'ci_sim': ci_sim, 'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec})

    aske.concept_derivation()
    aske.chunk_classification()
    mean_sim, std_sim, ci_sim, acc, f1, prec, rec = compute_metrics()
    phase_metrics.append({'phase': phases[2], 'mean_sim': mean_sim, 'std_sim': std_sim, 'ci_sim': ci_sim, 'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec})

    time_taken = time.time() - start_time
    pairwise_sim = []
    for label, data in aske.acg.items():
        term_vec = [tv for _, _, tv in data['terms']]
        if len(term_vec) > 1:
            for j in range(len(term_vec)):
                for k in range(j + 1, len(term_vec)):
                    pairwise_sim.append(util.cos_sim(term_vec[j], term_vec[k])[0][0].item())
    mean_pairwise = np.mean(pairwise_sim) if pairwise_sim else 0
    final = phase_metrics[-1].copy()
    final['mean_pairwise'] = mean_pairwise
    final['time'] = time_taken
    return {'phase_metrics': phase_metrics, 'final': final}