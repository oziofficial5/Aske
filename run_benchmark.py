import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import torch
from src.utils import load_previous_results, save_results
from src.benchmark import benchmark_model, evaluate_aske

# Disable W&B to avoid login prompt
import os
os.environ['WANDB_MODE'] = 'disabled'

def main():
    datasets_list = ['ecthr_a', 'ledgar', 'unfair_tos', 'scotus']
    models = {
        'LegalBERT': 'nlpaueb/legal-bert-base-uncased',
        'RoBERTa': 'roberta-base',
        'SBERT': 'sentence-transformers/all-mpnet-base-v2',
        'Longformer': 'allenai/longformer-base-4096'
    }
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Load previous results
    benchmark_results = load_previous_results()

    # Benchmark all models and ASKE
    for ds in datasets_list:
        if ds not in benchmark_results:
            benchmark_results[ds] = {}
        for m_name, m_path in models.items():
            print(f"Benchmarking {m_name} on {ds}")
            benchmark_results[ds][m_name] = benchmark_model(m_path, ds, embedder)
            save_results(benchmark_results)
            torch.cuda.empty_cache()
        print(f"Benchmarking ASKE on {ds}")
        benchmark_results[ds]['ASKE'] = evaluate_aske(ds)
        save_results(benchmark_results)
        torch.cuda.empty_cache()

    # Create summary table
    metrics = ['mean_sim', 'ci_sim', 'mean_pairwise', 'acc', 'f1', 'prec', 'rec', 'time']
    model_list = list(models.keys()) + ['ASKE']
    df_summary = pd.DataFrame(index=model_list, columns=[f"{ds}_{met}" for ds in datasets_list for met in metrics])
    for ds in datasets_list:
        for m in model_list:
            res = benchmark_results[ds][m]['final'] if m == 'ASKE' else benchmark_results[ds][m]
            for met in metrics:
                df_summary.loc[m, f"{ds}_{met}"] = res.get(met, np.nan)
    df_summary.to_json('benchmark_summary.json')

    # ASKE phase-wise tables
    for ds in datasets_list:
        phase_df = pd.DataFrame(benchmark_results[ds]['ASKE']['phase_metrics'])
        phase_df.to_json(f"{ds}_aske_phases.json")

    # Graphs
    for met in ['acc', 'f1', 'prec', 'rec']:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = pd.DataFrame({ds: df_summary[f"{ds}_{met}"] for ds in datasets_list})
        data.plot(kind='bar', ax=ax)
        plt.title(f"{met.capitalize()} Comparison Across Datasets")
        plt.ylabel(met.capitalize())
        plt.xlabel('Models')
        plt.xticks(rotation=45)
        plt.savefig(f"{met}_bar.png")
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sim_data = pd.DataFrame({m: [df_summary.loc[m, f"{ds}_mean_sim"] for ds in datasets_list] for m in model_list}, index=datasets_list)
    sim_data.plot(kind='line', ax=ax, marker='o')
    plt.title("Mean Similarity Scores Across Datasets")
    plt.ylabel("Mean Similarity")
    plt.xlabel("Datasets")
    plt.savefig("similarity_line.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    pair_data = pd.DataFrame({m: [df_summary.loc[m, f"{ds}_mean_pairwise"] for ds in datasets_list] for m in model_list}, index=datasets_list)
    pair_data.plot(kind='line', ax=ax, marker='o')
    plt.title("Mean Pairwise Similarity Across Datasets")
    plt.ylabel("Mean Pairwise Similarity")
    plt.xlabel("Datasets")
    plt.savefig("pairwise_line.png")
    plt.close()

    for ds in datasets_list:
        phase_df = pd.DataFrame(benchmark_results[ds]['ASKE']['phase_metrics'])
        fig, ax = plt.subplots(figsize=(10, 6))
        phase_df.set_index('phase')[['acc', 'f1', 'prec', 'rec']].plot(kind='line', ax=ax, marker='o')
        plt.title(f"ASKE Phase-Wise Improvement on {ds}")
        plt.ylabel("Metric Score")
        plt.xlabel("Phase")
        plt.savefig(f"aske_phase_improvement_{ds}.png")
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    time_data = pd.DataFrame({ds: df_summary[f"{ds}_time"] for ds in datasets_list})
    time_data.plot(kind='bar', ax=ax)
    plt.title("Time Taken for Evaluation Across Datasets")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.savefig("computational_efficiency_bar.png")
    plt.close()

    heatmap_metrics = ['acc', 'f1', 'mean_sim']
    heatmap_columns = [f"{ds}_{met}" for ds in datasets_list for met in heatmap_metrics]
    heatmap_data = df_summary[heatmap_columns].astype(float)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Heatmap of Model Performance Across Datasets and Metrics")
    plt.savefig("performance_heatmap.png")
    plt.close()

    print("Benchmarking completed. Results saved to JSON files, tables, and graphs.")

if __name__ == "__main__":
    main()