import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AffinityPropagation

class ASKE:
    def __init__(self, embedding_model='paraphrase-multilingual-MiniLM-L12-v2', alpha=0.3, beta=0.3, max_chunk_size=512, termination_threshold=0):
        self.embedder = SentenceTransformer(embedding_model)
        self.alpha = alpha
        self.beta = beta
        self.max_chunk_size = max_chunk_size
        self.termination_threshold = termination_threshold
        self.acg = {}
        self.derivation_links = []
        self.chunk_classifications = {}
        self.chunks = []
        self.chunk_vectors = None

    def prepare_data(self, documents, seed_concepts):
        self.chunks = []
        chunk_id = 0
        for doc in documents:
            words = doc.split()
            for i in range(0, len(words), self.max_chunk_size):
                chunk_text = ' '.join(words[i:i + self.max_chunk_size])
                self.chunks.append((chunk_id, chunk_text))
                chunk_id += 1
        self.chunk_vectors = self.embedder.encode([text for _, text in self.chunks])
        for label, desc in seed_concepts:
            terms = desc.split()
            term_vectors = self.embedder.encode(terms)
            concept_vector = np.mean(term_vectors, axis=0)
            self.acg[label] = {'vector': concept_vector, 'terms': list(zip(terms, ['' for _ in terms], term_vectors))}

    def chunk_classification(self):
        self.chunk_classifications = {}
        for i, (chunk_id, _) in enumerate(self.chunks):
            classifications = []
            for label, data in self.acg.items():
                sim = util.cos_sim(self.chunk_vectors[i], data['vector'])[0][0].item()
                if sim >= self.alpha:
                    classifications.append(label)
            self.chunk_classifications[chunk_id] = classifications

    def terminological_enrichment(self):
        for label, data in self.acg.items():
            new_terms = []
            for chunk_id, labels in self.chunk_classifications.items():
                if label in labels:
                    chunk_text = self.chunks[chunk_id][1]
                    terms = chunk_text.split()
                    term_vectors = self.embedder.encode(terms)
                    for t, v in zip(terms, term_vectors):
                        sim = util.cos_sim(v, data['vector'])[0][0].item()
                        if sim >= self.beta:
                            new_terms.append((t, '', v))
            data['terms'] += new_terms
            if data['terms']:
                data['vector'] = np.mean([tv for _, _, tv in data['terms']], axis=0)

    def concept_derivation(self):
        new_concepts = []
        for label, data in list(self.acg.items()):
            if len(data['terms']) < 2:
                continue
            term_vectors = [tv for _, _, tv in data['terms']]
            term_vectors = np.array(term_vectors)
            if term_vectors.ndim == 1:
                term_vectors = term_vectors.reshape(1, -1)
            if term_vectors.shape[0] > 1:
                af = AffinityPropagation(damping=0.75, random_state=0).fit(term_vectors)
                cluster_labels = af.labels_
                cluster_centers_indices = af.cluster_centers_indices_
                unique_clusters = set(cluster_labels)
                for cl in unique_clusters:
                    cluster_indices = [j for j in range(len(cluster_labels)) if cluster_labels[j] == cl]
                    if len(cluster_indices) < 1:
                        continue
                    cluster_terms = [data['terms'][j] for j in cluster_indices]
                    cluster_vector = np.mean([tv for _, _, tv in cluster_terms], axis=0)
                    distances = [util.cos_sim(cluster_vector, tv)[0][0].item() for _, _, tv in cluster_terms]
                    closest_idx = np.argmax(distances)
                    new_label = cluster_terms[closest_idx][0]
                    new_label = f"{label}_{new_label}"
                    new_concepts.append((new_label, cluster_vector, cluster_terms))
                    self.derivation_links.append((new_label, label))
        for nl, nv, nt in new_concepts:
            self.acg[nl] = {'vector': nv, 'terms': nt}
        return len(new_concepts)

    def run(self, documents, seed_concepts, max_generations=20):
        self.prepare_data(documents, seed_concepts)
        generation = 0
        while generation < max_generations:
            self.chunk_classification()
            self.terminological_enrichment()
            new_count = self.concept_derivation()
            if new_count <= self.termination_threshold:
                break
            generation += 1
        return generation