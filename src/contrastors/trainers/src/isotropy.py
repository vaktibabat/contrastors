import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import numpy as np
from . import data_utils
from IsoScore.IsoScore import *

device = torch.device("cuda")

concept_portion_to_train = 0.5
dataset_name = "msmarco"
data_split = "train-concepts"
data_portion = 1.0

def load_data(concept_to_attack=None, model_hf_name=None):
    if concept_to_attack is not None:
        with open(f"config/cover_alg/concept-{concept_to_attack}.yaml", "r") as f:
            import yaml
            concept_config = yaml.safe_load(f)
            concept_qids = concept_config['concept_qids']  # fetched from the attack config

        heldin_concept_qids, heldout_concept_qids = (concept_qids[:int(len(concept_qids)*concept_portion_to_train)],
                                                    concept_qids[int(len(concept_qids)*concept_portion_to_train):])

    # Load dataset:
    return data_utils.load_dataset(
        dataset_name=dataset_name,
        data_split=data_split,
        data_portion=data_portion,
        embedder_model_name=model_hf_name,
        filter_in_qids=None if concept_to_attack is None else concept_qids,
    )

# Compute the shrinkage matrix $\Sigma_{S_i}$ at epoch i
# slightly modified version of get_ci from https://github.com/bcbi-edu/p_eickhoff_isoscore/blob/main/I-STAR/training_utils.py#L40
def compute_shrinkage_matrix(data, model, max_points=250000):
    """Given the data and model of interest, generate a sample of size max_points,
    then calculate the covariance matrix. Run this as a warmup to generate a stable
    covariance matrix for IsoScore Regularization"""
    num_points = 0
    points_list = []
    model.eval()
    h = model[0].auto_model.config.hidden_size

    for idx, batch in enumerate(data):
        # send batch to device
        #batch = {key: value.to(device) for key, value in batch.items()}

        # Set model to eval and run input batches with no_grad to disable gradient calculations
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = model[0].auto_model(input_ids=batch[0], attention_mask=batch[1], output_hidden_states=True)
           
           points = torch.reshape(outputs.hidden_states, (-1,h))

        num_points += points.shape[0]

        points = points.to(torch.float32).detach().cpu().numpy()

       # Collect the last state representations to a list and keep track of the number of points
        points_list.append(points)

        if num_points > max_points:
            break
    # Convert model back to train mode:
    model.train()
    # Stack the points and calclate the sample covariance C0
    sample = np.vstack(points_list)
    C0 = np.cov(sample.T)
    return torch.tensor(C0, device=device)

class IsotropyEvaluator:
    def __init__(self,
                 qp_pairs_dataset,
                 corpus, queries, qrels,
                 results: dict):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.results = results
        self.qp_pairs_dataset = qp_pairs_dataset
        self.sim_func = "cos_sim"

    def gaslite_cosreg(self, model, general=False, x='passage', y='passage', n_evals=1500, batch_size=100):
        """Calculate the average similarity (`sim_func`) of a random query to a random passage."""
        assert x in ['query', 'passage'] and y in ['query', 'passage']
        random.seed(100)
        
        if general:
            x_texts = [self.corpus[k]["text"] for k in random.sample(list(self.corpus.keys()), n_evals)]
            y_texts = [self.corpus[k]["text"] for k in random.sample(list(self.corpus.keys()), n_evals)]
        else:
            x_texts = self.qp_pairs_dataset[x].copy()[:n_evals]
            y_texts = self.qp_pairs_dataset[y].copy()[-n_evals:]
        n_evals = min(n_evals, len(x_texts), len(y_texts))
        lst_sim_to_rand = []

        for _ in range(0, n_evals, batch_size):
            x_batch = model.encode(random.choices(x_texts, k=batch_size), convert_to_tensor=True)
            y_batch = model.encode(random.choices(y_texts, k=batch_size), convert_to_tensor=True)
            # if self.sim_func == 'cos_sim':  # then normalize before dot product  [WE CURRENTLY EXAMINE COS-SIM FOR ALL]
            x_batch = F.normalize(x_batch, p=2, dim=-1)
            y_batch = F.normalize(y_batch, p=2, dim=-1)
            curr_sim = torch.matmul(x_batch, y_batch.T)  # calculate the (pairwise) similarity matrix

            # Discard diagonal and flatten
            curr_sim = curr_sim[~torch.eye(curr_sim.shape[0]).bool()].flatten()
            lst_sim_to_rand.extend(curr_sim.tolist())

        return sum(lst_sim_to_rand) / len(lst_sim_to_rand)

    # Evaluate IsoScore* on a given dataset
    def iso_score_star(self, model, general=False, x="passage", y="passage", n_evals=1500, batch_size=100, zeta=0.2):
        model.train()
        random.seed(100)
        if general:
            samples = [self.corpus[k]["text"] for k in random.sample(list(self.corpus.keys()), n_evals)]
        else:
            samples = self.qp_pairs_dataset[x].copy()[:n_evals]

        texts = model.tokenizer(samples, return_tensors="pt", padding=True, truncation=True)
        texts = TensorDataset(texts["input_ids"].to(device), texts["attention_mask"].to(device))
        dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True)
        h = model[0].auto_model.config.hidden_size 
        reg = istar()
        C0 = compute_shrinkage_matrix(dataloader, model)
        isos = torch.tensor(0, dtype=torch.float64,device=torch.device("cuda"))

        for idx, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = model[0].auto_model(input_ids=batch[0], attention_mask=batch[1], normalize=False, output_hidden_states=True)
                points = torch.reshape(outputs.hidden_states, (-1,h))
                batch_iso = reg.IsoScore_star(points, C0, zeta=zeta, gpu_id=0, is_eval=False)
                
                isos += batch_iso
        
        # Return the mean IsoScore* across all batches
        return isos / len(dataloader)
    
# An Isotropy evaluator that evaluates the concept-specific isotropy for multiple concepts
class MultiConceptIsotropyEvaluator:
    def __init__(self, concepts_to_eval, model_hf_name):
        self.concepts_to_eval = concepts_to_eval
        self.concept_evals = {}
        # Construct a general MSMARCO evaluator
        full_corpus, full_queries, full_qrels, full_qp_pairs_dataset = load_data(concept_to_attack=None, model_hf_name=model_hf_name) 
        self.general_eval = IsotropyEvaluator(full_qp_pairs_dataset, full_corpus, full_queries, full_qrels, {})
        # Construct an evaluator for each of the concepts
        for concept in concepts_to_eval:
            concept_corpus, concept_queries, concept_qrels, concept_qp_pairs_dataset = load_data(concept_to_attack=concept, model_hf_name=model_hf_name) 
            self.concept_evals[concept] = IsotropyEvaluator(concept_qp_pairs_dataset, concept_corpus, concept_queries, concept_qrels, {})
    
    def evaluate(self, model):
        results = {}
        # Evaluate using the general evaluator
        results["general_cosreg"] = self.general_eval.gaslite_cosreg(model, general=True)
        results["general_isoscore"] = self.general_eval.iso_score_star(model, general=True)
        #results["general_isoscore"] = None

        # Evaluate the concept-specific metrics
        for concept in self.concepts_to_eval:
            results[f"{concept}_cosreg"] = self.concept_evals[concept].gaslite_cosreg(model)
            results[f"{concept}_isoscore"] = self.concept_evals[concept].iso_score_star(model)
        
        return results
