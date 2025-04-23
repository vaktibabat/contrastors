#from robustness import RobustnessEvaluator
from isotropy import IsotropyEvaluator
#from performance import PerformanceEvaluator

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

def iso_eval_wrapper(model, config):
    global full_corpus, full_queries, full_qrels, full_qp_pairs_dataset 

    concept_corpus, concept_queries, concept_qrels, concept_qp_pairs_dataset = load_data(config.concept, config.model_hf_name) 

    iso_eval = IsotropyEvaluator(model, full_qp_pairs_dataset, full_corpus, full_queries, full_qrels, {})
    iso_eval_concept_specific = IsotropyEvaluator(model, concept_qp_pairs_dataset, concept_corpus, concept_queries, concept_qrels, {})

    # hopefully i'll be able to use higher n_evals on the cluster
    return iso_eval.gaslite_cosreg(n_evals=1500), iso_eval.iso_score_star(batch_size=100, n_evals=200), iso_eval_concept_specific.iso_score_star()