from src.models.retriever import RetrieverModel
from src.covering.covering import CoverAlgorithm
from src import data_utils

concept_portion_to_train = 0.5
data_portion = 1.0
dataset_name = "msmarco"
data_split = "train-concepts"

class RobustnessEvaluator:
    def __init__(self, model_hf_name, concept_to_attack):
        with open(f"config/cover_alg/concept-{concept_to_attack}.yaml", "r") as f:
            import yaml
            concept_config = yaml.safe_load(f)
            concept_qids = concept_config['concept_qids']  # fetched from the attack config

        heldin_concept_qids, heldout_concept_qids = (concept_qids[:int(len(concept_qids)*concept_portion_to_train)],
                                                    concept_qids[int(len(concept_qids)*concept_portion_to_train):])

        # Load dataset:
        corpus, queries, qrels, qp_pairs_dataset = data_utils.load_dataset(
            dataset_name=dataset_name,
            data_split=data_split,
            data_portion=data_portion,
            embedder_model_name=model_hf_name,
            filter_in_qids=None if concept_to_attack is None else concept_qids,
        )

        self.model_hf_name = model_hf_name
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.qp_pairs_dataset = qp_pairs_dataset
        self.heldin_concept_qids = heldin_concept_qids
        self.heldout_concept_qids = heldout_concept_qids

    def evaluate(self, model, max_batch_size=256):
        retriever_model = RetrieverModel(
            model_hf_name=self.model_hf_name,
            sim_func_name="cos_sim",
            max_batch_size=max_batch_size,
            model=model)
        emb_targets = retriever_model.embed(
                texts=[self.queries[qid] for qid in self.heldin_concept_qids]  # held-in concept queries
            ).mean(dim=0).unsqueeze(0).cuda()
            
        cover_algo = CoverAlgorithm(
        model_hf_name=self.model_hf_name,
        sim_func='cos_sim',
        model_local_name=None,
        # batch_size=batch_size,
        dataset_name=dataset_name,
        covering_algo_name="kmeans",
        data_portion=1.0,
        data_split=data_split,
        n_clusters=1,
        corpus=self.corpus, 
        queries=self.queries, 
        qrels=self.qrels, 
        qp_pairs_dataset=self.qp_pairs_dataset)
        results = cover_algo.evaluate_retrieval(
        data_split_to_eval=data_split,
        data_portion_to_eval=1.0,
        filter_in_qids_to_eval=self.heldout_concept_qids,  # held-out concept queries
        centroid_vecs=emb_targets,
        skip_existing=False)

        return results["adv_appeared@10"], results["adv_scores_mean"]
