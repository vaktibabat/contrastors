from sentence_transformers.evaluation import NanoBEIREvaluator

# Each model has different query prefixes for evaluating NanoBEIR
query_prefixes = {
    "intfloat/e5-base-v2": "query: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
    "Snowflake/snowflake-arctic-embed-m": "Represent this sentence for searching relevant passages: ",
    "nomic-ai/nomic-embed-text-v1": "search_query: "
}
passage_prefixes = {
    "intfloat/e5-base-v2": "passage: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
    "Snowflake/snowflake-arctic-embed-m": "",
    "nomic-ai/nomic-embed-text-v1": "serach_passage: ",
}

class PerformanceEvaluator:
    def __init__(self, model_hf_name):
        datasets = ["QuoraRetrieval", "MSMARCO"]
        query_prompts = {
            "QuoraRetrieval": query_prefixes[model_hf_name],
            "MSMARCO": query_prefixes[model_hf_name]
        }
        self.evaluator = NanoBEIREvaluator(
            dataset_names=datasets,
            query_prompts=query_prompts
        )
    
    def evaluate(self, model):
        return self.evaluator(model)["NanoBEIR_mean_cosine_ndcg@10"]