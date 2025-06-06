import json
import os
import random
import sys
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from torch.nn.utils import clip_grad_norm_
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from contrastors.dataset.text_text_loader import StreamingShardDataset
from contrastors.distributed import DistributedWandbTracker, gather, print_rank_zero
from contrastors.optimizer import configure_optimizer

from IsoScore.IsoScore import *

from src.benchmarking.robustness import RobustnessEvaluator
from src.benchmarking.performance import PerformanceEvaluator
from src.isotropy import MultiConceptIsotropyEvaluator

from sentence_transformers import SentenceTransformer, models
from sentence_transformers.models import Normalize

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, config, dtype=torch.float32):
        self.config = config
        self.distributed = dist.is_initialized()
        self.print = print_rank_zero if self.distributed else print
        self.dtype = dtype

        self.profile = config.train_args.profile
        if self.profile:
            # create trace folder
            Path("trace").mkdir(parents=True, exist_ok=True)

        seed = config.data_args.seed
        self.set_seed(seed)

        if config.train_args.wandb:
            self.tracker = self.get_trackers(config)
        else:
            self.tracker = None

        self.print(json.dumps(config.dict(), indent=3))

        self.print(f"Using dtype: {dtype}")
        self.print(f"Using {dist.get_world_size() if dist.is_initialized() else 1} GPUs")

        self.deepspeed = config.deepspeed
        if self.deepspeed:
            ds_config = json.load(open(config.deepspeed_config))
        else:
            ds_config = {}

        self.tokenizer = self.get_tokenizer(config)

        self.model_type = config.model_args.model_type
        self.model = self.get_model(config)
        self.print(f"Model: {self.model}")

        all_models = [model for model in self.model.values()]
        num_params = 0
        for model in all_models:
            if isinstance(model, nn.Module):
                num_params += sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.print(f"Trainable parameters: {num_params:,}")
        if getattr(self.config.model_args, "num_experts", 0) > 0:
            num_experts = self.config.model_args.num_experts
            top_k = self.config.model_args.moe_top_k
            model = self.model["model"]
            if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                model = model.module
            num_layers = model.trunk.config.n_layer // self.config.model_args.moe_every_n_layers
            
            active_params = num_params

            params_per_expert = sum(p.numel() for p in model.trunk.encoder.layers[1].mlp.experts.mlp.parameters() if p.requires_grad) / num_experts
            active_params = active_params - num_layers * (num_experts - top_k) * params_per_expert
            self.print(f"Active parameters: {active_params:,}")

        self.dataloaders = self.get_dataloaders(config)
        self.optimizer = self.get_optimizer(config.train_args, ds_config)
        self.scheduler = self.get_scheduler(config.train_args, self.optimizer, ds_config)

        if self.deepspeed:
            self.print(f"Setting up deepspeed...")
            model, optimizer, dataloader, lr_scheduler = self.initialize_deepspeed(
                rank=dist.get_rank(), ds_config=ds_config
            )

            self.engine = model
            self.model = {"model": model, **{k: v for k, v in self.model.items() if k != "model"}}
            if optimizer is not None:
                self.optimizer = optimizer
            if dataloader is not None:
                self.dataloaders["train"] = dataloader
            if lr_scheduler is not None:
                self.scheduler = lr_scheduler

        self.reg = istar()
        self.concepts_to_eval = ["potter", "iphone", "vaccine"]
        self.robustness_evals = {k: RobustnessEvaluator("nomic-ai/nomic-embed-text-v1", k) for k in self.concepts_to_eval}
        self.perf_eval = PerformanceEvaluator("nomic-ai/nomic-embed-text-v1",)
        self.iso_eval = MultiConceptIsotropyEvaluator(self.concepts_to_eval, "nomic-ai/nomic-embed-text-v1")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @property
    def num_processes(self):
        return dist.get_world_size() if self.distributed else 1

    @property
    def process_index(self):
        return int(os.environ.get("LOCAL_RANK", -1)) if self.distributed else 0

    @property
    def global_rank(self):
        return int(os.environ.get("RANK", -1)) if self.distributed else 0

    @contextmanager
    def _goes_first(self, is_main):
        if not is_main:
            dist.barrier()

        yield

        if is_main:
            dist.barrier()

    @contextmanager
    def main_process_first(self):
        with self._goes_first(self.process_index == 0):
            yield

    def log(self, metrics, step=None):
        if self.global_rank == 0:
            self.tracker.log(metrics, step=step)

    def initialize_deepspeed(self, rank, ds_config):
        # don't let deepspeed print to stdout
        ds_config["steps_per_print"] = float("inf")

        if ds_config["gradient_accumulation_steps"] == "auto":
            ds_config["gradient_accumulation_steps"] = self.config.train_args.gradient_accumulation_steps

        # todo this could use some love
        if ds_config["train_micro_batch_size_per_gpu"] == "auto":
            data_args = self.config.data_args
            ds_config["train_micro_batch_size_per_gpu"] = data_args.batch_size // self.num_processes
            ds_config["train_batch_size"] = data_args.batch_size * ds_config["gradient_accumulation_steps"]

        if self.config.train_args.max_grad_norm != ds_config["gradient_clipping"]:
            ds_config["gradient_clipping"] = self.config.train_args.max_grad_norm

        return deepspeed.initialize(
            rank, model=self.model["model"], config=ds_config, optimizer=self.optimizer, lr_scheduler=self.scheduler
        )

    def get_trackers(self, config):
        tracker = None
        run_name = config.train_args.wandb_run_name
        if run_name is None:
            run_name = config.train_args.output_dir.replace("ckpts/", "")
            config.train_args.wandb_run_name = run_name
        if self.global_rank == 0:
            project_name = config.train_args.wandb_project_name
            entity = config.train_args.wandb_entity
            group = config.train_args.wandb_group

            hyperparams = {}
            for key, params in config.dict().items():
                if params is None or not isinstance(params, dict):
                    continue
                for k, v in params.items():
                    hyperparams[f"{key}_{k}"] = v

            tracker = wandb.init(project=project_name, entity=entity, name=run_name, config=hyperparams, group=group)

            if self.num_processes > 1:
                tracker = DistributedWandbTracker(tracker)

        return tracker

    def get_tokenizer(self, config):
        config = config.model_args
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        tokenizer.model_max_length = config.seq_len

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.cls_token is None:
            tokenizer.add_special_tokens({"cls_token": "<s>"})

        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<mask>"})

        return tokenizer

    @abstractmethod
    def get_model(self, config):
        pass

    @abstractmethod
    def get_dataloaders(self, config, epoch=0):
        pass

    def get_optimizer(self, config, ds_config=None):
        if ds_config:
            if "optimizer" in ds_config:
                optimizer = ds_config["optimizer"]
                optimizer["params"]["lr"] = config.learning_rate
                optimizer["params"]["weight_decay"] = config.weight_decay
                optimizer["params"]["betas"] = [config.adam_beta1, config.adam_beta2]
                return None

        models = [
            model
            for model in self.model.values()
            if isinstance(model, nn.Module) and any(p.requires_grad for p in model.parameters())
        ]
        optimizer = configure_optimizer(models, config)

        return optimizer

    def get_scheduler(self, config, optimizer, ds_config):
        if hasattr(config, "warmup_steps") and getattr(config, "warmup_steps") is not None:
            total_num_steps = self.total_num_steps * config.num_epochs
            warmup_steps = config.warmup_steps

        elif hasattr(config, "warmup_pct") and getattr(config, "warmup_pct") is not None:
            total_num_steps = self.total_num_steps * config.num_epochs
            warmup_steps = int(total_num_steps * config.warmup_pct)

        else:
            warmup_steps = 0

        self.print("*" * 50 + " SCHEDULER " + "*" * 50)
        self.print(f"Using {config.schedule_type} learning rate schedule")
        self.print(f"Warmup steps: {warmup_steps}")
        self.print(f"Total num steps: {total_num_steps}")

        if ds_config:
            if "scheduler" in ds_config:
                scheduler = ds_config["scheduler"]
                if scheduler["type"] == "WarmupDecayLR":
                    scheduler["params"]["warmup_min_lr"] = 0.0
                    scheduler["params"]["warmup_max_lr"] = config.learning_rate
                elif scheduler["type"] == "WarmupCosineLR":
                    scheduler["params"]["warmup_min_ratio"] = 0.0

                scheduler["params"]["warmup_num_steps"] = warmup_steps
                scheduler["params"]["total_num_steps"] = total_num_steps
                return None

        scheduler = get_scheduler(
            name=config.schedule_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(total_num_steps if config.schedule_type != "inverse_sqrt" else None),
        )

        return scheduler

    def unwrap(self, model):
        # TODO: add deepspeed unwrapping support
        model_types = (nn.parallel.DistributedDataParallel, nn.DataParallel, deepspeed.DeepSpeedEngine)
        while isinstance(model, model_types):
            model = model.module

        return model

    def save_model(self, output_dir):
        if self.global_rank == 0:
            unwrapped = self.unwrap(self.model["model"])
            if self.deepspeed:
                # TODO: need to add zero3 support
                # reduces bloat when saving with deepsped
                state_dict = clone_tensors_for_torch_save(unwrapped.state_dict())
            else:
                state_dict = None

            unwrapped.save_pretrained(output_dir, state_dict=state_dict)

    def load_model(self, model_path):
        loaded_model = self.model["model"].load_pretrained(model_path)

        return loaded_model

    def load_state(self, input_dir):
        if self.deepspeed:
            self.engine.load_checkpoint(input_dir)

        else:
            self.print(f"Loading model from {input_dir}/model")

            self.model["model"] = self.load_model(f"{input_dir}/model")

            self.print(f"Loading optimizer and scheduler state from {input_dir}/optimizer.pt")
            self.optimizer.load_state_dict(torch.load(f"{input_dir}/optimizer.pt"))

            self.print(f"Loading optimizer and scheduler state from {input_dir}/scheduler.pt")
            self.scheduler.load_state_dict(torch.load(f"{input_dir}/scheduler.pt"))

        self.print(f"Loading random states from {input_dir}/random_states_{self.process_index}.pt")
        random_states = torch.load(f"{input_dir}/random_states_{self.process_index}.pt")

        torch.set_rng_state(random_states["torch"])
        np.random.set_state(random_states["numpy"])
        random.setstate(random_states["random"])
        torch.cuda.set_rng_state_all(random_states["cuda"])

    def save_state(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.save_model(f"{output_dir}/model")

        if self.deepspeed:
            self.engine.save_checkpoint(output_dir)

        else:
            opt_state_dict = self.optimizer.state_dict()
            torch.save(opt_state_dict, f"{output_dir}/optimizer.pt")

            schedulr_state_dict = self.scheduler.state_dict()
            torch.save(schedulr_state_dict, f"{output_dir}/scheduler.pt")

            if isinstance(self.dataloaders["train"], StreamingShardDataset):
                data_config = self.config.data_args
                ds_state = data_config.input_shards.replace(".yaml", "")
                with open(f"{ds_state}/rank_{dist.get_rank()}_processed.json", "r") as f:
                    processed = json.load(f)

                with open(f"{output_dir}/rank_{dist.get_rank()}_processed.json", "w") as f:
                    json.dump(processed, f, indent=3)

        random_states = {}
        random_states["torch"] = torch.get_rng_state()
        random_states["numpy"] = np.random.get_state()
        random_states["random"] = random.getstate()
        random_states["cuda"] = torch.cuda.get_rng_state_all()

        torch.save(random_states, f"{output_dir}/random_states_{self.process_index}.pt")

    def backward(self, loss):
        if self.deepspeed:
            self.engine.backward(loss)
            self.engine.step()
        else:
            loss.backward()

    @abstractmethod
    def eval_loop(self, model, dataloader, step, **kwargs):
        pass

    @abstractmethod
    def forward_step(self, model, inputs, **kwargs):
        pass

    @abstractmethod
    def clip_gradients(self, max_grad_norm):
        clip_grad_norm_(self.model["model"].parameters(), max_grad_norm)

    @abstractmethod
    def training_step(

        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps, query_C0, doc_C0
    ):
        for param in model["model"].parameters():
            param.requires_grad = True

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(not self.deepspeed and not self.config.train_args.grad_cache)):
            loss = self.forward_step(inputs=batch, **model, step=step, query_C0=query_C0, doc_C0=doc_C0)
        self, model, batch, optimizer, scheduler, step, train_args, total_num_steps, gradient_accumulation_steps
    ):
        with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=(not self.deepspeed and not self.config.train_args.grad_cache)):
            loss = self.forward_step(inputs=batch, **model, step=step)

        self.backward(loss)

        # clip gradients
        if step % gradient_accumulation_steps == 0:
            if train_args.max_grad_norm is not None and train_args.max_grad_norm > 0:
                if not self.deepspeed:
                    self.clip_gradients(train_args.max_grad_norm)

        # all of this is handled by the deepspeed engine
        if not self.deepspeed:
            if (step + 1) % gradient_accumulation_steps == 0 or step == total_num_steps - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if self.model.get("ema", None) is not None:
            # TODO; add warmup + weighting
            ema_getter = self.model["ema_gettr"]
            model_to_update = ema_getter(model)
            self.model["ema"].update(model_to_update)

        # Log every 25 steps
        if step % 25 == 0:
            bi_encoder = model["model"].module

            # Build SentenceTransformer
            sentence_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
            sentence_model[0].auto_model = bi_encoder.trunk
            
            iso_results = self.iso_eval.evaluate(model["model"], sentence_model)
            robustness_results = {concept: self.robustness_evals[concept].evaluate(sentence_model) for concept in self.concepts_to_eval}
            ndcg = self.perf_eval.evaluate(sentence_model)

            to_log = {"step": step, "ndcg@10": ndcg}

            for concept in self.concepts_to_eval:
                to_log[f"{concept}_appeared@10"] = robustness_results[concept][0]
                to_log[f"{concept}_mean_cos_sim"] = robustness_results[concept][1]

            to_log["general_cosreg"] = iso_results["general_cosreg"]
            to_log["general_isoscore"] = iso_results["general_isoscore"]

            for concept in self.concepts_to_eval: 
                to_log[f"{concept}_cosreg"] = iso_results[f"{concept}_cosreg"]
                to_log[f"{concept}_isoscore"] = iso_results[f"{concept}_isoscore"]

            wandb.log(to_log)
    
            sentence_model.train()
            model["model"].train()

        if step == 25 * 50:
            sys.exit(0)
            
        return loss

    def train(self):
        train_args = self.config.train_args
        data_config = self.config.data_args

        dataloaders = self.dataloaders
        train_dataloader = dataloaders["train"]
        val_dataloader = dataloaders.get("val", None)

        self.model["model"] = self.model["model"].to(torch.bfloat16)

        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler

        total_num_steps = int(self.total_num_steps)
        total_training_steps = getattr(self, "total_training_steps", total_num_steps)
        gradient_accumulation_steps = train_args.gradient_accumulation_steps

        if train_args.checkpoint:
            self.load_state(train_args.checkpoint)
            # checkpoint will be something like /root/contrastors-dev/src/contrastors/ckpts/unlit-search-query-no-resampled-dfn-2b-vitb-65k-3-epoch/epoch_0
            # split by / and get the last element, then split by _ and get the last element
            if "epoch_" in train_args.checkpoint:
                start_epoch = int(train_args.checkpoint.split("/")[-1].split("_")[-1])
                initial_step = 0
            else:
                initial_step = int(train_args.checkpoint.split("/")[-1].split("_")[-1])
                start_epoch = 0
        else:
            initial_step = 0
            start_epoch = 0

        self.print(f"Starting training from epoch {start_epoch}, step {initial_step=}")
        for epoch in range(start_epoch, train_args.num_epochs):

            if epoch > 0 and getattr(data_config, "streaming", False) is False:
                # webdataset needs special handling for multi-epoch
                if "train_sampler" in dataloaders:
                    sampler = dataloaders["train_sampler"]
                    sampler.set_epoch(epoch)

                elif getattr(train_dataloader, "sampler", None) is not None and isinstance(
                    train_dataloader.sampler, DistributedSampler
                ):
                    train_dataloader.sampler.set_epoch(epoch)

            elif epoch > 0 and getattr(data_config, "streaming", False):
                temp_config = self.config.copy(deep=True)
                temp_config.data_args.seed = data_config.seed + epoch
                train_dataloader = self.get_dataloaders(temp_config, epoch=epoch)["train"]

            self.print(f"Total training steps: {total_training_steps}; before computing shrinkage matrix")

            self.print("before computing shrinkage matrix")

            # Compute C0 matrix for I-STAR
            # (768, 768) matrices
            query_C0, doc_C0 = compute_shrinkage_matrix(train_dataloader, model["model"])
            self.print(f"Total training steps: {total_training_steps}")

            progbar = tqdm(
                train_dataloader, desc=f"Epoch {epoch}", disable=not self.global_rank == 0, total=total_training_steps, initial=initial_step
            )

            # TODO: fix resuming from a checkpoint
            if self.profile:
                context = partial(
                    profile,
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
                    on_trace_ready=tensorboard_trace_handler("trace"),
                    with_stack=True,
                    record_shapes=True,
                    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
                )
            else:
                context = nullcontext

            with context() as p:
                for step, batch in enumerate(progbar):
                    if step >= total_training_steps:
                        break

                    # if using deepspeed, it handles gradient accumulation
                    curr_step = epoch * total_training_steps + step + initial_step

                    loss = self.training_step(
                        model,
                        batch,
                        optimizer,
                        scheduler,
                        curr_step,
                        train_args,
                        total_training_steps,
                        gradient_accumulation_steps,
                        query_C0,
                        doc_C0,
                    )

                    if self.profile:
                        p.step()

                    if isinstance(loss, torch.Tensor):
                        loss = gather(loss.detach().float())
                        metrics = {"loss": loss}
                    elif isinstance(loss, dict):
                        metrics = {}
                        for k, v in loss.items():
                            metrics[k] = gather(v.detach().float())
                    else:
                        raise TypeError(f"Unexpected loss type: {type(loss)}")

                    if train_args.wandb:
                        self.log({k: torch.mean(v).item() for k, v in metrics.items()}, step=curr_step)
                    else:
                        self.print(f'Loss: { {k: torch.mean(v).item() for k, v in metrics.items()} }')

                        #self.print(f"IsoScore*: {metrics['avg_batch_iso']}")

                        # only print every gradient accumulation steps
                        if step % gradient_accumulation_steps == 0 and step > 0 or step == total_training_steps - 1:
                            self.print(f"LR: {scheduler.get_last_lr()[0]}")

                    if val_dataloader is not None:
                        if (
                            step > 0
                            and train_args.eval_strategy == "steps"
                            and train_args.eval_steps > 0
                            and (step / gradient_accumulation_steps) % train_args.eval_steps == 0
                        ):
                            self.eval_loop(dataloader=val_dataloader, step=curr_step, **model)

                    # log LR in case something weird happens
                    if step > 0 and step % (train_args.log_lr_every) == 0:
                        if train_args.wandb:
                            self.log({"lr": scheduler.get_last_lr()[0]}, step=curr_step)

                    if step > 0 and train_args.save_every > 0 and step % train_args.save_every == 0:
                        self.save_state(f"{train_args.output_dir}/step_{curr_step}")

                    if self.profile and step >= 10:
                        return

            if val_dataloader is not None and train_args.eval_strategy == "epochs":
                self.eval_loop(dataloader=val_dataloader, step=curr_step, **model)

            if train_args.save_every > 0:
                self.save_model(f"{train_args.output_dir}/epoch_{epoch}_model")
                if train_args.num_epochs > 1:
                    self.save_state(f"{train_args.output_dir}/epoch_{epoch}")

        if train_args.num_epochs > 1 and train_args.save_every > 0:
            torch.distributed.barrier()
            self.save_model(f"{train_args.output_dir}/final_model")

# Compute the shrinkage matrix $\Sigma_{S_i}$ at epoch i
# slightly modified version of get_ci from https://github.com/bcbi-edu/p_eickhoff_isoscore/blob/main/I-STAR/training_utils.py#L40
def compute_shrinkage_matrix(data, model, max_points=250000):
    """Given the data and model of interest, generate a sample of size max_points,
    then calculate the covariance matrix. Run this as a warmup to generate a stable
    covariance matrix for IsoScore Regularization"""
    num_points = 0
    # We compute C0 for both the queries and the documents
    query_point_list = []
    doc_point_list = []
    model.eval()
    h = 768

    for idx, batch in enumerate(data):
        # send batch to device
        batch = {key: value.to(device=model.device, dtype=torch.int64) for key, value in batch.items() if isinstance(value, torch.Tensor)}

        # Set model to eval and run input batches with no_grad to disable gradient calculations
        with torch.no_grad():        
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                query_outputs = model(input_ids=batch["query_input_ids"], attention_mask=batch["query_attention_mask"], normalize=False, output_hidden_states=True)["hidden_states"]
                doc_outputs = model(input_ids=batch["document_input_ids"], attention_mask=batch["document_attention_mask"], normalize=False, output_hidden_states=True)["hidden_states"]

            query_points = torch.reshape(query_outputs, (-1,h))
            doc_points = torch.reshape(doc_outputs, (-1,h))

        # We can track the # of points for both the queries and the documents with the same variable
        num_points += query_points.shape[0]

       # Collect the last state representations to a list and keep track of the number of points
        query_points = query_points.to(torch.float32).detach().cpu().numpy()
        doc_points = doc_points.to(torch.float32).detach().cpu().numpy()

        query_point_list.append(query_points)
        doc_point_list.append(doc_points)

        if num_points > max_points:
            break
    # Convert model back to train mode:
    model.train()
    # Stack the points and calclate the sample covariance C0
    query_sample, doc_sample = np.vstack(query_point_list), np.vstack(doc_point_list)
    query_C0, doc_C0 = np.cov(query_sample.T), np.cov(doc_sample.T)

    return torch.tensor(query_C0, device=model.device), torch.tensor(doc_C0, device=model.device)