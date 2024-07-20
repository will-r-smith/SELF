import torch
import importlib
import pandas as pd
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

from src.eval_utils.metrics import Metrics, DatasetMetrics, ContextAnswerLogProb

from src.matrix_utils import relative_error, do_lr, do_mm
from copy import deepcopy

import torch.nn as nn
import torch.optim as optim



class Experiment:


    def __init__(self, args, config):

        self.args = args

        self.config = config

        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics()

        # Device for the experiment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(self.device)

        self.load_model()

        self.norms = {}


    def intervene(self):
        "approximate a series of layers"

        print(f"{self.args.model} model loaded")

        norms = {}

        # base_name = config[args.model]["naming_conv"]["base"]
        layer_mappings = self.config[self.args.model]["naming_conv"]["layers"]

        if self.args.lname[0] == "all":
            layer_keys = layer_mappings.values()
        else:
            layer_keys = []
            for n in self.args.lname:
                layer_keys.append(layer_mappings[n])

        if self.args.lnum[0] == "all":
            layer_numbers = range(self.config[self.args.model]['num_layers'])
        else: 
            layer_numbers = [int(i) for i in self.args.lnum]

        # Collect names of parameters to modify
        params_to_modify = []

        for name, param in self.model.named_parameters():
            for key in layer_keys:
                if key in name:
                    for layer_num in layer_numbers:
                        if f".{layer_num}." in name:
                            params_to_modify.append((name, param))

        self.trainable_parameters = []

        # Perform modifications outside the iteration
        for name, param in params_to_modify:
            print(f"Approximating layer: .{layer_num}.{key}") 

            original_mat = param.detach().numpy().copy()
            original_mat_tensor = deepcopy(param)

            if self.args.intervention == "lr":
                self.model, approx_mat, parameters = do_lr(self.model, name, original_mat_tensor.type(torch.float32), (1 - self.args.rate))

            elif self.args.intervention == "mm":
                self.model, approx_mat = do_mm(original_mat)

            self.trainable_parameters.extend(parameters)

            norm = relative_error(original_mat_tensor.type(torch.float32), approx_mat)

            layer_num = name.split('.')[2]  # Extract layer number from the parameter name
            key = name.split('.')[-1]  # Extract key from the parameter name
            norms[f'layer_{layer_num}_{key}'] = norm

        path = f"outputs/results/approximate/{self.args.intervention}/{self.args.model}/norms.json"

        with open(path, 'w') as json_file:
            json.dump(norms, json_file, indent=4)

        self.model.to(self.device)

        self.norms = norms

        print(norms)



    def load_model(self):

        print(f"Loading pre-trained {self.args.model} model.")

        if self.args.model == "roberta":
            from transformers import RobertaForMaskedLM
            llm_name = "roberta-base"
            model = RobertaForMaskedLM.from_pretrained(llm_name, cache_dir='./cache')
            
        elif self.args.model == "pythia":
            from transformers import AutoModelForCausalLM
            llm_name = "EleutherAI/pythia-160m-deduped-v0"
            model = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir='./cache')
        
        elif self.args.model == "gptj":
            from transformers import GPTJForCausalLM
            llm_name = "EleutherAI/gpt-j-6B"
            model = GPTJForCausalLM.from_pretrained(
                llm_name,
                revision="float16",
                torch_dtype=torch.float16,
                cache_dir='./cache'
            ) 

        print("finished loading")
        self.model = model
        print("adding to self")
        self.llm_name = llm_name

        self.model.to(self.device)
        print("added to device")




    def load_dataset(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)

        if self.args.model in ["pythia", "gptj"]:
            # Add padding and mask token if they don't exist
            self.tokenizer.add_special_tokens({
                'pad_token': self.tokenizer.eos_token,
                'mask_token': '[MASK]'
            })
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Dynamically import the required data_utils function
        module = importlib.import_module(f"src.data_utils.{self.args.dataset}")
        load_dataset = getattr(module, 'load_dataset')

        # load dataset
        self.dataset = load_dataset(self)

        self.dataset_size = len(self.dataset)



    def evaluate(self):

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()

        self.load_dataset()

        if self.config[self.args.model]["type"] == "decoder":
            from src.eval_utils.decoder_only import eval_dataset
        if self.config[self.args.model]["type"] == "encoder_decoder":
            from src.eval_utils.encoder_decoder import eval_dataset
            
        eval_dataset(self)

        self.terminate_and_save()

        print(f"\nCorrectness:  {self.results["0-1 correctness"]}")



    def terminate_and_save(self):

        self.dataset_metric.terminate()

        results = self.dataset_metric.agg_to_dict()
        for k, v in self.args.__dict__.items():
            results["args/%s" % k] = v

        results["Frobenius_ratio"] = self.norms

        path = f"outputs/results/evaluate/{self.args.intervention}/{self.args.model}/{self.args.dataset}.csv"
        
        self.save_results(results, path)

        self.results = results



    def save_results(self, results, path):
            
        results_df = pd.DataFrame([results])

        # Check if the CSV file exists and is not empty
        if os.path.exists(path):
            try:
                existing_results_df = pd.read_csv(path)
                df = pd.concat([existing_results_df, results_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                df = results_df
        else:
            df = results_df

        df.to_csv(path, index=False)



    def fine_tune(self):     

        self.load_dataset()

        for name, param in self.model.named_parameters():
            print(name)

        print(self.trainable_parameters)

        # Only the parameters with 'U', 'S', or 'Vt' in their names will be trained
        trainable_params = [param for name, param in self.model.named_parameters() if name.split(".")[-1] in self.trainable_parameters]

        optimizer = optim.Adam(trainable_params, lr=self.args.learning_rate)
        loss_fn = nn.CrossEntropyLoss()


        self.model.train()
        for epoch in range(self.args.num_epochs):
            for i in tqdm(range(0, self.dataset_size, self.args.batch_size)):

                # Prepare questions
                my_batch_size = min(self.args.batch_size, self.dataset_size - i)
                batch = self.dataset[i: i + my_batch_size]

                mask_token_ids, batch_gold_answer_token_ids, batch_token_ids_and_mask, gold_answers = self.get_token_ids(batch)

                # Generate log probabilities over masked tokens, 1 per data point
                logits = self.model(**batch_token_ids_and_mask).logits

                # Ensure logits require gradient
                logits = logits.clone().detach().requires_grad_(True)

                # Use the mask_token_ids to gather the logits at the masked positions
                mask_token_ids = mask_token_ids.unsqueeze(1) 
                mask_token_logits = logits.gather(1, mask_token_ids.unsqueeze(2).expand(-1, -1, logits.size(-1)))[:, 0, :]

                optimizer.zero_grad()

                # Calculate loss using the logits for the masked positions
                loss = loss_fn(mask_token_logits, batch_gold_answer_token_ids.squeeze(1))
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{self.args.num_epochs}, Loss: {loss.item()}")



