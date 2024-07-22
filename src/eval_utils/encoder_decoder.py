from src.eval_utils.metrics import ContextAnswerLogProb
from tqdm import tqdm
import torch


def get_token_ids(self, batch):
    batch_token_ids_and_mask = self.tokenizer([question for question, _ in batch],
                                            return_tensors="pt", padding="longest").to(self.device)

    # Find position of the masked_token_id
    mask_token_flag = \
        (batch_token_ids_and_mask["input_ids"] == self.tokenizer.mask_token_id).float()         # batch x max_length
    
    # Check for entries with exactly one mask token
    valid_mask_entries = (mask_token_flag.sum(1) == 1.0)
    if not valid_mask_entries.all().item():
        # Filter out invalid entries
        batch = [batch[i] for i in range(len(batch)) if valid_mask_entries[i].item()]
        batch_token_ids_and_mask = self.tokenizer(
            [question for question, _ in batch],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).to(self.device)
        mask_token_flag = (batch_token_ids_and_mask["input_ids"] == self.tokenizer.mask_token_id).float()


    assert (mask_token_flag.sum(1) == 1.0).all().item()
    mask_token_ids = mask_token_flag.argmax(dim=1)                                         # batch

    # Prepare gold answers
    gold_answers = [gold_answer if gold_answer.startswith(" ") else f" {gold_answer}" for _, gold_answer in batch]

    batch_gold_answer_token_ids = []
    for gold_answer in gold_answers:
        gold_answer_token_ids = self.tokenizer(gold_answer)["input_ids"]
        if not (len(gold_answer_token_ids) == 3 and
                gold_answer_token_ids[0] == 0 and
                gold_answer_token_ids[2] == 2):
            raise AssertionError(f"Gold answer {gold_answer} has tokens {gold_answer_token_ids}")
        batch_gold_answer_token_ids.append(gold_answer_token_ids[1])

    batch_gold_answer_token_ids = torch.LongTensor(batch_gold_answer_token_ids).unsqueeze(1).to(self.device)  # batch x 1

    return mask_token_ids, batch_gold_answer_token_ids, batch_token_ids_and_mask, gold_answers

def eval_dataset(self):

    for i in tqdm(range(0, self.dataset_size, self.args.batch_size)):

        # Prepare questions
        my_batch_size = min(self.args.batch_size, self.dataset_size - i)
        batch = self.dataset[i: i + my_batch_size]

        batch_token_ids_and_mask = self.tokenizer([question for question, _ in batch],
                                                return_tensors="pt", padding="longest").to(self.device)

        # Find position of the masked_token_id
        mask_token_flag = \
            (batch_token_ids_and_mask["input_ids"] == self.tokenizer.mask_token_id).float()         # batch x max_length
        
        # Check for entries with exactly one mask token
        valid_mask_entries = (mask_token_flag.sum(1) == 1.0)
        if not valid_mask_entries.all().item():
            # Filter out invalid entries
            batch = [batch[i] for i in range(len(batch)) if valid_mask_entries[i].item()]
            batch_token_ids_and_mask = self.tokenizer(
                [question for question, _ in batch],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.tokenizer.model_max_length
            ).to(self.device)
            mask_token_flag = (batch_token_ids_and_mask["input_ids"] == self.tokenizer.mask_token_id).float()


        assert (mask_token_flag.sum(1) == 1.0).all().item()
        mask_token_ids = mask_token_flag.argmax(dim=1)                                         # batch

        # Prepare gold answers
        gold_answers = [gold_answer if gold_answer.startswith(" ") else f" {gold_answer}" for _, gold_answer in batch]

        batch_gold_answer_token_ids = []
        for gold_answer in gold_answers:
            gold_answer_token_ids = self.tokenizer(gold_answer)["input_ids"]
            if not (len(gold_answer_token_ids) == 3 and
                    gold_answer_token_ids[0] == 0 and
                    gold_answer_token_ids[2] == 2):
                raise AssertionError(f"Gold answer {gold_answer} has tokens {gold_answer_token_ids}")
            batch_gold_answer_token_ids.append(gold_answer_token_ids[1])

        batch_gold_answer_token_ids = torch.LongTensor(batch_gold_answer_token_ids).unsqueeze(1).to(self.device)  # batch x 1

        # Generate log probabilities over masked tokens, 1 per data point
        with torch.no_grad():
            logits = self.model(**batch_token_ids_and_mask).logits       # batch x max_length x vocab
            logprob = torch.log_softmax(logits, dim=2)                   # batch x max_length x vocab

        vocab_size = logprob.shape[2]
        mask_token_ids = mask_token_ids.view(my_batch_size, 1, 1)
        mask_token_ids = mask_token_ids.expand([my_batch_size, 1, vocab_size])

        predicted_logprob = torch.gather(logprob, index=mask_token_ids, dim=1)     # batch size x 1 x vocab_size
        predicted_logprob = predicted_logprob[:, 0, :]                             # batch x vocab_size

        # Generate top-k tokens
        sorted_logprob, sorted_indices = torch.sort(predicted_logprob, descending=True)    # both are batch x vocab_size
        sorted_logprob = sorted_logprob[:, :self.args.k].detach().cpu().numpy()                    # batch x k
        sorted_indices = sorted_indices[:, :self.args.k].detach().cpu().numpy()                    # batch x k

        # Compute top-k accuracy
        batch_top_10_tokens = [
            [self.tokenizer.decode(sorted_indices[j, l]).lower().strip() for l in range(10)]
            for j in range(my_batch_size)
        ]

        batch_top_1_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:1]
                                for j in range(my_batch_size)]
        batch_top_5_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:5]
                                for j in range(my_batch_size)]
        batch_top_10_accuracy = [gold_answers[j].lower().strip() in batch_top_10_tokens[j][:10]
                                    for j in range(my_batch_size)]

        # Compute log_prob using the probability of gold tokens
        gold_log_prob = torch.gather(predicted_logprob, index=batch_gold_answer_token_ids, dim=1)[:, 0]   # batch

        # Compute perplexity
        for j in range(my_batch_size):

            # Update the accuracy metric
            answer_log_prob = gold_log_prob[j].item()
            answer_len = 1
            logprob_results = ContextAnswerLogProb(total_log_prob=None,
                                                    answer_log_prob=answer_log_prob,
                                                    answer_len=answer_len)

            self.dataset_metric.accept(is_correct=batch_top_1_accuracy[j],
                                        f1pr_score=None,
                                        log_prob_results=logprob_results,
                                        top_k_acc={1: batch_top_1_accuracy[j],
                                                    5: batch_top_5_accuracy[j],
                                                    10: batch_top_10_accuracy[j]})
