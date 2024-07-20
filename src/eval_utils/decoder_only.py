from src.eval_utils.metrics import ContextAnswerLogProb
from tqdm import tqdm
import torch


def eval_dataset(self):

    for i in tqdm(range(0, self.dataset_size, self.dataset_size)):


        question, answer = self.dataset[i]
        # Given that we do 1-token look up we do the following:
        # - Compute log-prob of the gold token
        # - Compute top-1, top-5 and top-10 accuracies

        print(question)

        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        gold_answer_token_ids = self.tokenizer(answer)["input_ids"]
        answer_len = len(gold_answer_token_ids)
        assert answer_len == 1, f"For GPTJ+CounterFact special case, we assume the answer " \
                                f"has 1 token. Found {gold_answer_token_ids}."
        gold_answer_token_id = int(gold_answer_token_ids[0])

        with torch.no_grad():
            # Compute log probability of question
            results = self.model(inputs.input_ids)
            logits = results.logits[0]                                      # question length x vocab
            log_prob = torch.nn.functional.log_softmax(logits, dim=1)       # question length x vocab

            last_token_logprob = log_prob[-1]                               # vocab
            answer_log_prob = last_token_logprob[gold_answer_token_id].item()

            sorted_logprob, sorted_indices = torch.sort(last_token_logprob, descending=True)

            top_k_logprob = sorted_logprob[:10].detach().cpu().numpy()
            top_k_indices = sorted_indices[:10].detach()

            decoded_tokens = self.tokenizer.batch_decode(top_k_indices)
            top_k_tokens = [token for token in decoded_tokens]
            assert len(top_k_tokens) == 10

            print(top_k_tokens)
            print(answer)

            is_correct = answer.lower().strip() == top_k_tokens[0].lower().strip()
            top_1_acc = float(is_correct)
            top_5_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:5]])
            top_10_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:10]])

            # Compute log-prob of question and answer
            selected_log_prob = log_prob[:-1, :]  # question - 1 x vocab
            indices = inputs.input_ids[0, 1:].unsqueeze(1)  # question - 1 x 1

            selected_log_prob = torch.gather(selected_log_prob,
                                                index=indices,
                                                dim=1)  # question - 1 x 1
            question_log_prob = selected_log_prob.sum().item()
            total_log_prob = question_log_prob + answer_log_prob

            logprob_results = ContextAnswerLogProb(total_log_prob=total_log_prob,
                                                    answer_log_prob=answer_log_prob,
                                                    answer_len=answer_len)

        self.dataset_metric.accept(is_correct=is_correct,
                                    f1pr_score=None,
                                    log_prob_results=logprob_results,
                                    top_k_acc={1: top_1_acc, 5: top_5_acc, 10: top_10_acc})
