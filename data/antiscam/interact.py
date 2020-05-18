# # Copyright (c) 2019-present, HuggingFace Inc.
# Yu Li-End-to-end trainable non-collaborative dialog system
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import nltk

import torch
import torch.nn.functional as F

from pytorch_pretrained_bert import OpenAIGPTTokenizer, GPT2DoubleHeadsModel, GPT2Tokenizer
from model import MultipleIntentsModel
from train import SPECIAL_TOKENS, build_input_from_segments
from utils import get_dataset_personalities, download_pretrained_model

DA_LIST = {0: "greeting", 1: "closing", 2: "open_question", 3: "yes_no_question",
4: "positive_answer", 5: "negative_answer", 6: "elicitation", 7: "providing_info",
8: "refusal", 9: "thanking", 10: "response_to_thanking", 11: "apology", 12: "hold",
13: "responsive_statement", 14: "nonresponsive_statement"}

SE_LIST = {0: "order_detail", 1: "order_update", 2: "payment", 3: "name",
4: "identity", 5: "address", 6: "phone_num", 7: "card_info", 8: "card_num",
9: "card_csv", 10: "card_date", 11: "account_detail", 12: "others"}

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    bos, eos, speaker1, speaker2, sep, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    special_tokens_ids = [bos, eos, speaker1, speaker2, pad]
    if current_output is None:
        current_output = []
    sys_da_output = []
    usr_da_output = []
    end_output = False
    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        mc_token_ids = torch.tensor(instance["mc_token_ids"], device=args.device).unsqueeze(0).unsqueeze(0)
        sys_da_token_ids = torch.tensor(instance["sys_da_token_ids"], device=args.device).unsqueeze(0).unsqueeze(0)
        usr_da_token_ids = torch.tensor(instance["usr_da_token_ids"], device=args.device).unsqueeze(0).unsqueeze(0)
        prev_token_id = torch.tensor(instance["prev_token_id"], device=args.device).unsqueeze(0).unsqueeze(0)
        cur_token_id = torch.tensor(instance["cur_token_id"], device=args.device).unsqueeze(0).unsqueeze(0)


        lm_logits, mc_logits, sys_da_logits, sys_se_logits, usr_da_logits, usr_se_logits  = model(input_ids, 
            mc_token_ids, sys_da_token_ids, usr_da_token_ids, prev_token_id, cur_token_id, token_type_ids=token_type_ids)

        #todo
        if "gpt2" == args.model:
            lm_logits = lm_logits[0]
            logits = lm_logits[0, -1, :] / args.temperature
        else:
            logits = lm_logits[0, 0, -1, :] / args.temperature

        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)
        # need to generate one more token
        if end_output is True:
            # add dialog_act
            sys_da_probs = F.softmax(sys_da_logits, dim=-1)
            sys_se_probs = F.softmax(sys_se_logits, dim=-1)
            usr_da_probs = F.softmax(usr_da_logits, dim=-1)
            usr_se_probs = F.softmax(usr_se_logits, dim=-1)
            sys_da_pred = torch.topk(sys_da_probs, 1)[1][0].transpose(0,1).squeeze(0).tolist()
            sys_se_pred = torch.topk(sys_se_probs, 1)[1][0].transpose(0,1).squeeze(0).tolist()
            usr_da_pred = torch.topk(usr_da_probs, 1)[1][0].transpose(0,1).squeeze(0).tolist()
            usr_se_pred = torch.topk(usr_se_probs, 1)[1][0].transpose(0,1).squeeze(0).tolist()
            sys_da_ids_list = torch.tensor(instance["sys_da_token_ids"], device=args.device).tolist()
            usr_da_ids_list = torch.tensor(instance["usr_da_token_ids"], device=args.device).tolist()
            if 0 in sys_da_ids_list:
                sys_da_pred = sys_da_pred[0:sys_da_ids_list.index(0)]
                sys_se_pred = sys_se_pred[0:sys_da_ids_list.index(0)]
            if 0 in usr_da_ids_list:
                usr_da_pred = usr_da_pred[0:usr_da_ids_list.index(0)]
                usr_se_pred = usr_se_pred[0:usr_da_ids_list.index(0)]

            sys_da_output = [DA_LIST[da_index] for da_index in sys_da_pred]
            sys_se_output = [SE_LIST[se_index] for se_index in sys_se_pred]
            usr_da_output = [DA_LIST[da_index] for da_index in usr_da_pred]
            usr_se_output = [SE_LIST[se_index] for se_index in usr_se_pred]
            break
        if prev.item() in special_tokens_ids:
            end_output = True
        current_output.append(prev.item())

    return current_output, sys_da_output, sys_se_output, usr_da_output, usr_se_output

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/json_gpt2.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        args.model_checkpoint = download_pretrained_model()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model else MultipleIntentsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    model.eval()

    logger.info("Sample a personality")
    personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)

    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
    nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    history = []
    sep = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2])
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        #todo
        segs = nltk_tokenizer.tokenize(raw_text)
        input_text = []
        for seg in segs:
            input_text = input_text + tokenizer.encode(seg) + [sep]
        history.append(input_text)
        print("*******************************All sampled candidates*******************************")
        for num_can in range(16):
            with torch.no_grad():
                #finish_output = sample_sequence(personality, history, tokenizer, model, args)
                out_ids, sys_da_output, sys_se_output, usr_da_output, usr_se_output = sample_sequence(personality, history, tokenizer, model, args)
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            print("------------Candidate", num_can, "---------------")
            print(out_text)
            print("user_intent: " + ', '.join(
                usr_da_output))
            print("user_semantic_slot: " + ', '.join(
                usr_se_output))
            print("sys_intent: " + ', '.join(sys_da_output))
            print("sys_semantic_slot: " + ', '.join(
                sys_se_output))
            print("--------------------------------------------")
        print(out_text)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
if __name__ == "__main__":
    run()
