# Yu Li-End-to-end trainable non-collaborative dialog system
import os
import random
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import torch.nn.functional as F

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTTokenizer,
                                     GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
from model import MultipleIntentsModel
from utils import get_dataset
from utils import download_pretrained_model

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<sep>", "<pad>"]
#add dialog_act

DIALOG_ACT_TOKENS = ["<greeting>", "<closing>", "<openq>", "<yesnoq>", "<posans>", "<negans>", "<elicit>",
"<provinfo>", "<refusal>", "<thanking>", "<restothank>", "<apology>","<hold>", "<resstat>", "<noresstat>"]


SE_VALUE_TOKENS = ["<orddetail>", "<ordupdate>", "<payment>", "<name>",
"<identity>", "<address>", "<phonenum>", "<cardinfo>", "<cardnum>", "<cardcsv>", "<carddate>", "<accdetail>", "<others>"]

MODEL_INPUTS = ["input_ids", "mc_token_ids", "sys_da_token_ids", "usr_da_token_ids", "prev_token_id", "cur_token_id", "lm_labels", "mc_labels", "token_type_ids", "sys_dialog_act_labels", "usr_dialog_act_labels", "sys_se_val_labels", "usr_se_val_labels"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2, sep = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]

    #mc_token_ids list
    sys_da_token_ids = [0]*10
    i_index = 0
    mc_token_index = 0
    for token in sequence[-1]:
        if token == sep:
            sys_da_token_ids[mc_token_index] = len(instance["input_ids"]) - len(sequence[-1]) + i_index
            mc_token_index+=1
        i_index+=1
    sys_da_token_ids[mc_token_index] = len(instance["input_ids"]) - 1
    #keep 3 token ids
    if mc_token_index>2:
        sys_da_token_ids = sys_da_token_ids[(mc_token_index-2):(mc_token_index+1)]
    else:
        sys_da_token_ids = sys_da_token_ids[0:3]
    #dialog_act_token_ids list
    usr_da_token_ids = [0]*10
    i_index = 0
    da_token_index = 0
    for token in sequence[-2]:
        if token == sep:
            usr_da_token_ids[da_token_index] = len(instance["input_ids"]) - len(sequence[-2]) - len(sequence[-1]) + i_index
            da_token_index += 1
        i_index += 1
    if da_token_index>3:
        usr_da_token_ids = usr_da_token_ids[(da_token_index-3):(da_token_index)]
    else:
        usr_da_token_ids = usr_da_token_ids[0:3]

    # add prev_token_id and cur_token_id
    prev_token_id = [len(instance["input_ids"]) - len(sequence[-2]) - len(sequence[-1]) - 1]
    if 0 in usr_da_token_ids:
        if usr_da_token_ids.index(0) == 0:
            cur_token_id = [prev_token_id]
            logger.error("Wrong cur_token_id")
        else:
            cur_token_id = [usr_da_token_ids[usr_da_token_ids.index(0) - 1]]
    else:
        cur_token_id = [usr_da_token_ids[-1]]

    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["sys_da_token_ids"] = sys_da_token_ids
    instance["usr_da_token_ids"] = usr_da_token_ids
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    instance["prev_token_id"] = prev_token_id
    instance["cur_token_id"] = cur_token_id
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    sep = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2])
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0:
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for utterance in dialog["utterances"]:
                #add <sep> in his
                history = []
                for uttr_his in utterance["history"][-(2*args.max_history+1):]:
                    history_sen = []
                    for sen in uttr_his["sen_list"]:
                        history_sen = history_sen + sen + [sep]
                    history.append(history_sen)
                #add dialog act
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    sys_dialog_act_index = [tokenizer.convert_tokens_to_ids(tokenizer.decode(sys_da)) - tokenizer.convert_tokens_to_ids(DIALOG_ACT_TOKENS[0]) for sys_da in utterance["sys_dialog_act"][-num_candidates:][j]["da_list"]]#normalize to correct da
                    usr_dialog_act_index = [tokenizer.convert_tokens_to_ids(tokenizer.decode(usr_da)) - tokenizer.convert_tokens_to_ids(DIALOG_ACT_TOKENS[0]) for usr_da in utterance["usr_dialog_act"][0]["da_list"]]#normalize to correct da
                    #max sentences = 3
                    if len(sys_dialog_act_index)>6:
                        sys_dialog_act_index = sys_dialog_act_index[-6:]
                    if len(usr_dialog_act_index)>6:
                        usr_dialog_act_index = usr_dialog_act_index[-6:]
                    sys_dialog_act_labels = [-1]*3
                    sys_se_val_labels = [-1]*3
                    usr_dialog_act_labels = [-1] * 3
                    usr_se_val_labels = [-1] * 3
                    i_index = 0
                    for i_da_index, da_index in enumerate(sys_dialog_act_index):
                        if i_da_index%2==0 and i_index<3:
                            sys_dialog_act_labels[i_index] = da_index
                        elif i_da_index%2==1 and i_index<3:
                            sys_se_val_labels[i_index] = da_index - len(DIALOG_ACT_TOKENS)
                            i_index += 1
                    i_index = 0
                    for i_da_index, da_index in enumerate(usr_dialog_act_index):
                        if i_da_index % 2 == 0 and i_index < 3:
                            usr_dialog_act_labels[i_index] = da_index
                        elif i_da_index % 2 == 1 and i_index < 3:
                            usr_se_val_labels[i_index] = da_index - len(DIALOG_ACT_TOKENS)
                            i_index += 1

                    lm_labels = bool(j == num_candidates-1)
                    #add sep in candidate
                    candidate_input = []
                    for sen in candidate["sen_list"][:-1]:
                        candidate_input = candidate_input + sen + [sep]
                    candidate_input = candidate_input + candidate["sen_list"][-1]

                    instance, _ = build_input_from_segments(persona, history, candidate_input, tokenizer, lm_labels = lm_labels)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["sys_dialog_act_labels"].append(sys_dialog_act_labels)
                    datasets[dataset_name]["usr_dialog_act_labels"].append(usr_dialog_act_labels)
                    datasets[dataset_name]["sys_se_val_labels"].append(sys_se_val_labels)
                    datasets[dataset_name]["usr_se_val_labels"].append(usr_se_val_labels)
                datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                datasets[dataset_name]["n_candidates"] = num_candidates

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/json_gpt.json", help="Path or url of the dataset.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="ff", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    #dialog act loss coefficient
    parser.add_argument("--da_coef", type=float, default=0.5, help="Dialog act loss coefficient")
    parser.add_argument("--se_coef", type=float, default=0.5, help="Semantic loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    #double fine-tune
    if args.model_checkpoint == "ff":
        args.model_checkpoint = download_pretrained_model()
        logger.info("model source: %s", args.model_checkpoint)

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model else MultipleIntentsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    tokenizer.set_special_tokens(DIALOG_ACT_TOKENS  + SE_VALUE_TOKENS + SPECIAL_TOKENS)
    model.set_num_special_tokens(len(DIALOG_ACT_TOKENS + SE_VALUE_TOKENS + SPECIAL_TOKENS))
    model.to(args.device)
    optimizer = OpenAIAdam(model.parameters(), lr=args.lr)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        lm_loss, mc_loss, sys_da_loss, sys_se_loss, usr_da_loss, usr_se_loss = model(*batch)
        #todo loss function should be related to the da or se mapping
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef + sys_da_loss * args.da_coef +
                usr_da_loss * args.da_coef + sys_se_loss * args.se_coef + usr_se_loss * args.se_coef) / args.gradient_accumulation_steps
        #print(lm_loss, mc_loss, sys_da_loss, usr_da_loss, sys_se_loss, usr_se_loss)
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, sys_da_token_ids, usr_da_token_ids, prev_token_id, cur_token_id, lm_labels, mc_labels, token_type_ids, sys_dialog_act_labels, usr_dialog_act_labels, sys_se_val_labels, usr_se_val_labels = batch
            model_outputs = model(input_ids, mc_token_ids, sys_da_token_ids, usr_da_token_ids, prev_token_id, cur_token_id, token_type_ids=token_type_ids)
            lm_logits, mc_logits, sys_da_logits, sys_se_logits, usr_da_logits, usr_se_logits = model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[3], model_outputs[4] , model_outputs[5]
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics 
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=None)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer),
                         event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                                                              another_engine=trainer),
                         event_name=Events.EPOCH_COMPLETED)
        checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)
    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir,
                                                                     WEIGHTS_NAME))
        tb_logger.close()
if __name__ == "__main__":
    random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)
    train()
