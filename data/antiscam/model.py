# Yu Li-End-to-end trainable non-collaborative dialog system
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from pytorch_pretrained_bert.modeling_openai import OpenAIGPTPreTrainedModel, OpenAIGPTModel, OpenAIGPTLMHead, OpenAIGPTMultipleChoiceHead


class ClassifierHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config, num_classes):
        super(ClassifierHead, self).__init__()
        self.n_embd = config.n_embd
        # self.multiple_choice_token = multiple_choice_token
        self.dropout = nn.Dropout2d(config.resid_pdrop)  # To reproduce the noise_shape parameter of TF implementation
        self.linear = nn.Linear(2*config.n_embd, num_classes)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids, prev_token_ids):
        # Classification logits
        # hidden_state (bsz, num_choices, seq_length, hidden_size)
        # mc_token_ids (bsz, 1, num_choices)
        # add sentence dim
        # mc_token_ids (bsz, num_choices, 1)
        mc_token_ids = mc_token_ids.transpose(1,2)
        prev_token_ids = prev_token_ids.transpose(1,2)

        # (bsz, num_choices, num_candidate, hidden_size)
        mc_token_ids = mc_token_ids.unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1)).transpose(1,2)
        prev_token_ids = prev_token_ids.unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1)).transpose(1, 2)
        prev_token_ids = prev_token_ids.expand(-1, -1, mc_token_ids.size(-2), -1)
        multiple_choice_cur = hidden_states.gather(2, mc_token_ids)
        multiple_choice_prev = hidden_states.gather(2, prev_token_ids)
        multiple_choice_h = torch.cat((multiple_choice_prev, multiple_choice_cur), dim = -1)
        # (bsz, num_choices * num_candidate, hidden_size)
        multiple_choice_h = multiple_choice_h.reshape(multiple_choice_h.size(0), -1, multiple_choice_h.size(-1))
        # (bsz, num_choices, hidden_size)
        multiple_choice_h = self.dropout(multiple_choice_h.transpose(1, 2)).transpose(1, 2)
        multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
        # (bsz, num_choices)
        return multiple_choice_logits

class MultipleIntentsModel(OpenAIGPTPreTrainedModel):
    """OpenAI GPT model with a Language Modeling and a Multiple Choice head ("Improving Language Understanding by Generative Pre-Training").

    OpenAI GPT use a single embedding matrix to store the word and special embeddings.
    Special tokens embeddings are additional tokens that are not pre-trained: [SEP], [CLS]...
    Special tokens need to be trained during the fine-tuning if you use them.
    The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

    The embeddings are ordered as follow in the token embeddings matrice:
        [0,                                                         ----------------------
         ...                                                        -> word embeddings
         config.vocab_size - 1,                                     ______________________
         config.vocab_size,
         ...                                                        -> special embeddings
         config.vocab_size + config.n_special - 1]                  ______________________

    where total_tokens_embeddings can be obtained as config.total_tokens_embeddings and is:
        total_tokens_embeddings = config.vocab_size + config.n_special
    You should use the associate indices to index the embeddings.

    Params:
        config: a OpenAIGPTConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length] with the BPE token
            indices selected in the range [0, total_tokens_embeddings[
        `mc_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token from
            which we should take the hidden state to feed the multiple choice classifier (usually last token of the sequence)
        `position_ids`: an optional torch.LongTensor with the same shape as input_ids
            with the position indices (selected in the range [0, config.n_positions - 1[.
        `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
            You can use it to add a third type of embedding to each input token in the sequence
            (the previous two being the word and position embeddings).
            The input, position and token_type embeddings are summed inside the Transformer before the first
            self-attention block.
        `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with indices selected in [-1, 0, ..., total_tokens_embeddings]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., total_tokens_embeddings]
        `multiple_choice_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `lm_labels` and `multiple_choice_labels` are not `None`:
            Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
        else: a tuple with
            `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, total_tokens_embeddings]
            `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]

    Example usage:
    ```python
    # Already been converted into BPE token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]]])  # (bsz, number of choice, seq length)
    mc_token_ids = torch.LongTensor([[2], [1]]) # (bsz, number of choice)

    config = modeling_openai.OpenAIGPTConfig()

    model = modeling_openai.OpenAIGPTLMHeadModel(config)
    lm_logits, multiple_choice_logits = model(input_ids, mc_token_ids)
    ```
    """

    def __init__(self, config):
        super(MultipleIntentsModel, self).__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = OpenAIGPTLMHead(self.transformer.tokens_embed.weight, config)
        self.multiple_choice_head = OpenAIGPTMultipleChoiceHead(config)
        self.usr_da_classifier = ClassifierHead(config, 15)
        self.sys_da_classifier = ClassifierHead(config, 15)
        self.usr_se_classifier = ClassifierHead(config, 13)
        self.sys_se_classifier = ClassifierHead(config, 13)
        self.apply(self.init_weights)
        self.usr_da_weight =[0.012, 0.1737, 0.029, 0.0127, 0.0277, 0.0455, 0.0059, 0.0099, 0.0831, 0.0294, 0.4777, 0.0516, 0.0273, 0.0103, 0.0042]
        self.usr_se_weight =[0.0149, 0.0667, 0.1044, 0.0217, 0.0727, 0.0217, 0.0615, 0.0407, 0.044, 0.16, 0.2526, 0.1297, 0.0093]
        self.sys_da_weight =[0.0173, 0.1456, 0.0081, 0.0096, 0.0213, 0.0397, 0.0082, 0.0111, 0.0201, 0.021, 0.3494, 0.2912, 0.0182, 0.0349, 0.0043]
        self.sys_se_weight =[0.0117, 0.0573, 0.1667, 0.0214, 0.0833, 0.0174, 0.0359, 0.0582, 0.0495, 0.141, 0.2292, 0.1222, 0.0061]

    def set_num_special_tokens(self, num_special_tokens):
        """ Update input and output embeddings with new embedding matrice
            Make sure we are sharing the embeddings
        """
        self.transformer.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.transformer.tokens_embed.weight)

    def forward(self, input_ids, mc_token_ids, sys_da_token_ids, usr_da_token_ids, prev_token_id, cur_token_id, lm_labels=None, mc_labels=None, token_type_ids=None, sys_da_labels=None, usr_da_labels=None, sys_se_val_labels=None, usr_se_val_labels=None, position_ids=None):
        hidden_states = self.transformer(input_ids, position_ids, token_type_ids)
        #da and se
        sys_da_logits = self.sys_da_classifier(hidden_states, sys_da_token_ids, cur_token_id)
        sys_se_logits = self.sys_se_classifier(hidden_states, sys_da_token_ids, cur_token_id)
        usr_da_logits = self.usr_da_classifier(hidden_states, usr_da_token_ids, prev_token_id)
        usr_se_logits = self.usr_se_classifier(hidden_states, usr_da_token_ids, prev_token_id)

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids)
        #da_logits = self.da_classifier(hidden_states, mc_token_ids)
        losses = []
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            losses.append(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1)))
        if sys_da_labels is not None:
            sys_da_logits = sys_da_logits[:,-3:,:].contiguous()
            sys_da_labels = sys_da_labels.unsqueeze(-1)
            sys_da_labels = sys_da_labels[:,-1:,:].contiguous()
            sys_da_weight_t = torch.FloatTensor(self.sys_da_weight).cuda()
            loss_fct = CrossEntropyLoss(ignore_index=-1, weight = sys_da_weight_t)
            losses.append(loss_fct(sys_da_logits.view(-1, sys_da_logits.size(-1)), sys_da_labels.view(-1)))
        if sys_se_val_labels is not None:
            sys_se_logits = sys_se_logits[:,-3:,:].contiguous()
            sys_se_val_labels = sys_se_val_labels.unsqueeze(-1)
            sys_se_val_labels = sys_se_val_labels[:,-1:,:].contiguous()
            sys_se_weight_t = torch.FloatTensor(self.sys_se_weight).cuda()
            loss_fct = CrossEntropyLoss(ignore_index=-1, weight = sys_se_weight_t)
            losses.append(loss_fct(sys_se_logits.view(-1, sys_se_logits.size(-1)), sys_se_val_labels.view(-1)))
        if usr_da_labels is not None:
            usr_da_logits = usr_da_logits[:,-3:,:].contiguous()
            usr_da_labels = usr_da_labels.unsqueeze(-1)
            usr_da_labels = usr_da_labels[:,-1:,:].contiguous()
            usr_da_weight_t = torch.FloatTensor(self.usr_da_weight).cuda()
            loss_fct = CrossEntropyLoss(ignore_index=-1, weight = usr_da_weight_t)
            losses.append(loss_fct(usr_da_logits.view(-1, usr_da_logits.size(-1)), usr_da_labels.view(-1)))
        if usr_se_val_labels is not None:
            usr_se_logits = usr_se_logits[:,-3:,:].contiguous()
            usr_se_val_labels = usr_se_val_labels.unsqueeze(-1)
            usr_se_val_labels = usr_se_val_labels[:,-1:,:].contiguous()
            usr_se_weight_t = torch.FloatTensor(self.sys_se_weight).cuda()
            loss_fct = CrossEntropyLoss(ignore_index=-1, weight = usr_se_weight_t)
            losses.append(loss_fct(usr_se_logits.view(-1, usr_se_logits.size(-1)), usr_se_val_labels.view(-1)))
        if losses:
            return losses
        return lm_logits, mc_logits, sys_da_logits, sys_se_logits, usr_da_logits, usr_se_logits
