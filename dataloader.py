<<<<<<< HEAD
<<<<<<< HEAD
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer

# import os
# import sys
#
# pre_dir = os.path.dirname(os.path.dirname("preprocess.py"))
# if not pre_dir in sys.path:
#     sys.path.append(pre_dir)
#
# con_dir = os.path.dirname(os.path.dirname("constants.py"))
# if not con_dir in sys.path:
#     sys.path.append(con_dir)

from preprocess import passage_blocks, get_question
from constants import *

def collate_fn(batch):
    # for training
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    tags = nbatch['tags']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    turn_mask = nbatch['turn_mask']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)
    ntags = pad_sequence(tags, batch_first=True, padding_value=-1)
    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['tags'] = ntags
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    nbatch['turn_mask'] = torch.tensor(turn_mask, dtype=torch.uint8)
    return nbatch


def collate_fn1(batch):
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)
    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    return nbatch


def get_inputs(context, q, tokenizer, title="", max_len=200, ans=[], head_entity=None):
    query = tokenizer.tokenize(q)
    tags = [tag_idxs['O']]*len(context)
    for i, an in enumerate(ans):
        start, end = an[1:-1]
        end = end-1
        if start != end:
            tags[start] = tag_idxs['B']
            tags[end] = tag_idxs['E']
            for i in range(start+1, end):
                tags[i] = tag_idxs['M']
        else:
            tags[start] = tag_idxs['S']
    if head_entity:
        h_start, h_end = head_entity[1], head_entity[2]
        context = context[:h_start]+['[unused0]'] + \
            context[h_start:h_end]+["[unused1]"]+context[h_end:]
        tags = tags[:h_start]+[tag_idxs['O']] + \
            tags[h_start:h_end]+[tag_idxs['O']]+tags[h_end:]
    txt_len = len(query)+len(title)+len(context) + \
        4 if title else len(query)+len(context)+3
    if txt_len > max_len:
        context = context[:max_len -
                          len(query) - 3] if not title else context[:max_len-len(query)-len(title)-4]
        tags = tags[:max_len -
                    len(query) - 3] if not title else tags[:max_len-len(query)-len(title)-4]
    if title:
        txt = ['[CLS]'] + query+['[SEP]'] + \
            title + ['[SEP]'] + context + ['[SEP]']
    else:
        txt = ['[CLS]'] + query + ['[SEP]'] + context + ['[SEP]']
    txt_ids = tokenizer.convert_tokens_to_ids(txt)
    # [CLS] is used to judge whether there is an answe
    if not title:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1] * (len(query) + 1) + tags + [-1]
        context_mask = [1] + [0] * (len(query) + 1) + [1] * len(context) + [0]
        token_type_ids = [0] * (len(query) + 2) + [1] * (len(context) + 1)
    else:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1]*(len(query)+len(title)+2) + tags + [-1]
        context_mask = [1] + [0] * \
            (len(query)+len(title)+2) + [1] * len(context) + [0]
        token_type_ids = [0]*(len(query)+len(title)+3)+[1]*(len(context) + 1)
    return txt_ids, tags1, context_mask, token_type_ids


def query2relation(question, question_templates):
    turn2_questions = question_templates['qa_turn2']
    turn2_questions = {v: k for k, v in turn2_questions.items()}
    for k, v in turn2_questions.items():
        k1 = k.replace("XXX.", "")
        if question.startswith(k1):
            return eval(v)
    raise Exception("cannot find the relation type corresponding to the query, if the \
                 query template is changed, please re-implement this function according to the new template")


class MyDataset:
    def __init__(self, path, tokenizer, max_len=512, threshold=5):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.init_data()

    def init_data(self):
        self.all_t1 = []
        self.all_t2 = []
        idx1s = ace2005_idx1
        idx2s = ace2005_idx2
        dist = ace2005_dist
        question_templates = ace2005_question_templates
        for d in tqdm(self.data, desc="dataset"):
            context = d['context']
            title = d['title']
            qa_pairs = d['qa_pairs']
            t1 = qa_pairs[0]
            t2 = qa_pairs[1]
            t1_qas = []
            t2_qas = []
            for i, (q, ans) in enumerate(t1.items()):
                txt_ids, tags, context_mask, token_type_ids = get_inputs(
                    context, q, self.tokenizer, title, self.max_len, ans)
                t1_qas.append(
                    {"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask, "token_type_ids": token_type_ids, 'turn_mask': 0})
            for t in t2:
                head_entity = t['head_entity']
                for q, ans in t['qas'].items():
                    rel = query2relation(q, question_templates)
                    idx1, idx2 = rel[0], rel[1:]
                    idx1, idx2 = idx1s[idx1], idx2s[idx2]
                    if dist[idx1][idx2] >= self.threshold:
                        txt_ids, tags, context_mask, token_type_ids = get_inputs(
                            context, q, self.tokenizer, title, self.max_len, ans, head_entity)
                        t2_qas.append({"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask,
                                       "token_type_ids": token_type_ids, 'turn_mask': 1})
            self.all_t1.extend(t1_qas)
            self.all_t2.extend(t2_qas)
        self.all_qas = self.all_t1+self.all_t2

    def __len__(self):
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]


class T1Dataset:
    def __init__(self, test_path, tokenizer, window_size, overlap, max_len=512):
        with open(test_path, encoding="utf=8") as f:
            data = json.load(f)
        dataset_entities = ace2005_entities
        question_templates = ace2005_question_templates
        self.t1_qas = []
        self.passages = []
        self.entities = []
        self.relations = []
        self.titles = []
        self.window_size = window_size
        self.overlap = overlap
        self.t1_querys = []
        self.t1_ids = []
        self.t1_gold = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.passage_windows = []
        self.query_offset1 = []
        self.window_offset_base = window_size-overlap
        for ent_type in dataset_entities:
            query = get_question(question_templates, ent_type)
            self.t1_querys.append(query)
        for p_id, d in enumerate(tqdm(data, desc="t1_dataset")):
            # print(d.keys())
            passage = d["context"]
            entities=[]
            for x in d['qa_pairs'][0]:
                if "entities  in the context." in x:
                    for y in d['qa_pairs'][0][x]:
                        entities.append(y)
            title = d['title']
            self.passages.append(passage)
            self.entities.append(entities)
            self.titles.append(title)
            blocks, _ = passage_blocks(passage, window_size, overlap)
            self.passage_windows.append(blocks)
            for ent in entities:
                self.t1_gold.append((p_id, tuple(ent[:-1])))
            for b_id, block in enumerate(blocks):
                for q_id, q in enumerate(self.t1_querys):
                    txt_ids, _, context_mask, token_type_ids = get_inputs(
                        block, q, tokenizer, title, max_len)
                    self.t1_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                        "token_type_ids": token_type_ids})
                    self.t1_ids.append((p_id, b_id, dataset_entities[q_id]))
                    ofs = len(title)+len(tokenizer.tokenize(q))+3
                    self.query_offset1.append(ofs)

    def __len__(self):
        return len(self.t1_qas)

    def __getitem__(self, i):
        return self.t1_qas[i]


class T2Dataset:
    def __init__(self, t1_dataset, t1_predict, threshold=5):
        idx1s = ace2005_idx1
        idx2s = ace2005_idx2
        dist = ace2005_dist
        dataset_entities = ace2005_entities
        dataset_relations = ace2005_relations
        question_templates = ace2005_question_templates
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        titles = t1_dataset.titles
        passage_windows = t1_dataset.passage_windows
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1 = t1_dataset.query_offset1
        window_offset_base = t1_dataset.window_offset_base
        for passage_id, (ents, rels) in enumerate(zip(entities, relations)):
            for re in rels:
                head, rel, end = ents[re[1]], re[0], ents[re[2]]
                self.t2_gold.append(
                    (passage_id, (tuple(head[:-1]), rel, tuple(end[:-1]))))
        for i, (_id, pre) in enumerate(zip(tqdm(t1_ids, desc="t2 dataset"), t1_predict)):
            passage_id, window_id, head_entity_type = _id
            window_offset = window_offset_base*window_id
            context = passage_windows[passage_id][window_id]
            title = titles[passage_id]
            head_entities = []
            for start, end in pre:
                start1, end1 = start - \
                    query_offset1[i]+window_offset, end - \
                    query_offset1[i]+window_offset
                ent_str = tokenizer.convert_tokens_to_string(
                    passages[passage_id][start1:end1])
                head_entity = (head_entity_type, start1, end1, ent_str)
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in dataset_relations:
                    for end_ent_type in dataset_entities:
                        idx1, idx2 = idx1s[head_entity[0]
                                           ], idx2s[(rel, end_ent_type)]
                        if dist[idx1][idx2] >= threshold:
                            query = get_question(
                                question_templates, head_entity, rel, end_ent_type)
                            window_head_entity = (
                                head_entity[0], head_entity[1]-window_offset, head_entity[2]-window_offset, head_entity[3])
                            txt_ids, _, context_mask, token_type_ids = get_inputs(
                                context, query, tokenizer, title, max_len, [], window_head_entity)
                            self.t2_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                                "token_type_ids": token_type_ids})
                            self.t2_ids.append(
                                (passage_id, window_id, head_entity[:-1], rel, end_ent_type))
                            ofs = len(title) + \
                                len(tokenizer.tokenize(query)) + 3
                            self.query_offset2.append(ofs)

    def __len__(self):
        return len(self.t2_qas)

    def __getitem__(self, i):
        return self.t2_qas[i]


def load_data(file_path, batch_size, max_len, dist=False, shuffle=False, threshold=5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = MyDataset(file_path, tokenizer,
                        max_len, threshold)
    sampler = DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, shuffle=shuffle if not sampler else False,
                            collate_fn=collate_fn)
    return dataloader


def reload_data(old_dataloader, batch_size, max_len, threshold, shuffle=True):
    dataset = old_dataloader.dataset
    old_max_len, old_threshold = dataset.max_len, dataset.threshold
    if not(old_max_len == max_len and old_threshold == threshold):
        dataset.max_len = max_len
        dataset.threshold = threshold
        dataset.init_data()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size, sampler=sampler, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t1_data(test_path,  window_size, overlap, batch_size=10, max_len=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    t1_dataset = T1Dataset(test_path,
                           tokenizer, window_size, overlap, max_len)
    dataloader = DataLoader(t1_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader


def load_t2_data(t1_dataset, t1_predict, batch_size=10, threshold=5):
    t2_dataset = T2Dataset(t1_dataset, t1_predict, threshold)
    dataloader = DataLoader(t2_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader
=======
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer

# import os
# import sys
#
# pre_dir = os.path.dirname(os.path.dirname("preprocess.py"))
# if not pre_dir in sys.path:
#     sys.path.append(pre_dir)
#
# con_dir = os.path.dirname(os.path.dirname("constants.py"))
# if not con_dir in sys.path:
#     sys.path.append(con_dir)

from preprocess import passage_blocks, get_question
from constants import *

def collate_fn(batch):
    # for training
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    tags = nbatch['tags']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    turn_mask = nbatch['turn_mask']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)
    ntags = pad_sequence(tags, batch_first=True, padding_value=-1)
    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['tags'] = ntags
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    nbatch['turn_mask'] = torch.tensor(turn_mask, dtype=torch.uint8)
    return nbatch


def collate_fn1(batch):
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)
    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    return nbatch


def get_inputs(context, q, tokenizer, title="", max_len=200, ans=[], head_entity=None):
    query = tokenizer.tokenize(q)
    tags = [tag_idxs['O']]*len(context)
    for i, an in enumerate(ans):
        start, end = an[1:-1]
        end = end-1
        if start != end:
            tags[start] = tag_idxs['B']
            tags[end] = tag_idxs['E']
            for i in range(start+1, end):
                tags[i] = tag_idxs['M']
        else:
            tags[start] = tag_idxs['S']
    if head_entity:
        h_start, h_end = head_entity[1], head_entity[2]
        context = context[:h_start]+['[unused0]'] + \
            context[h_start:h_end]+["[unused1]"]+context[h_end:]
        tags = tags[:h_start]+[tag_idxs['O']] + \
            tags[h_start:h_end]+[tag_idxs['O']]+tags[h_end:]
    txt_len = len(query)+len(title)+len(context) + \
        4 if title else len(query)+len(context)+3
    if txt_len > max_len:
        context = context[:max_len -
                          len(query) - 3] if not title else context[:max_len-len(query)-len(title)-4]
        tags = tags[:max_len -
                    len(query) - 3] if not title else tags[:max_len-len(query)-len(title)-4]
    if title:
        txt = ['[CLS]'] + query+['[SEP]'] + \
            title + ['[SEP]'] + context + ['[SEP]']
    else:
        txt = ['[CLS]'] + query + ['[SEP]'] + context + ['[SEP]']
    txt_ids = tokenizer.convert_tokens_to_ids(txt)
    # [CLS] is used to judge whether there is an answe
    if not title:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1] * (len(query) + 1) + tags + [-1]
        context_mask = [1] + [0] * (len(query) + 1) + [1] * len(context) + [0]
        token_type_ids = [0] * (len(query) + 2) + [1] * (len(context) + 1)
    else:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1]*(len(query)+len(title)+2) + tags + [-1]
        context_mask = [1] + [0] * \
            (len(query)+len(title)+2) + [1] * len(context) + [0]
        token_type_ids = [0]*(len(query)+len(title)+3)+[1]*(len(context) + 1)
    return txt_ids, tags1, context_mask, token_type_ids


def query2relation(question, question_templates):
    turn2_questions = question_templates['qa_turn2']
    turn2_questions = {v: k for k, v in turn2_questions.items()}
    for k, v in turn2_questions.items():
        k1 = k.replace("XXX.", "")
        if question.startswith(k1):
            return eval(v)
    raise Exception("cannot find the relation type corresponding to the query, if the \
                 query template is changed, please re-implement this function according to the new template")


class MyDataset:
    def __init__(self, path, tokenizer, max_len=512, threshold=5):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.init_data()

    def init_data(self):
        self.all_t1 = []
        self.all_t2 = []
        idx1s = ace2005_idx1
        idx2s = ace2005_idx2
        dist = ace2005_dist
        question_templates = ace2005_question_templates
        for d in tqdm(self.data, desc="dataset"):
            context = d['context']
            title = d['title']
            qa_pairs = d['qa_pairs']
            t1 = qa_pairs[0]
            t2 = qa_pairs[1]
            t1_qas = []
            t2_qas = []
            for i, (q, ans) in enumerate(t1.items()):
                txt_ids, tags, context_mask, token_type_ids = get_inputs(
                    context, q, self.tokenizer, title, self.max_len, ans)
                t1_qas.append(
                    {"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask, "token_type_ids": token_type_ids, 'turn_mask': 0})
            for t in t2:
                head_entity = t['head_entity']
                for q, ans in t['qas'].items():
                    rel = query2relation(q, question_templates)
                    idx1, idx2 = rel[0], rel[1:]
                    idx1, idx2 = idx1s[idx1], idx2s[idx2]
                    if dist[idx1][idx2] >= self.threshold:
                        txt_ids, tags, context_mask, token_type_ids = get_inputs(
                            context, q, self.tokenizer, title, self.max_len, ans, head_entity)
                        t2_qas.append({"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask,
                                       "token_type_ids": token_type_ids, 'turn_mask': 1})
            self.all_t1.extend(t1_qas)
            self.all_t2.extend(t2_qas)
        self.all_qas = self.all_t1+self.all_t2

    def __len__(self):
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]


class T1Dataset:
    def __init__(self, test_path, tokenizer, window_size, overlap, max_len=512):
        with open(test_path, encoding="utf=8") as f:
            data = json.load(f)
        dataset_entities = ace2005_entities
        question_templates = ace2005_question_templates
        self.t1_qas = []
        self.passages = []
        self.entities = []
        self.relations = []
        self.titles = []
        self.window_size = window_size
        self.overlap = overlap
        self.t1_querys = []
        self.t1_ids = []
        self.t1_gold = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.passage_windows = []
        self.query_offset1 = []
        self.window_offset_base = window_size-overlap
        for ent_type in dataset_entities:
            query = get_question(question_templates, ent_type)
            self.t1_querys.append(query)
        for p_id, d in enumerate(tqdm(data, desc="t1_dataset")):
            # print(d.keys())
            passage = d["context"]
            entities=[]
            for x in d['qa_pairs'][0]:
                if "entities  in the context." in x:
                    for y in d['qa_pairs'][0][x]:
                        entities.append(y)
            title = d['title']
            self.passages.append(passage)
            self.entities.append(entities)
            self.titles.append(title)
            blocks, _ = passage_blocks(passage, window_size, overlap)
            self.passage_windows.append(blocks)
            for ent in entities:
                self.t1_gold.append((p_id, tuple(ent[:-1])))
            for b_id, block in enumerate(blocks):
                for q_id, q in enumerate(self.t1_querys):
                    txt_ids, _, context_mask, token_type_ids = get_inputs(
                        block, q, tokenizer, title, max_len)
                    self.t1_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                        "token_type_ids": token_type_ids})
                    self.t1_ids.append((p_id, b_id, dataset_entities[q_id]))
                    ofs = len(title)+len(tokenizer.tokenize(q))+3
                    self.query_offset1.append(ofs)

    def __len__(self):
        return len(self.t1_qas)

    def __getitem__(self, i):
        return self.t1_qas[i]


class T2Dataset:
    def __init__(self, t1_dataset, t1_predict, threshold=5):
        idx1s = ace2005_idx1
        idx2s = ace2005_idx2
        dist = ace2005_dist
        dataset_entities = ace2005_entities
        dataset_relations = ace2005_relations
        question_templates = ace2005_question_templates
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        titles = t1_dataset.titles
        passage_windows = t1_dataset.passage_windows
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1 = t1_dataset.query_offset1
        window_offset_base = t1_dataset.window_offset_base
        for passage_id, (ents, rels) in enumerate(zip(entities, relations)):
            for re in rels:
                head, rel, end = ents[re[1]], re[0], ents[re[2]]
                self.t2_gold.append(
                    (passage_id, (tuple(head[:-1]), rel, tuple(end[:-1]))))
        for i, (_id, pre) in enumerate(zip(tqdm(t1_ids, desc="t2 dataset"), t1_predict)):
            passage_id, window_id, head_entity_type = _id
            window_offset = window_offset_base*window_id
            context = passage_windows[passage_id][window_id]
            title = titles[passage_id]
            head_entities = []
            for start, end in pre:
                start1, end1 = start - \
                    query_offset1[i]+window_offset, end - \
                    query_offset1[i]+window_offset
                ent_str = tokenizer.convert_tokens_to_string(
                    passages[passage_id][start1:end1])
                head_entity = (head_entity_type, start1, end1, ent_str)
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in dataset_relations:
                    for end_ent_type in dataset_entities:
                        idx1, idx2 = idx1s[head_entity[0]
                                           ], idx2s[(rel, end_ent_type)]
                        if dist[idx1][idx2] >= threshold:
                            query = get_question(
                                question_templates, head_entity, rel, end_ent_type)
                            window_head_entity = (
                                head_entity[0], head_entity[1]-window_offset, head_entity[2]-window_offset, head_entity[3])
                            txt_ids, _, context_mask, token_type_ids = get_inputs(
                                context, query, tokenizer, title, max_len, [], window_head_entity)
                            self.t2_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                                "token_type_ids": token_type_ids})
                            self.t2_ids.append(
                                (passage_id, window_id, head_entity[:-1], rel, end_ent_type))
                            ofs = len(title) + \
                                len(tokenizer.tokenize(query)) + 3
                            self.query_offset2.append(ofs)

    def __len__(self):
        return len(self.t2_qas)

    def __getitem__(self, i):
        return self.t2_qas[i]


def load_data(file_path, batch_size, max_len, dist=False, shuffle=False, threshold=5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = MyDataset(file_path, tokenizer,
                        max_len, threshold)
    sampler = DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, shuffle=shuffle if not sampler else False,
                            collate_fn=collate_fn)
    return dataloader


def reload_data(old_dataloader, batch_size, max_len, threshold, shuffle=True):
    dataset = old_dataloader.dataset
    old_max_len, old_threshold = dataset.max_len, dataset.threshold
    if not(old_max_len == max_len and old_threshold == threshold):
        dataset.max_len = max_len
        dataset.threshold = threshold
        dataset.init_data()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size, sampler=sampler, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t1_data(test_path,  window_size, overlap, batch_size=10, max_len=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    t1_dataset = T1Dataset(test_path,
                           tokenizer, window_size, overlap, max_len)
    dataloader = DataLoader(t1_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader


def load_t2_data(t1_dataset, t1_predict, batch_size=10, threshold=5):
    t2_dataset = T2Dataset(t1_dataset, t1_predict, threshold)
    dataloader = DataLoader(t2_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader
>>>>>>> 4cf9ab73d9db67401f3e6a17a45675eb711809a2
=======
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer

# import os
# import sys
#
# pre_dir = os.path.dirname(os.path.dirname("preprocess.py"))
# if not pre_dir in sys.path:
#     sys.path.append(pre_dir)
#
# con_dir = os.path.dirname(os.path.dirname("constants.py"))
# if not con_dir in sys.path:
#     sys.path.append(con_dir)

from preprocess import passage_blocks, get_question
from constants import *

def collate_fn(batch):
    # for training
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    tags = nbatch['tags']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    turn_mask = nbatch['turn_mask']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)
    ntags = pad_sequence(tags, batch_first=True, padding_value=-1)
    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['tags'] = ntags
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    nbatch['turn_mask'] = torch.tensor(turn_mask, dtype=torch.uint8)
    return nbatch


def collate_fn1(batch):
    nbatch = {}
    for b in batch:
        for k, v in b.items():
            nbatch[k] = nbatch.get(k, []) + [torch.tensor(v)]
    txt_ids = nbatch['txt_ids']
    context_mask = nbatch['context_mask']
    token_type_ids = nbatch['token_type_ids']
    ntxt_ids = pad_sequence(txt_ids, batch_first=True,
                            padding_value=0)
    ncontext_mask = pad_sequence(
        context_mask, batch_first=True, padding_value=0)
    ntoken_type_ids = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1)
    attention_mask = torch.zeros(ntxt_ids.shape)
    for i in range(len(ntxt_ids)):
        txt_len = len(txt_ids[i])
        attention_mask[i, :txt_len] = 1
    nbatch['txt_ids'] = ntxt_ids
    nbatch['context_mask'] = ncontext_mask
    nbatch['token_type_ids'] = ntoken_type_ids
    nbatch['attention_mask'] = attention_mask
    return nbatch


def get_inputs(context, q, tokenizer, title="", max_len=200, ans=[], head_entity=None):
    query = tokenizer.tokenize(q)
    tags = [tag_idxs['O']]*len(context)
    for i, an in enumerate(ans):
        start, end = an[1:-1]
        end = end-1
        if start != end:
            tags[start] = tag_idxs['B']
            tags[end] = tag_idxs['E']
            for i in range(start+1, end):
                tags[i] = tag_idxs['M']
        else:
            tags[start] = tag_idxs['S']
    if head_entity:
        h_start, h_end = head_entity[1], head_entity[2]
        context = context[:h_start]+['[unused0]'] + \
            context[h_start:h_end]+["[unused1]"]+context[h_end:]
        tags = tags[:h_start]+[tag_idxs['O']] + \
            tags[h_start:h_end]+[tag_idxs['O']]+tags[h_end:]
    txt_len = len(query)+len(title)+len(context) + \
        4 if title else len(query)+len(context)+3
    if txt_len > max_len:
        context = context[:max_len -
                          len(query) - 3] if not title else context[:max_len-len(query)-len(title)-4]
        tags = tags[:max_len -
                    len(query) - 3] if not title else tags[:max_len-len(query)-len(title)-4]
    if title:
        txt = ['[CLS]'] + query+['[SEP]'] + \
            title + ['[SEP]'] + context + ['[SEP]']
    else:
        txt = ['[CLS]'] + query + ['[SEP]'] + context + ['[SEP]']
    txt_ids = tokenizer.convert_tokens_to_ids(txt)
    # [CLS] is used to judge whether there is an answe
    if not title:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1] * (len(query) + 1) + tags + [-1]
        context_mask = [1] + [0] * (len(query) + 1) + [1] * len(context) + [0]
        token_type_ids = [0] * (len(query) + 2) + [1] * (len(context) + 1)
    else:
        tags1 = [tag_idxs['O'] if len(
            ans) > 0 else tag_idxs['S']] + [-1]*(len(query)+len(title)+2) + tags + [-1]
        context_mask = [1] + [0] * \
            (len(query)+len(title)+2) + [1] * len(context) + [0]
        token_type_ids = [0]*(len(query)+len(title)+3)+[1]*(len(context) + 1)
    return txt_ids, tags1, context_mask, token_type_ids


def query2relation(question, question_templates):
    turn2_questions = question_templates['qa_turn2']
    turn2_questions = {v: k for k, v in turn2_questions.items()}
    for k, v in turn2_questions.items():
        k1 = k.replace("XXX.", "")
        if question.startswith(k1):
            return eval(v)
    raise Exception("cannot find the relation type corresponding to the query, if the \
                 query template is changed, please re-implement this function according to the new template")


class MyDataset:
    def __init__(self, path, tokenizer, max_len=512, threshold=5):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.init_data()

    def init_data(self):
        self.all_t1 = []
        self.all_t2 = []
        idx1s = ace2005_idx1
        idx2s = ace2005_idx2
        dist = ace2005_dist
        question_templates = ace2005_question_templates
        for d in tqdm(self.data, desc="dataset"):
            context = d['context']
            title = d['title']
            qa_pairs = d['qa_pairs']
            t1 = qa_pairs[0]
            t2 = qa_pairs[1]
            t1_qas = []
            t2_qas = []
            for i, (q, ans) in enumerate(t1.items()):
                txt_ids, tags, context_mask, token_type_ids = get_inputs(
                    context, q, self.tokenizer, title, self.max_len, ans)
                t1_qas.append(
                    {"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask, "token_type_ids": token_type_ids, 'turn_mask': 0})
            for t in t2:
                head_entity = t['head_entity']
                for q, ans in t['qas'].items():
                    rel = query2relation(q, question_templates)
                    idx1, idx2 = rel[0], rel[1:]
                    idx1, idx2 = idx1s[idx1], idx2s[idx2]
                    if dist[idx1][idx2] >= self.threshold:
                        txt_ids, tags, context_mask, token_type_ids = get_inputs(
                            context, q, self.tokenizer, title, self.max_len, ans, head_entity)
                        t2_qas.append({"txt_ids": txt_ids, "tags": tags, "context_mask": context_mask,
                                       "token_type_ids": token_type_ids, 'turn_mask': 1})
            self.all_t1.extend(t1_qas)
            self.all_t2.extend(t2_qas)
        self.all_qas = self.all_t1+self.all_t2

    def __len__(self):
        return len(self.all_qas)

    def __getitem__(self, i):
        return self.all_qas[i]


class T1Dataset:
    def __init__(self, test_path, tokenizer, window_size, overlap, max_len=512):
        with open(test_path, encoding="utf=8") as f:
            data = json.load(f)
        dataset_entities = ace2005_entities
        question_templates = ace2005_question_templates
        self.t1_qas = []
        self.passages = []
        self.entities = []
        self.relations = []
        self.titles = []
        self.window_size = window_size
        self.overlap = overlap
        self.t1_querys = []
        self.t1_ids = []
        self.t1_gold = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.passage_windows = []
        self.query_offset1 = []
        self.window_offset_base = window_size-overlap
        for ent_type in dataset_entities:
            query = get_question(question_templates, ent_type)
            self.t1_querys.append(query)
        for p_id, d in enumerate(tqdm(data, desc="t1_dataset")):
            # print(d.keys())
            passage = d["context"]
            entities=[]
            for x in d['qa_pairs'][0]:
                if "entities  in the context." in x:
                    for y in d['qa_pairs'][0][x]:
                        entities.append(y)
            title = d['title']
            self.passages.append(passage)
            self.entities.append(entities)
            self.titles.append(title)
            blocks, _ = passage_blocks(passage, window_size, overlap)
            self.passage_windows.append(blocks)
            for ent in entities:
                self.t1_gold.append((p_id, tuple(ent[:-1])))
            for b_id, block in enumerate(blocks):
                for q_id, q in enumerate(self.t1_querys):
                    txt_ids, _, context_mask, token_type_ids = get_inputs(
                        block, q, tokenizer, title, max_len)
                    self.t1_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                        "token_type_ids": token_type_ids})
                    self.t1_ids.append((p_id, b_id, dataset_entities[q_id]))
                    ofs = len(title)+len(tokenizer.tokenize(q))+3
                    self.query_offset1.append(ofs)

    def __len__(self):
        return len(self.t1_qas)

    def __getitem__(self, i):
        return self.t1_qas[i]


class T2Dataset:
    def __init__(self, t1_dataset, t1_predict, threshold=5):
        idx1s = ace2005_idx1
        idx2s = ace2005_idx2
        dist = ace2005_dist
        dataset_entities = ace2005_entities
        dataset_relations = ace2005_relations
        question_templates = ace2005_question_templates
        tokenizer = t1_dataset.tokenizer
        max_len = t1_dataset.max_len
        t1_ids = t1_dataset.t1_ids
        passages = t1_dataset.passages
        titles = t1_dataset.titles
        passage_windows = t1_dataset.passage_windows
        self.t2_qas = []
        self.t2_ids = []
        self.t2_gold = []
        self.query_offset2 = []
        relations = t1_dataset.relations
        entities = t1_dataset.entities
        query_offset1 = t1_dataset.query_offset1
        window_offset_base = t1_dataset.window_offset_base
        for passage_id, (ents, rels) in enumerate(zip(entities, relations)):
            for re in rels:
                head, rel, end = ents[re[1]], re[0], ents[re[2]]
                self.t2_gold.append(
                    (passage_id, (tuple(head[:-1]), rel, tuple(end[:-1]))))
        for i, (_id, pre) in enumerate(zip(tqdm(t1_ids, desc="t2 dataset"), t1_predict)):
            passage_id, window_id, head_entity_type = _id
            window_offset = window_offset_base*window_id
            context = passage_windows[passage_id][window_id]
            title = titles[passage_id]
            head_entities = []
            for start, end in pre:
                start1, end1 = start - \
                    query_offset1[i]+window_offset, end - \
                    query_offset1[i]+window_offset
                ent_str = tokenizer.convert_tokens_to_string(
                    passages[passage_id][start1:end1])
                head_entity = (head_entity_type, start1, end1, ent_str)
                head_entities.append(head_entity)
            for head_entity in head_entities:
                for rel in dataset_relations:
                    for end_ent_type in dataset_entities:
                        idx1, idx2 = idx1s[head_entity[0]
                                           ], idx2s[(rel, end_ent_type)]
                        if dist[idx1][idx2] >= threshold:
                            query = get_question(
                                question_templates, head_entity, rel, end_ent_type)
                            window_head_entity = (
                                head_entity[0], head_entity[1]-window_offset, head_entity[2]-window_offset, head_entity[3])
                            txt_ids, _, context_mask, token_type_ids = get_inputs(
                                context, query, tokenizer, title, max_len, [], window_head_entity)
                            self.t2_qas.append({"txt_ids": txt_ids, "context_mask": context_mask,
                                                "token_type_ids": token_type_ids})
                            self.t2_ids.append(
                                (passage_id, window_id, head_entity[:-1], rel, end_ent_type))
                            ofs = len(title) + \
                                len(tokenizer.tokenize(query)) + 3
                            self.query_offset2.append(ofs)

    def __len__(self):
        return len(self.t2_qas)

    def __getitem__(self, i):
        return self.t2_qas[i]


def load_data(file_path, batch_size, max_len, dist=False, shuffle=False, threshold=5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = MyDataset(file_path, tokenizer,
                        max_len, threshold)
    sampler = DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, shuffle=shuffle if not sampler else False,
                            collate_fn=collate_fn)
    return dataloader


def reload_data(old_dataloader, batch_size, max_len, threshold, shuffle=True):
    dataset = old_dataloader.dataset
    old_max_len, old_threshold = dataset.max_len, dataset.threshold
    if not(old_max_len == max_len and old_threshold == threshold):
        dataset.max_len = max_len
        dataset.threshold = threshold
        dataset.init_data()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size, sampler=sampler, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t1_data(test_path,  window_size, overlap, batch_size=10, max_len=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    t1_dataset = T1Dataset(test_path,
                           tokenizer, window_size, overlap, max_len)
    dataloader = DataLoader(t1_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader


def load_t2_data(t1_dataset, t1_predict, batch_size=10, threshold=5):
    t2_dataset = T2Dataset(t1_dataset, t1_predict, threshold)
    dataloader = DataLoader(t2_dataset, batch_size, collate_fn=collate_fn1)
    return dataloader
>>>>>>> da9aae038c33fbb1399c5b531fe581777a787358
