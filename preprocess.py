<<<<<<< HEAD
import os
import json
import re
import argparse
from transformers import BertTokenizer
from tqdm import tqdm
from constants import *

#处理ann文件格式
def aligment_ann(original, newtext, ann_file, offset):
    original = original.lower()
    newtext = newtext.lower()
    annotation = []
    relations=[]
    dict={}
    terms = {}
    ends = {}
    for line in open(ann_file):
        if line[0]=='T':
            annots = line.strip().split("\t")
            typeregion = annots[1].split(" ")
            start = eval(typeregion[1]) - offset
            end = eval(typeregion[2]) - offset
            if not start in terms:
                terms[start] = []
            if not end in ends:
                ends[end] = []
            if len(annots) == 3:
                terms[start].append([start, end, annots[0], typeregion[0], annots[2]])
            else:
                terms[start].append([start, end, annots[0], typeregion[0], ""])
            ends[end].append(start)
        else:
            annotation.append(line)
            rel_id, rel_type, rel_e1, rel_e2 = line.strip().split()
            relations.append([rel_id,rel_type,rel_e1[5:],rel_e2[5:]])

    orgidx = 0
    newidx = 0
    orglen = len(original)
    newlen = len(newtext)

    while orgidx < orglen and newidx < newlen:
        if original[orgidx] == newtext[newidx]:
            orgidx += 1
            newidx += 1
        elif newtext[newidx] == '\n':
            newidx += 1
        elif original[orgidx] == '\n':
            orgidx += 1
        elif newtext[newidx] == ' ':
            newidx += 1
        elif original[orgidx] == ' ':
            orgidx += 1
        elif newtext[newidx] == '\t':
            newidx += 1
        elif original[orgidx] == '\t':
            orgidx += 1
        elif newtext[newidx] == '.':
            newidx += 1
        else:
            assert False, "%d\t$%s$\t$%s$" % (
                orgidx, original[orgidx:orgidx + 20], newtext[newidx:newidx + 20])
        if orgidx in terms:
            for l in terms[orgidx]:
                l[0] = newidx
        if orgidx in ends:
            for start in ends[orgidx]:
                for l in terms[start]:
                    if l[1] == orgidx:
                        l[1] = newidx
            del ends[orgidx]

    entities = []
    i = 0
    for ts in terms.values():
        for term in ts:
            # term = [start, end, annots[0], typeregion[0], annots[2]]
            if term[4] == "":
                entities.append([term[3], term[0], term[1], newtext[term[0]:term[1]]])
            else:
                assert newtext[term[0]:term[1]].replace(" ", "").replace('\n', "").\
                           replace("&AMP;", "&").replace("&amp;", "&") == \
                    term[4].replace(" ", "").lower(
                ), newtext[term[0]:term[1]] + "<=>" + term[4]
                entities.append([term[3], term[0], term[1],
                                 newtext[term[0]:term[1]].replace("\n", " ")])
            dict[term[2]] = i
            i += 1

    relations1 = []
    for rel in relations:
        _, rel_type, rel_e1, rel_e2 = rel
        rel_e1_idx = dict[rel_e1]
        rel_e2_idx = dict[rel_e2]
        relations1.append([rel_type, rel_e1_idx, rel_e2_idx])
    relations.clear()
    relations=relations1

    # entities : list of [the types of entities, start, end, entities]
    # relations: list of [the types of relations, entity1_idx, entity2_idx]
    return entities, relations


def passage_blocks(txt, window_size, overlap):
    blocks = []
    regions = []
    for i in range(0, len(txt), window_size-overlap):
        b = txt[i:i+window_size]
        blocks.append(b)
        regions.append((i, i+window_size))
    return blocks, regions


def get_block_er(txt, entities, relations, window_size, overlap, tokenizer):
    blocks, block_range = passage_blocks(txt, window_size, overlap)
    ber = [[[], [], []] for i in range(len(block_range))]
    e_dict = {}
    for i, (s, e) in enumerate(block_range):
        es = []
        for j, (entity_type, start, end, entity_str) in enumerate(entities):
            if start >= s and end <= e:
                nstart, nend = start-s, end-s
                if tokenizer.convert_tokens_to_string(blocks[i][nstart:nend]) == entity_str:
                    es.append((entity_type, nstart, nend, entity_str))
                    e_dict[j] = e_dict.get(j, [])+[i]
                else:
                    print("The entity string and its corresponding index are inconsistent")
        ber[i][0] = blocks[i]
        ber[i][1].extend(es)
    for r, e1i, e2i in relations:
        if e1i not in e_dict or e2i not in e_dict:
            print("Entity lost due to sliding window")
            continue
        i1s, i2s = e_dict[e1i], e_dict[e2i]
        intersec = set.intersection(set(i1s), set(i2s))
        if intersec:
            for i in intersec:
                t1, s1, e1, es1 = entities[e1i][0], entities[e1i][1] - \
                    block_range[i][0], entities[e1i][2] - \
                    block_range[i][0], entities[e1i][3]
                t2, s2, e2, es2 = entities[e2i][0], entities[e2i][1] - \
                    block_range[i][0], entities[e2i][2] - \
                    block_range[i][0], entities[e2i][3]
                ber[i][2].append((r, (t1, s1, e1, es1), (t2, s2, e2, es2)))
        else:
            print("The two entities of the relationship are not on the same sentence")
    return ber


def get_question(question_templates, head_entity, relation_type=None, end_entity_type=None):
    if relation_type == None:
        question = question_templates['qa_turn1'][head_entity[0]] if isinstance(
            head_entity, tuple) else question_templates['qa_turn1'][head_entity]
    else:
        question = question_templates['qa_turn2'][str(
            (head_entity[0], relation_type, end_entity_type))]
        question = question.replace('XXX', head_entity[3])
    return question


def block2qas(ber, title="", threshold=1, max_distance=45):
    entities = ace2005_entities
    relations = ace2005_relations
    idx1s = ace2005_idx1
    idx2s = ace2005_idx2
    dist = ace2005_dist
    question_templates = ace2005_question_templates

    block, ents, relas = ber
    res = {'context': block, 'title': title}
    # QA turn 1
    dict1 = {k: get_question(question_templates, k) for k in entities}
    # dict1: entity_type: question
    qat1 = {dict1[k]: [] for k in dict1}
    # qat1: question: entity
    for en in ents:
        q = dict1[en[0]]
        qat1[q].append(en)
    # QA turn 2
    dict2 = {(rel[1], rel[0], rel[2][0]): [] for rel in relas}
    if max_distance > 0:
        for rel in relas:
            dict2[(rel[1], rel[0], rel[2][0])].append(rel[2])
        qat2 = []
        ents1 = sorted(ents, key=lambda x: x[1])
        for i, ent1 in enumerate(ents1):
            start = ent1[1]
            qas = {}
            for j, ent2 in enumerate(ents1[i+1:], i+1):
                if ent2[1] > start+max_distance:
                    break
                else:
                    head_type, end_type = ent1[0], ent2[0]
                    for rel_type in relations:
                        idx1, idx2 = idx1s[head_type], idx2s[(
                            rel_type, end_type)]
                        if dist[idx1][idx2] >= threshold:
                            k = (ent1, rel_type, end_type)
                            q = get_question(
                                question_templates, ent1, rel_type, end_type)
                            qas[q] = dict2.get(k, [])
            qat2.append({"head_entity": ent1, "qas": qas})

    else:
        for rel in relas:
            dict2[(rel[1], rel[0], rel[2][0])].append(rel[2])
        qat2 = []
        for ent in ents:
            qas = {}
            for rel_type in relations:
                for ent_type in entities:
                    k = (ent, rel_type, ent_type)
                    idx1, idx2 = idx1s[ent[0]], idx2s[(rel_type, ent_type)]
                    if dist[idx1][idx2] >= threshold:
                        q = get_question(question_templates,
                                         ent, rel_type, ent_type)
                        qas[q] = dict2.get(k, [])
            qat2.append({'head_entity': ent, "qas": qas})
    qas = [qat1, qat2]
    res["qa_pairs"] = qas
    return res


def char_to_wordpiece(passage, entities, tokenizer):
    entities1 = []
    tpassage = tokenizer.tokenize(passage)
    for ent in entities:
        ent_type, start, end, ent_str = ent
        s = tokenizer.tokenize(passage[:start])
        start1 = len(s)
        ent_str1 = tokenizer.tokenize(ent_str)
        end1 = start1 + len(ent_str1)
        ent_str2 = tokenizer.convert_tokens_to_string(ent_str1)
        entities1.append((ent_type, start1, end1, ent_str2))
    return entities1


def process(data_dir, output_dir, tokenizer, is_test, window_size, overlap, threshold=1, max_distance=45):
    ann_files = []
    txt_files = []
    data = []
    for f in os.listdir(data_dir):
        if f.endswith('.txt'):
            txt_files.append(os.path.join(data_dir, f))
        elif f.endswith('.ann'):
            ann_files.append(os.path.join(data_dir, f))
    ann_files = sorted(ann_files)
    txt_files = sorted(txt_files)
    for ann_path, txt_path in tqdm(zip(ann_files, txt_files), total=len(ann_files)):
        with open(txt_path, encoding='utf-8') as f:
            raw_txt = f.read()
            txt = [t for t in raw_txt.split('\n') if t.strip()]
        title = re.search('[A-Za-z_]+[A-Za-z]', txt[0]).group().split('-')+txt[1].strip().split()
        title = " ".join(title)
        title = tokenizer.tokenize(title)
        ntxt = ' '.join(txt[3:])
        ntxt1 = tokenizer.tokenize(ntxt)
        ntxt2 = tokenizer.convert_tokens_to_string(ntxt1)
        offset = raw_txt.index(txt[3])
        entities, relations = aligment_ann(
            raw_txt[offset:], ntxt2, ann_path, offset)
        entities = char_to_wordpiece(ntxt2, entities, tokenizer)
        if is_test:
            data.append({"title": title, "passage": ntxt1,
                         "entities": entities, "relations": relations})
        else:
            block_er = get_block_er(ntxt1, entities, relations, window_size, overlap, tokenizer)
            for ber in block_er:
                data.append(block2qas(ber, title, threshold, max_distance))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, os.path.split(data_dir)[-1]+".json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/ACE2005/train')
    # parser.add_argument("--data_dir", default='./data/ACE2005/dev')
    # parser.add_argument("--data_dir", default='./data/ACE2005/test')
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=15)
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--output_base_dir",
                        default="./data/cleaned_data/ACE2005")
    parser.add_argument("--max_distance", type=int, default=45,
                        help="used to filter relations by distance from the head entity")
    args = parser.parse_args()
    if not args.is_test:
        output_dir = "{}/{}_overlap_{}_window_{}_threshold_{}_max_distance_{}".format(args.output_base_dir, os.path.split(
            args.pretrained_model_path)[-1], args.overlap, args.window_size, args.threshold, args.max_distance)
    else:
        output_dir = args.output_base_dir
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    process(args.data_dir, output_dir, tokenizer, args.is_test,
            args.window_size, args.overlap, args.threshold, args.max_distance)
=======
import os
import json
import re
import argparse
from transformers import BertTokenizer
from tqdm import tqdm
from constants import *

#处理ann文件格式
def aligment_ann(original, newtext, ann_file, offset):
    original = original.lower()
    newtext = newtext.lower()
    annotation = []
    relations=[]
    dict={}
    terms = {}
    ends = {}
    for line in open(ann_file):
        if line[0]=='T':
            annots = line.strip().split("\t")
            typeregion = annots[1].split(" ")
            start = eval(typeregion[1]) - offset
            end = eval(typeregion[2]) - offset
            if not start in terms:
                terms[start] = []
            if not end in ends:
                ends[end] = []
            if len(annots) == 3:
                terms[start].append([start, end, annots[0], typeregion[0], annots[2]])
            else:
                terms[start].append([start, end, annots[0], typeregion[0], ""])
            ends[end].append(start)
        else:
            annotation.append(line)
            rel_id, rel_type, rel_e1, rel_e2 = line.strip().split()
            relations.append([rel_id,rel_type,rel_e1[5:],rel_e2[5:]])

    orgidx = 0
    newidx = 0
    orglen = len(original)
    newlen = len(newtext)

    while orgidx < orglen and newidx < newlen:
        if original[orgidx] == newtext[newidx]:
            orgidx += 1
            newidx += 1
        elif newtext[newidx] == '\n':
            newidx += 1
        elif original[orgidx] == '\n':
            orgidx += 1
        elif newtext[newidx] == ' ':
            newidx += 1
        elif original[orgidx] == ' ':
            orgidx += 1
        elif newtext[newidx] == '\t':
            newidx += 1
        elif original[orgidx] == '\t':
            orgidx += 1
        elif newtext[newidx] == '.':
            newidx += 1
        else:
            assert False, "%d\t$%s$\t$%s$" % (
                orgidx, original[orgidx:orgidx + 20], newtext[newidx:newidx + 20])
        if orgidx in terms:
            for l in terms[orgidx]:
                l[0] = newidx
        if orgidx in ends:
            for start in ends[orgidx]:
                for l in terms[start]:
                    if l[1] == orgidx:
                        l[1] = newidx
            del ends[orgidx]

    entities = []
    i = 0
    for ts in terms.values():
        for term in ts:
            # term = [start, end, annots[0], typeregion[0], annots[2]]
            if term[4] == "":
                entities.append([term[3], term[0], term[1], newtext[term[0]:term[1]]])
            else:
                assert newtext[term[0]:term[1]].replace(" ", "").replace('\n', "").\
                           replace("&AMP;", "&").replace("&amp;", "&") == \
                    term[4].replace(" ", "").lower(
                ), newtext[term[0]:term[1]] + "<=>" + term[4]
                entities.append([term[3], term[0], term[1],
                                 newtext[term[0]:term[1]].replace("\n", " ")])
            dict[term[2]] = i
            i += 1

    relations1 = []
    for rel in relations:
        _, rel_type, rel_e1, rel_e2 = rel
        rel_e1_idx = dict[rel_e1]
        rel_e2_idx = dict[rel_e2]
        relations1.append([rel_type, rel_e1_idx, rel_e2_idx])
    relations.clear()
    relations=relations1

    # entities : list of [the types of entities, start, end, entities]
    # relations: list of [the types of relations, entity1_idx, entity2_idx]
    return entities, relations


def passage_blocks(txt, window_size, overlap):
    blocks = []
    regions = []
    for i in range(0, len(txt), window_size-overlap):
        b = txt[i:i+window_size]
        blocks.append(b)
        regions.append((i, i+window_size))
    return blocks, regions


def get_block_er(txt, entities, relations, window_size, overlap, tokenizer):
    blocks, block_range = passage_blocks(txt, window_size, overlap)
    ber = [[[], [], []] for i in range(len(block_range))]
    e_dict = {}
    for i, (s, e) in enumerate(block_range):
        es = []
        for j, (entity_type, start, end, entity_str) in enumerate(entities):
            if start >= s and end <= e:
                nstart, nend = start-s, end-s
                if tokenizer.convert_tokens_to_string(blocks[i][nstart:nend]) == entity_str:
                    es.append((entity_type, nstart, nend, entity_str))
                    e_dict[j] = e_dict.get(j, [])+[i]
                else:
                    print("The entity string and its corresponding index are inconsistent")
        ber[i][0] = blocks[i]
        ber[i][1].extend(es)
    for r, e1i, e2i in relations:
        if e1i not in e_dict or e2i not in e_dict:
            print("Entity lost due to sliding window")
            continue
        i1s, i2s = e_dict[e1i], e_dict[e2i]
        intersec = set.intersection(set(i1s), set(i2s))
        if intersec:
            for i in intersec:
                t1, s1, e1, es1 = entities[e1i][0], entities[e1i][1] - \
                    block_range[i][0], entities[e1i][2] - \
                    block_range[i][0], entities[e1i][3]
                t2, s2, e2, es2 = entities[e2i][0], entities[e2i][1] - \
                    block_range[i][0], entities[e2i][2] - \
                    block_range[i][0], entities[e2i][3]
                ber[i][2].append((r, (t1, s1, e1, es1), (t2, s2, e2, es2)))
        else:
            print("The two entities of the relationship are not on the same sentence")
    return ber


def get_question(question_templates, head_entity, relation_type=None, end_entity_type=None):
    if relation_type == None:
        question = question_templates['qa_turn1'][head_entity[0]] if isinstance(
            head_entity, tuple) else question_templates['qa_turn1'][head_entity]
    else:
        question = question_templates['qa_turn2'][str(
            (head_entity[0], relation_type, end_entity_type))]
        question = question.replace('XXX', head_entity[3])
    return question


def block2qas(ber, title="", threshold=1, max_distance=45):
    entities = ace2005_entities
    relations = ace2005_relations
    idx1s = ace2005_idx1
    idx2s = ace2005_idx2
    dist = ace2005_dist
    question_templates = ace2005_question_templates

    block, ents, relas = ber
    res = {'context': block, 'title': title}
    # QA turn 1
    dict1 = {k: get_question(question_templates, k) for k in entities}
    # dict1: entity_type: question
    qat1 = {dict1[k]: [] for k in dict1}
    # qat1: question: entity
    for en in ents:
        q = dict1[en[0]]
        qat1[q].append(en)
    # QA turn 2
    dict2 = {(rel[1], rel[0], rel[2][0]): [] for rel in relas}
    if max_distance > 0:
        for rel in relas:
            dict2[(rel[1], rel[0], rel[2][0])].append(rel[2])
        qat2 = []
        ents1 = sorted(ents, key=lambda x: x[1])
        for i, ent1 in enumerate(ents1):
            start = ent1[1]
            qas = {}
            for j, ent2 in enumerate(ents1[i+1:], i+1):
                if ent2[1] > start+max_distance:
                    break
                else:
                    head_type, end_type = ent1[0], ent2[0]
                    for rel_type in relations:
                        idx1, idx2 = idx1s[head_type], idx2s[(
                            rel_type, end_type)]
                        if dist[idx1][idx2] >= threshold:
                            k = (ent1, rel_type, end_type)
                            q = get_question(
                                question_templates, ent1, rel_type, end_type)
                            qas[q] = dict2.get(k, [])
            qat2.append({"head_entity": ent1, "qas": qas})

    else:
        for rel in relas:
            dict2[(rel[1], rel[0], rel[2][0])].append(rel[2])
        qat2 = []
        for ent in ents:
            qas = {}
            for rel_type in relations:
                for ent_type in entities:
                    k = (ent, rel_type, ent_type)
                    idx1, idx2 = idx1s[ent[0]], idx2s[(rel_type, ent_type)]
                    if dist[idx1][idx2] >= threshold:
                        q = get_question(question_templates,
                                         ent, rel_type, ent_type)
                        qas[q] = dict2.get(k, [])
            qat2.append({'head_entity': ent, "qas": qas})
    qas = [qat1, qat2]
    res["qa_pairs"] = qas
    return res


def char_to_wordpiece(passage, entities, tokenizer):
    entities1 = []
    tpassage = tokenizer.tokenize(passage)
    for ent in entities:
        ent_type, start, end, ent_str = ent
        s = tokenizer.tokenize(passage[:start])
        start1 = len(s)
        ent_str1 = tokenizer.tokenize(ent_str)
        end1 = start1 + len(ent_str1)
        ent_str2 = tokenizer.convert_tokens_to_string(ent_str1)
        entities1.append((ent_type, start1, end1, ent_str2))
    return entities1


def process(data_dir, output_dir, tokenizer, is_test, window_size, overlap, threshold=1, max_distance=45):
    ann_files = []
    txt_files = []
    data = []
    for f in os.listdir(data_dir):
        if f.endswith('.txt'):
            txt_files.append(os.path.join(data_dir, f))
        elif f.endswith('.ann'):
            ann_files.append(os.path.join(data_dir, f))
    ann_files = sorted(ann_files)
    txt_files = sorted(txt_files)
    for ann_path, txt_path in tqdm(zip(ann_files, txt_files), total=len(ann_files)):
        with open(txt_path, encoding='utf-8') as f:
            raw_txt = f.read()
            txt = [t for t in raw_txt.split('\n') if t.strip()]
        title = re.search('[A-Za-z_]+[A-Za-z]', txt[0]).group().split('-')+txt[1].strip().split()
        title = " ".join(title)
        title = tokenizer.tokenize(title)
        ntxt = ' '.join(txt[3:])
        ntxt1 = tokenizer.tokenize(ntxt)
        ntxt2 = tokenizer.convert_tokens_to_string(ntxt1)
        offset = raw_txt.index(txt[3])
        entities, relations = aligment_ann(
            raw_txt[offset:], ntxt2, ann_path, offset)
        entities = char_to_wordpiece(ntxt2, entities, tokenizer)
        if is_test:
            data.append({"title": title, "passage": ntxt1,
                         "entities": entities, "relations": relations})
        else:
            block_er = get_block_er(ntxt1, entities, relations, window_size, overlap, tokenizer)
            for ber in block_er:
                data.append(block2qas(ber, title, threshold, max_distance))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, os.path.split(data_dir)[-1]+".json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/ACE2005/train')
    # parser.add_argument("--data_dir", default='./data/ACE2005/dev')
    # parser.add_argument("--data_dir", default='./data/ACE2005/test')
    parser.add_argument("--window_size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=15)
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--output_base_dir",
                        default="./data/cleaned_data/ACE2005")
    parser.add_argument("--max_distance", type=int, default=45,
                        help="used to filter relations by distance from the head entity")
    args = parser.parse_args()
    if not args.is_test:
        output_dir = "{}/{}_overlap_{}_window_{}_threshold_{}_max_distance_{}".format(args.output_base_dir, os.path.split(
            args.pretrained_model_path)[-1], args.overlap, args.window_size, args.threshold, args.max_distance)
    else:
        output_dir = args.output_base_dir
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    process(args.data_dir, output_dir, tokenizer, args.is_test,
            args.window_size, args.overlap, args.threshold, args.max_distance)
>>>>>>> 4cf9ab73d9db67401f3e6a17a45675eb711809a2
