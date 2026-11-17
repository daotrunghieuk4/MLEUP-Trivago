import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from trivago_price_features import (
    PRICE_FEATURE_SCHEMA, build_price_features, candidate_price_row,
)

def normalize_candidate_list(candidates, target=None):
    """Rank each hotel once, keeping its first impression position for ties."""
    unique = list(dict.fromkeys(int(item) for item in candidates))
    if not unique or any(item <= 0 for item in unique):
        raise ValueError('Candidate list is empty or contains invalid item IDs')
    if target is not None and int(target) not in unique:
        raise ValueError('Target is absent from its candidate list')
    return unique

def mask_data(inputData):
    len_data = [len(nowData) for nowData in inputData]
    max_len = max(len_data)
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len for le in len_data]
    return us_msks, max_len

def complete_data(inputData):
    len_data = [len(nowData) for nowData in inputData]
    max_len = max(len_data)
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) for upois, le in zip(inputData, len_data)]
    us_pois.append([0] * max_len)
    return us_pois

def computeCoSeq(inputData, groups, dates_diff, max_len, si_dc, sample_order):
    length = len(inputData)
    inputData.append([0])
    out_edge_nodes_co, out_edge_weight_co, out_node_edges_co = [], [], []
    out_f_adj_nodes_seq, out_f_adj_weight_seq, out_b_adj_nodes_seq, out_b_adj_weight_seq = [],[],[],[]
    for i in range(length):
        items = np.flipud(np.unique(inputData[i]))
        con_sess_groups = [inputData[id] for id in groups[i]]
        cur_sess = inputData[i]
        con_dates_diff = dates_diff[i]
        edge_nodes_co, edge_weight_co, node_edges_co = generate_adjacency_co(cur_sess, con_sess_groups, con_dates_diff, items, si_dc, max_len)
        out_edge_nodes_co.append(edge_nodes_co)
        out_edge_weight_co.append(edge_weight_co)
        out_node_edges_co.append(node_edges_co)
        con_sess_groups.append(cur_sess)
        con_dates_diff = np.append(con_dates_diff, 0)
        f_adj_nodes_seq, f_adj_weight_seq, b_adj_nodes_seq, b_adj_weight_seq = generate_adjacency_seq(con_sess_groups, con_dates_diff, items, max_len, si_dc, sample_order)
        out_f_adj_nodes_seq.append(f_adj_nodes_seq)
        out_f_adj_weight_seq.append(f_adj_weight_seq)
        out_b_adj_nodes_seq.append(b_adj_nodes_seq)
        out_b_adj_weight_seq.append(b_adj_weight_seq)
    return out_edge_nodes_co, out_edge_weight_co, out_node_edges_co, out_f_adj_nodes_seq, out_f_adj_weight_seq, out_b_adj_nodes_seq, out_b_adj_weight_seq

class Data(Dataset):
    def __init__(self, data, groups_data, sample_order, si_dc, lazy_graphs=False,
                 action_sequences=None, event_sequences=None,
                 candidate_lists=None, candidate_price_records=None,
                 price_scaler=None):
        self.length = len(data[0])
        self.targets = np.asarray(data[1])
        if len(self.targets) != self.length:
            raise ValueError("Item sequences and targets must have equal lengths")
        if action_sequences is not None and event_sequences is not None:
            raise ValueError("Choose one action encoder, not both")
        self.event_sequences = event_sequences
        self.candidate_lists = None
        self.candidate_target_positions = None
        self.candidate_price_records = None
        self.price_scaler = None
        if candidate_lists is not None:
            if len(candidate_lists) != self.length:
                raise ValueError("Candidate lists must align with examples")
            normalized = []
            target_positions = []
            for index, (target, candidates) in enumerate(zip(self.targets, candidate_lists)):
                try:
                    unique_candidates = normalize_candidate_list(candidates, target)
                except ValueError as error:
                    raise ValueError(f'Example {index}: {error}') from error
                normalized.append(unique_candidates)
                target_positions.append(unique_candidates.index(int(target)))
            self.candidate_lists = normalized
            self.candidate_target_positions = np.asarray(target_positions, dtype=np.int64)
        if candidate_price_records is not None:
            if self.candidate_lists is None:
                raise ValueError("Price records require candidate lists")
            if price_scaler is None or price_scaler.get("schema") != PRICE_FEATURE_SCHEMA:
                raise ValueError("Price records require a matching train-fitted scaler")
            offsets = np.asarray(candidate_price_records.get("offsets"), dtype=np.int64)
            item_ids = np.asarray(candidate_price_records.get("item_ids"), dtype=np.int64)
            raw_positions = np.asarray(candidate_price_records.get("raw_positions"), dtype=np.int64)
            prices = np.asarray(candidate_price_records.get("prices"), dtype=np.float64)
            query_ids = np.asarray(candidate_price_records.get("query_ids"), dtype=np.int64)
            target_ids = np.asarray(candidate_price_records.get("target_ids"), dtype=np.int64)
            if (offsets.shape != (self.length + 1,) or offsets[0] != 0
                    or np.any(offsets[1:] <= offsets[:-1])
                    or offsets[-1] != len(item_ids)
                    or len(raw_positions) != len(item_ids) or len(prices) != len(item_ids)
                    or query_ids.shape != (self.length,) or target_ids.shape != (self.length,)):
                raise ValueError("Price candidate records have inconsistent offsets or lengths")
            if len(np.unique(query_ids)) != self.length:
                raise ValueError("Price candidate query IDs must be unique within a split")
            if not np.array_equal(target_ids, self.targets):
                raise ValueError("Price candidate targets do not align with prepared examples")
            finite_prices = prices[np.isfinite(prices)]
            if np.isinf(prices).any() or np.any(finite_prices <= 0):
                raise ValueError("Candidate prices must be positive finite values or NaN")
            for index, candidates in enumerate(self.candidate_lists):
                start, end = int(offsets[index]), int(offsets[index + 1])
                if item_ids[start:end].tolist() != candidates:
                    raise ValueError(f"Price candidate IDs do not align for query {int(query_ids[index])}")
                if np.any(raw_positions[start + 1:end] <= raw_positions[start:end - 1]):
                    raise ValueError("Raw impression positions must remain strictly increasing")
            self.candidate_price_records = dict(
                offsets=offsets, item_ids=item_ids, raw_positions=raw_positions,
                prices=prices, query_ids=query_ids, target_ids=target_ids,
            )
            self.price_scaler = price_scaler
        if event_sequences is not None:
            if len(event_sequences) != self.length:
                raise ValueError("Event sequences must align with examples")
            if any(any(int(token) <= 0 for token in seq) for seq in event_sequences):
                raise ValueError("Event histories may be empty but tokens must be positive")
        if action_sequences is not None:
            if len(action_sequences) != self.length:
                raise ValueError("Action sequences must align with item sequences")
            for index, (items, actions) in enumerate(zip(data[0], action_sequences)):
                if (len(items) != len(actions)
                        or not items):
                    raise ValueError(
                        f"Item/action history {index} is empty or has mismatched lengths"
                    )
                if any(int(item) <= 0 for item in items):
                    raise ValueError(f"Item history {index} contains a padding/invalid item")
                if any(int(action) < 1 or int(action) > 6 for action in actions):
                    raise ValueError(f"Action history {index} must contain IDs 1..6")
        self.empty_current = np.asarray([len(seq) == 0 for seq in data[0]], dtype=bool)
        self.item_sequences = [list(seq) if len(seq) else [0] for seq in data[0]]
        mask, max_len = mask_data(self.item_sequences)
        for index, is_empty in enumerate(self.empty_current):
            if is_empty:
                mask[index] = [0] * max_len
        self.mask = np.asarray(mask)
        re_inputs = complete_data(self.item_sequences)
        self.re_inputs = np.asarray(re_inputs)
        self.max_len = max_len
        self.re_actions = None
        if action_sequences is not None:
            reversed_actions = [
                list(reversed(sequence)) + [0] * (max_len - len(sequence))
                for sequence in action_sequences
            ]
            self.re_actions = np.asarray(reversed_actions, dtype=np.int64)
            if self.re_actions.shape != (self.length, max_len):
                raise ValueError("Padded action histories have an unexpected shape")
            if not np.array_equal(self.re_actions != 0, self.mask != 0):
                raise ValueError("Action padding mask must match item history mask")
        self.groups = np.asarray(groups_data[0])
        self.neg_groups = np.asarray(groups_data[1])
        self.dates_diff = np.asarray(groups_data[2])
        self.lazy_graphs = lazy_graphs
        if lazy_graphs:
            # Full Trivago has too many examples to prebuild every graph.
            self.raw_sequences = list(self.item_sequences) + [[0]]  # group padding ID
            self.sample_order = sample_order
            self.si_dc = si_dc
        else:
            edge_nodes_co, edge_weight_co, node_edges_co, f_adj_nodes_seq, f_adj_weight_seq, b_adj_nodes_seq, b_adj_weight_seq = computeCoSeq(self.item_sequences, self.groups, self.dates_diff, max_len, si_dc, sample_order)
            self.edge_nodes_co = edge_nodes_co
            self.edge_weight_co = edge_weight_co
            self.node_edges_co = node_edges_co
            self.f_adj_nodes_seq = f_adj_nodes_seq
            self.f_adj_weight_seq = f_adj_weight_seq
            self.b_adj_nodes_seq = b_adj_nodes_seq
            self.b_adj_weight_seq = b_adj_weight_seq

    def __getitem__(self, index):
        re_inputs = self.re_inputs[index]
        node = np.flipud(np.unique(re_inputs))
        items = list(node) + (self.max_len - len(node)) * [0]
        alias_re_inputs = [np.where(node == i)[0][0] for i in re_inputs]
        re_groups = self.re_inputs[self.groups[index]]
        re_neg_groups = self.re_inputs[self.neg_groups[index]]
        mask_groups = np.where(re_groups != 0, 1, 0)
        if self.lazy_graphs:
            current = self.raw_sequences[index]
            real_groups = [(self.raw_sequences[group_index], day_diff)
                           for group_index, day_diff in zip(
                               self.groups[index], self.dates_diff[index])
                           if group_index < self.length]
            group_sequences = [sequence for sequence, _ in real_groups]
            group_dates = np.asarray([day_diff for _, day_diff in real_groups])
            unique_items = np.flipud(np.unique(current))
            edge_nodes_co, edge_weight_co, node_edges_co = generate_adjacency_co(
                current, group_sequences, group_dates, unique_items,
                self.si_dc, self.max_len,
            )
            f_adj_nodes, f_adj_weights, b_adj_nodes, b_adj_weights = generate_adjacency_seq(
                group_sequences + [current],
                np.append(group_dates, 0), unique_items,
                self.max_len, self.si_dc, self.sample_order,
            )
        else:
            edge_nodes_co = self.edge_nodes_co[index]
            edge_weight_co = self.edge_weight_co[index]
            node_edges_co = self.node_edges_co[index]
            f_adj_nodes = self.f_adj_nodes_seq[index]
            f_adj_weights = self.f_adj_weight_seq[index]
            b_adj_nodes = self.b_adj_nodes_seq[index]
            b_adj_weights = self.b_adj_weight_seq[index]
        sample = (alias_re_inputs, items, self.mask[index], self.targets[index], \
               re_inputs, re_groups, re_neg_groups, mask_groups, \
               edge_nodes_co, edge_weight_co, node_edges_co, \
               f_adj_nodes, f_adj_weights, b_adj_nodes, b_adj_weights)
        if self.re_actions is not None:
            sample += (self.re_actions[index],)
        if self.event_sequences is not None:
            events = torch.as_tensor(self.event_sequences[index], dtype=torch.long)
            sample += (events, torch.ones_like(events))
        if self.candidate_lists is not None:
            candidates = torch.as_tensor(self.candidate_lists[index], dtype=torch.long)
            sample += (candidates, torch.ones_like(candidates),
                       torch.tensor(self.candidate_target_positions[index], dtype=torch.long))
        if self.candidate_price_records is not None:
            prices = candidate_price_row(self.candidate_price_records, index)
            price_features = build_price_features(prices, self.price_scaler)
            sample += (torch.as_tensor(price_features, dtype=torch.float32),)
        return sample

    def __len__(self):
        return self.length

def collate_fn(batch):
    if not batch or len({len(sample) for sample in batch}) != 1:
        raise ValueError("Every sample in a batch must share its data-field contract")
    sample_size = len(batch[0])
    if sample_size not in (15, 16, 17, 19, 20):
        raise ValueError(f"Expected 15, 16, 17, 19 or 20 sample fields, got {sample_size}")
    legacy = [sample[:15] for sample in batch]
    (alias_re_inputs, items, mask, targets, re_inputs, re_groups, re_neg_groups,
     mask_groups, edge_nodes_co, edge_weight_co, node_edges_co,
     f_adjacency_nodes_seq, f_adjacency_weight_seq,
     b_adjacency_nodes_seq, b_adjacency_weight_seq) = zip(*legacy)
    edge_weight_co = pad_sequence(edge_weight_co, batch_first=True)
    max_edge_width = edge_weight_co.shape[1]
    edge_nodes_co = [torch.cat((edge, torch.zeros(edge.shape[0], max_edge_width-edge.shape[1])),1) for edge in edge_nodes_co]
    edge_nodes_co = pad_sequence(edge_nodes_co, batch_first=True).permute(0, 2, 1)
    node_edges_co = pad_sequence(node_edges_co, batch_first=True).permute(0, 2, 1)
    f_adjacency_nodes_seq = pad_sequence(f_adjacency_nodes_seq, batch_first=True).permute(0, 2, 3, 1)
    f_adjacency_weight_seq = pad_sequence(f_adjacency_weight_seq, batch_first=True).permute(0, 2, 3, 1)
    b_adjacency_nodes_seq = pad_sequence(b_adjacency_nodes_seq, batch_first=True).permute(0, 2, 3, 1)
    b_adjacency_weight_seq = pad_sequence(b_adjacency_weight_seq, batch_first=True).permute(0, 2, 3, 1)
    result = (torch.tensor(np.asarray(alias_re_inputs)), torch.tensor(items), torch.tensor(np.asarray(mask)), torch.tensor(targets), \
           torch.tensor(np.asarray(re_inputs)), torch.tensor(np.asarray(re_groups)), torch.tensor(np.asarray(re_neg_groups)), torch.tensor(np.asarray(mask_groups)), \
           edge_nodes_co, edge_weight_co, node_edges_co, f_adjacency_nodes_seq, f_adjacency_weight_seq, b_adjacency_nodes_seq, b_adjacency_weight_seq)
    if sample_size == 16:
        action_batch = torch.as_tensor(
            np.asarray([sample[15] for sample in batch]), dtype=torch.long
        )
        result += (action_batch,)
    if sample_size == 17:
        event_offset = 15
        event_sequences = [s[event_offset] for s in batch]
        event_width = max(1, max((len(seq) for seq in event_sequences), default=0))
        event_batch = torch.zeros((len(batch), event_width), dtype=torch.long)
        event_mask = torch.zeros_like(event_batch)
        for row, events in enumerate(event_sequences):
            if len(events):
                event_batch[row, :len(events)] = events
                event_mask[row, :len(events)] = 1
        result += (event_batch, event_mask)
    if sample_size == 19:
        action_batch = torch.as_tensor(
            np.asarray([sample[15] for sample in batch]), dtype=torch.long
        )
        candidate_sequences = [sample[16] for sample in batch]
        candidate_width = max(len(sequence) for sequence in candidate_sequences)
        candidate_batch = torch.zeros((len(batch), candidate_width), dtype=torch.long)
        candidate_mask = torch.zeros_like(candidate_batch)
        for row, candidates in enumerate(candidate_sequences):
            candidate_batch[row, :len(candidates)] = candidates
            candidate_mask[row, :len(candidates)] = 1
        candidate_targets = torch.as_tensor(
            [sample[18] for sample in batch], dtype=torch.long
        )
        result += (action_batch, candidate_batch, candidate_mask, candidate_targets)
    if sample_size == 20:
        action_batch = torch.as_tensor(
            np.asarray([sample[15] for sample in batch]), dtype=torch.long
        )
        candidate_sequences = [sample[16] for sample in batch]
        candidate_width = max(len(sequence) for sequence in candidate_sequences)
        candidate_batch = torch.zeros((len(batch), candidate_width), dtype=torch.long)
        candidate_mask = torch.zeros_like(candidate_batch)
        price_width = batch[0][19].shape[-1]
        price_batch = torch.zeros((len(batch), candidate_width, price_width), dtype=torch.float32)
        for row, sample in enumerate(batch):
            candidates, features = sample[16], sample[19]
            length = len(candidates)
            candidate_batch[row, :length] = candidates
            candidate_mask[row, :length] = 1
            if features.shape != (length, price_width):
                raise ValueError("Price features must align with each candidate sequence")
            price_batch[row, :length] = features
        candidate_targets = torch.as_tensor(
            [sample[18] for sample in batch], dtype=torch.long
        )
        result += (action_batch, candidate_batch, candidate_mask, candidate_targets,
                   price_batch)
    return result

def generate_adjacency_co(cur_sess, neig_re_group, neig_dates_diff, items, si_dc, max_len):
    node_edges_co_dict = {}
    for node in items:
        node_edges_co_dict[node] = []
    edge_nodes_co = []
    edge_nodes_co.append(torch.tensor([0]))
    id = 1
    for edge in neig_re_group:
        edge = np.unique(edge)
        edge_nodes_co.append(torch.tensor(edge))
        for node in edge:
            if node in items:
                node_edges_co_dict[node].append(id)
        id += 1
    m = id
    cur_edge_nodes_co = []
    le = len(cur_sess)
    if le <= 20:
        for i in np.arange(1, le + 1):
            for j in np.arange(le - i + 1):
                cur_edge_nodes_co.append(np.flipud(np.unique(cur_sess[j:j + i])))
    else:
        for i in np.arange(1,19):
            for j in np.arange(le-i+1):
                cur_edge_nodes_co.append(np.flipud(np.unique(cur_sess[j:j + i])))
        cur_edge_nodes_co.append(cur_sess)
    cur_edge_nodes_co = set(map(tuple, cur_edge_nodes_co))  # 去掉重复超边
    for edge in cur_edge_nodes_co:
        edge_nodes_co.append(torch.tensor(edge))
        for node in edge:
            node_edges_co_dict[node].append(id)
        id += 1
    edge_nodes_co = pad_sequence(edge_nodes_co, batch_first=True).permute(1, 0)

    edge_weight_co = [0]
    for ddates in neig_dates_diff:
        edge_weight_co.append(1/pow(si_dc, ddates))
    for _ in np.arange(id-m):
        edge_weight_co.append(1)
    edge_weight_co = torch.tensor(edge_weight_co)

    node_edges_co = []
    for value in node_edges_co_dict.values():
        node_edges_co.append(torch.tensor(value))
    for _ in np.arange(max_len - len(items)):
        node_edges_co.append(torch.tensor([0]))
    node_edges_co = pad_sequence(node_edges_co, batch_first=True).permute(1, 0)
    return edge_nodes_co, edge_weight_co, node_edges_co

def generate_adjacency_seq(temp_re_group, temp_dates_diff, items, max_len, si_dc, sample_order):
    f_temp_seq = {}
    b_temp_seq = {}
    for seq, ddates in zip(temp_re_group, temp_dates_diff):
        for i in np.arange(len(seq) - 1):
            item1 = seq[i]
            item2 = seq[i + 1]
            if item2 in f_temp_seq.keys():
                if item1 in f_temp_seq[item2][0]:
                    index = f_temp_seq[item2][0].index(item1)
                    f_temp_seq[item2][1][index] += 1 / pow(si_dc, ddates)
                else:
                    f_temp_seq[item2][0].append(item1)
                    f_temp_seq[item2][1].append(1 / pow(si_dc, ddates))
            else:
                f_temp_seq[item2] = [[item1], [1 / pow(si_dc, ddates)]]
            if item1 in b_temp_seq.keys():
                if item2 in b_temp_seq[item1][0]:
                    index = b_temp_seq[item1][0].index(item2)
                    b_temp_seq[item1][1][index] += 1 / pow(si_dc, ddates)
                else:
                    b_temp_seq[item1][0].append(item2)
                    b_temp_seq[item1][1].append(1 / pow(si_dc, ddates))
            else:
                b_temp_seq[item1] = [[item2], [1 / pow(si_dc, ddates)]]
    f_adjacency_nodes_seq, f_adjacency_weight_seq = compute_all_order_seq(f_temp_seq, items, sample_order, max_len)
    b_adjacency_nodes_seq, b_adjacency_weight_seq = compute_all_order_seq(b_temp_seq, items, sample_order, max_len)
    return f_adjacency_nodes_seq, f_adjacency_weight_seq, b_adjacency_nodes_seq, b_adjacency_weight_seq

def compute_all_order_seq(temp_seq, items, sample_order, max_len):
    all_adjacency_nodes_seq = []
    all_adjacency_weight_seq = []
    for item in items:
        adjacency_nodes_seq = []
        adjacency_weight_seq = []
        if item in temp_seq.keys():
            nodes = temp_seq[item][0]
            adjacency_nodes_seq.append(torch.tensor(nodes))
            adjacency_weight_seq.append(torch.tensor(temp_seq[item][1]))
            all_nodes = set(nodes)
            for _ in np.arange(1, sample_order):
                seq_k_n = []
                seq_k_w = []
                for n in nodes:
                    if n in temp_seq.keys():
                        for nn in temp_seq[n][0]:
                            ww = temp_seq[n][1][temp_seq[n][0].index(nn)]
                            if nn in seq_k_n:
                                seq_k_w[seq_k_n.index(nn)] += ww
                            else:
                                seq_k_n.append(nn)
                                seq_k_w.append(ww)
                temp_nodes = set(seq_k_n)
                nodes = temp_nodes - all_nodes
                if len(nodes) == 0:
                    adjacency_nodes_seq.append(torch.tensor([0]))
                    adjacency_weight_seq.append(torch.tensor([0]))
                    continue
                already_nodes = list(temp_nodes & all_nodes)
                for al_nodes in already_nodes:
                    index = seq_k_n.index(al_nodes)
                    del seq_k_n[index]
                    del seq_k_w[index]
                adjacency_nodes_seq.append(torch.tensor(seq_k_n))
                adjacency_weight_seq.append(torch.tensor(seq_k_w))
                all_nodes = temp_nodes | all_nodes
        else:
            for _ in np.arange(sample_order):
                adjacency_nodes_seq.append(torch.tensor([0]))
                adjacency_weight_seq.append(torch.tensor([0]))
        adjacency_nodes_seq = pad_sequence(adjacency_nodes_seq, batch_first=True).permute(1, 0)
        adjacency_weight_seq = pad_sequence(adjacency_weight_seq, batch_first=True).permute(1, 0)
        all_adjacency_nodes_seq.append(adjacency_nodes_seq)
        all_adjacency_weight_seq.append(adjacency_weight_seq)
    le = all_adjacency_nodes_seq[0].shape[0]
    for _ in np.arange(max_len-len(items)):
        all_adjacency_nodes_seq.append(torch.Tensor(le, sample_order).fill_(0))
        all_adjacency_weight_seq.append(torch.Tensor(le, sample_order).fill_(0))
    all_adjacency_nodes_seq = pad_sequence(all_adjacency_nodes_seq, batch_first=True).permute(1, 0, 2)
    all_adjacency_weight_seq = pad_sequence(all_adjacency_weight_seq, batch_first=True).permute(1, 0, 2)
    return all_adjacency_nodes_seq, all_adjacency_weight_seq
