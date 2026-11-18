import math
import torch
from torch import nn
from torch.nn import Module

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class FL(Module):
    def __init__(self, num_f, emb_size, n_node):
        super(FL, self).__init__()
        self.num_f = num_f
        self.n_node = n_node
        self.emb_size = emb_size
        self.embedding_f = nn.Embedding(self.num_f, self.emb_size)
        self.u = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W = nn.Linear(self.emb_size * 2, self.emb_size)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout70 = nn.Dropout(0.7)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, adjacency_fi, embedding_i):
        neigh_embedding = embedding_i[adjacency_fi]
        agg_embedding = self.agg_feature(adjacency_fi, neigh_embedding)
        new_embedding_f = self.upd_feature(self.embedding_f.weight, agg_embedding)
        return new_embedding_f

    def agg_feature(self, adjacency_fi, neigh_embedding):
        attention_mask = torch.where(adjacency_fi != 0, 0, -10000)
        alpha = torch.matmul(neigh_embedding, self.u).squeeze(-1)
        alpha = alpha + attention_mask
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        neigh_embedding = torch.sum(alpha * neigh_embedding, 1)
        return self.dropout70(neigh_embedding)

    def upd_feature(self, emb1, emb2):
        gate = torch.sigmoid(self.W(torch.cat([emb1, emb2], 1)))
        new_embedding = gate * emb1 + (1 - gate) * emb2
        return self.dropout50(new_embedding)

class MLEUP(Module):
    def __init__(self, opt, n_node, all_feature, adjacency_fi, adjacency_if):
        super(MLEUP, self).__init__()
        self.m = opt.m
        self.emb_size = opt.emb_size
        self.batch_size = opt.batch_size
        self.n_node = n_node
        self.embedding_i = nn.Embedding(self.n_node, self.emb_size)
        self.f_embedding_pos = nn.Embedding(200, self.emb_size)
        self.b_embedding_pos = nn.Embedding(200, self.emb_size)

        self.sample_order = opt.sample_order
        self.adjacency_fi = adjacency_fi
        self.adjacency_if = adjacency_if
        self.all_feature = all_feature
        self.n_feature = len(all_feature)
        # Only the Trivago adapter opts into this algebraically equivalent
        # low-memory execution of W1_feature. Original datasets use concat.
        self.metadata_projection = getattr(opt, 'metadata_projection', 'concat')
        # Original datasets keep their loss unchanged. The Trivago adapter
        # masks absent neighbor pairs instead of training on padding sessions.
        self.valid_groups_only = getattr(opt, 'valid_groups_only', False)
        self.action_mode = getattr(opt, 'action_mode', 'none')
        self.price_mode = getattr(opt, 'price_mode', 'off')
        if not self.n_feature and (self.action_mode != 'none' or self.price_mode != 'off' or opt.beta):
            raise ValueError('Metadata-free runs require action=none, price=off and beta=0')
        if self.price_mode not in ('off', 'relative_mlp_v1'):
            raise ValueError(f'Unsupported price_mode: {self.price_mode!r}')
        if self.price_mode != 'off' and self.action_mode != 'event_graph_candidate':
            raise ValueError('relative_mlp_v1 requires event_graph_candidate mode')
        if self.action_mode not in ('none', 'item_event', 'all_event', 'event_graph_candidate'):
            raise ValueError(f"Unsupported action_mode: {self.action_mode!r}")
        if self.action_mode == 'item_event':
            from trivago_action_types import ACTION_ENCODER, ACTION_FUSION, NUM_ACTIONS
            if getattr(opt, 'num_actions', None) != NUM_ACTIONS:
                raise ValueError(f'item_event requires num_actions={NUM_ACTIONS}')
            if getattr(opt, 'action_encoder', None) != ACTION_ENCODER:
                raise ValueError('item_event requires the configured chronological GRU encoder')
            if getattr(opt, 'action_fusion', None) != ACTION_FUSION:
                raise ValueError('item_event requires the configured post-graph action fusion')
            self.action_embedding = nn.Embedding(
                opt.num_actions, self.emb_size, padding_idx=0,
            )
            self.action_gru = nn.GRU(
                input_size=self.emb_size, hidden_size=self.emb_size,
                num_layers=1, batch_first=True, bidirectional=False,
            )
        if self.action_mode == 'event_graph_candidate':
            from trivago_action_types import EVENT_GRAPH_ENCODER, EVENT_GRAPH_FUSION, NUM_ACTIONS
            if getattr(opt, 'num_actions', None) != NUM_ACTIONS:
                raise ValueError(f'event_graph_candidate requires num_actions={NUM_ACTIONS}')
            if (getattr(opt, 'action_encoder', None) != EVENT_GRAPH_ENCODER
                    or getattr(opt, 'action_fusion', None) != EVENT_GRAPH_FUSION):
                raise ValueError('event_graph_candidate requires its versioned graph and candidate encoders')
            self.action_embedding = nn.Embedding(opt.num_actions, self.emb_size, padding_idx=0)
            self.event_input = nn.Linear(3 * self.emb_size, self.emb_size)
            self.event_temporal_value = nn.Linear(self.emb_size, self.emb_size, bias=False)
            self.event_repeat_value = nn.Linear(self.emb_size, self.emb_size, bias=False)
            self.event_graph_update = nn.Linear(3 * self.emb_size, self.emb_size)
            self.event_graph_norm = nn.LayerNorm(self.emb_size)
            self.event_graph_dropout = nn.Dropout(0.1)
            self.candidate_query = nn.Linear(2 * self.emb_size, self.emb_size)
            self.candidate_event_key = nn.Linear(self.emb_size, self.emb_size, bias=False)
            self.candidate_event_value = nn.Linear(self.emb_size, self.emb_size, bias=False)
            self.candidate_event_score = nn.Sequential(
                nn.Linear(2 * self.emb_size, self.emb_size),
                nn.Tanh(),
                nn.Linear(self.emb_size, 1),
            )
        if self.action_mode == 'all_event':
            from trivago_action_types import ALL_EVENT_ENCODER, ALL_EVENT_FUSION
            if (getattr(opt, 'action_encoder', None) != ALL_EVENT_ENCODER
                    or getattr(opt, 'action_fusion', None) != ALL_EVENT_FUSION):
                raise ValueError('all_event requires the joint-pair GRU and gated session fusion')
            if getattr(opt, 'num_event_tokens', 0) < 11:
                raise ValueError('all_event requires a fitted event vocabulary')
            token_actions = getattr(opt, 'event_token_action_ids', None)
            if (token_actions is None or len(token_actions) != opt.num_event_tokens
                    or list(token_actions[:11]) != list(range(11))
                    or any(int(a) not in range(1, 11) for a in token_actions[11:])):
                raise ValueError('all_event requires a valid token-to-action mapping')
            self.register_buffer('event_token_action_ids', torch.tensor(token_actions, dtype=torch.long))
            self.event_embedding = nn.Embedding(opt.num_event_tokens, self.emb_size, padding_idx=0)
            self.event_type_embedding = nn.Embedding(12, self.emb_size, padding_idx=0)
            self.event_gru = nn.GRU(self.emb_size, self.emb_size, batch_first=True)
            self.event_gate = nn.Linear(2 * self.emb_size, self.emb_size)
        self.all_feature_model = []
        for i, num in zip(range(self.n_feature), self.all_feature):
            fm = FL(num, self.emb_size, self.n_node)
            self.add_module('feature_model_{}'.format(i), fm)
            self.all_feature_model.append(fm)
        self.W1_feature = (nn.Linear(self.emb_size * self.n_feature, self.emb_size)
                           if self.n_feature else None)
        self.query = nn.Linear(self.emb_size , self.emb_size)
        self.key = nn.Linear(self.emb_size , self.emb_size)
        self.value = nn.Linear(self.emb_size , self.emb_size)
        self.W2_feature = nn.Linear(self.emb_size * 2, self.emb_size)
        self.num_heads = opt.num_heads  # 4
        self.attention_head_size = int(self.emb_size / self.num_heads)
        self.dropout20 = nn.Dropout(0.2)
        self.u_feature = nn.Parameter(torch.Tensor(self.emb_size,1))
        self.beta = opt.beta

        self.q1_co = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.q2_co = nn.Parameter(torch.Tensor(self.emb_size + 1, self.emb_size))
        self.q3_co = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W4_co = nn.Linear(self.emb_size * 2, self.emb_size)
        event_input_dim = self.emb_size * (3 if self.action_mode in ('item_event', 'event_graph_candidate') else 2)
        self.W5_co = nn.Linear(event_input_dim, self.emb_size)
        self.W6_co = nn.Linear(self.emb_size, self.emb_size)
        self.W7_co = nn.Linear(self.emb_size, self.emb_size)
        self.W8_co = nn.Linear(self.emb_size, self.emb_size)
        self.q4_co = nn.Parameter(torch.Tensor(self.emb_size, 1))

        self.W1_seq = nn.Linear(self.emb_size * 2, self.emb_size)
        self.q1_seq = nn.Parameter(torch.Tensor(self.emb_size + 1, self.emb_size))
        self.q2_seq = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W2_seq = nn.Linear(self.emb_size * 2, self.emb_size)
        self.q3_seq = nn.Parameter(torch.Tensor(self.emb_size + 1, self.emb_size))
        self.q4_seq = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.W3_seq = nn.Linear(self.emb_size * 2, self.emb_size)
        self.W4_seq = nn.Linear(event_input_dim, self.emb_size)
        self.W5_seq = nn.Linear(self.emb_size, self.emb_size)
        self.W6_seq = nn.Linear(self.emb_size, self.emb_size)
        self.W7_seq = nn.Linear(self.emb_size, self.emb_size)
        self.q5_seq = nn.Parameter(torch.Tensor(self.emb_size, 1))

        self.W1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.LN = nn.LayerNorm(self.emb_size)
        self.W2 = nn.Linear(self.emb_size, self.emb_size)
        self.W3 = nn.Linear(self.emb_size, self.emb_size)
        self.W4 = nn.Linear(self.emb_size, self.emb_size)
        self.W5 = nn.Linear(self.emb_size, self.emb_size)
        self.W6 = nn.Linear(self.emb_size, self.emb_size)
        self.W7 = nn.Linear(self.emb_size, self.emb_size)
        self.W8 = nn.Linear(self.emb_size, self.emb_size)
        self.W9 = nn.Linear(self.emb_size, self.emb_size)
        self.W10 = nn.Linear(self.emb_size, self.emb_size)
        self.W11 = nn.Linear(self.emb_size, self.emb_size)
        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.reset_parameters()
        # Create the optional head after initializing the legacy backbone so
        # price_mode=off preserves the original initialization path exactly.
        if self.price_mode == 'relative_mlp_v1':
            self.price_encoder = nn.Sequential(
                nn.Linear(9, self.emb_size), nn.Tanh(),
                nn.Linear(self.emb_size, self.emb_size),
            )
            self.price_gate = nn.Linear(2 * self.emb_size + 1, 1)
            self.price_head = nn.Sequential(
                nn.Linear(5 * self.emb_size + 1, self.emb_size), nn.Tanh(),
                nn.Linear(self.emb_size, 1),
            )
            with torch.no_grad():
                self.price_head[-1].weight.mul_(0.01)
                self.price_head[-1].bias.zero_()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if self.action_mode == 'item_event':
            with torch.no_grad():
                self.action_embedding.weight[0].zero_()
        if self.action_mode == 'event_graph_candidate':
            with torch.no_grad():
                self.action_embedding.weight[0].zero_()
                self.event_graph_norm.weight.fill_(1.0)
                self.event_graph_norm.bias.zero_()
        if self.action_mode == 'all_event':
            with torch.no_grad():
                self.event_embedding.weight[:11].zero_()
                self.event_type_embedding.weight[0].zero_()

    def encode_all_events(self, event_tokens, event_mask):
        """Many-to-one GRU of chronological (action, reference) tokens.

        This independent mask includes global actions after the last hotel
        interaction; the item-graph mask must never truncate these events.
        """
        if event_tokens is None or event_mask is None or event_tokens.shape != event_mask.shape:
            raise ValueError('all_event requires aligned [B,T] tokens and masks')
        lengths = event_mask.long().sum(1)
        positions = torch.arange(event_tokens.shape[1], device=event_tokens.device)[None, :]
        if not torch.equal(event_mask.bool(), positions < lengths[:, None]):
            raise ValueError('Events require chronological right-padded histories')
        if not torch.equal(event_tokens.ne(0), event_mask.bool()):
            raise ValueError('Event token padding must match event mask')
        result = self.event_embedding.weight.new_zeros((event_tokens.shape[0], self.emb_size))
        nonempty = lengths.gt(0)
        if not nonempty.any():
            return result
        active_tokens = event_tokens[nonempty]
        active_lengths = lengths[nonempty]
        # Share type information across references. Unseen references use only
        # the learned action-type vector rather than an untrained UNK pair.
        pair_h = self.event_embedding(active_tokens) * active_tokens.ge(11).unsqueeze(-1)
        event_h = pair_h + self.event_type_embedding(self.event_token_action_ids[active_tokens])
        packed = nn.utils.rnn.pack_padded_sequence(event_h, active_lengths.cpu(),
                                                batch_first=True, enforce_sorted=False)
        _, hidden = self.event_gru(packed)  # zero initial state; no padded steps
        result[nonempty] = hidden[-1]
        return result

    def encode_action_sequence(self, re_actions, mask):
        """Encode reverse-chronological, right-padded event actions with a GRU."""
        batch_size, max_len = re_actions.shape
        lengths = mask.long().sum(dim=1)
        positions = torch.arange(max_len, device=re_actions.device).unsqueeze(0)
        reverse_indices = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
        chrono_actions = re_actions.gather(1, reverse_indices)
        chrono_actions = chrono_actions * mask.long()
        chrono_embeddings = self.action_embedding(chrono_actions)
        chrono_hidden, _ = self.action_gru(chrono_embeddings)
        gather_indices = reverse_indices.unsqueeze(-1).expand(
            batch_size, max_len, self.emb_size,
        )
        reverse_hidden = chrono_hidden.gather(1, gather_indices)
        return reverse_hidden * mask.unsqueeze(-1).to(reverse_hidden.dtype)

    def encode_event_graph(self, item_nodes, position_nodes, re_inputs, re_actions, mask):
        """Message-pass over observed item/action events in the current session.

        Edges connect neighboring events in time and separate occurrences of
        the same hotel. Graph nodes only contain events in the current prefix.
        """
        valid = mask.bool()
        action_nodes = self.action_embedding(re_actions)
        node_input = torch.cat((item_nodes, position_nodes, action_nodes), dim=-1)
        nodes = torch.tanh(self.event_input(node_input)) * mask.unsqueeze(-1)

        _, length = re_inputs.shape
        positions = torch.arange(length, device=re_inputs.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        temporal = distance.eq(1)[None, :, :] & valid[:, :, None] & valid[:, None, :]
        same_hotel = re_inputs[:, :, None].eq(re_inputs[:, None, :])
        eye = torch.eye(length, device=re_inputs.device, dtype=torch.bool)[None, :, :]
        same_hotel = (same_hotel & ~eye & re_inputs[:, :, None].gt(0)
                      & valid[:, :, None] & valid[:, None, :])

        temporal_weights = temporal.to(nodes.dtype)
        repeat_weights = same_hotel.to(nodes.dtype)
        temporal_sum = torch.bmm(temporal_weights, self.event_temporal_value(nodes))
        repeat_sum = torch.bmm(repeat_weights, self.event_repeat_value(nodes))
        temporal_context = temporal_sum / temporal_weights.sum(-1, keepdim=True).clamp_min(1.0)
        repeat_context = repeat_sum / repeat_weights.sum(-1, keepdim=True).clamp_min(1.0)
        update = torch.tanh(self.event_graph_update(torch.cat(
            (nodes, temporal_context, repeat_context), dim=-1
        )))
        return self.event_graph_norm(nodes + self.event_graph_dropout(update)) * mask.unsqueeze(-1)

    def score_event_candidates(self, item_embedding, feature_embedding,
                               item_preference, feature_preference, event_states,
                               candidates, candidate_mask, history_mask,
                               price_features=None):
        """Rerank each impression candidate using its own attention over events."""
        if candidates is None or candidate_mask is None:
            raise ValueError('event_graph_candidate requires candidate IDs and a candidate mask')
        candidate_item = item_embedding[candidates]
        candidate_feature = feature_embedding[candidates]
        query = torch.tanh(self.candidate_query(torch.cat(
            (candidate_item, candidate_feature), dim=-1
        )))
        keys = self.candidate_event_key(event_states)
        values = self.candidate_event_value(event_states)
        logits = torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(self.emb_size)
        logits = logits.masked_fill(~history_mask.bool()[:, None, :], -1e4)
        attention = torch.softmax(logits, dim=-1)
        attention = attention * history_mask[:, None, :].to(attention.dtype)
        context = torch.bmm(attention, values)
        correction = self.candidate_event_score(torch.cat((query, context), dim=-1)).squeeze(-1)
        base_scores = (
            (candidate_item * item_preference[:, None, :]).sum(-1)
            + (candidate_feature * feature_preference[:, None, :]).sum(-1)
        )
        scores = base_scores + correction
        if self.price_mode == 'relative_mlp_v1':
            if price_features is None or price_features.shape[:2] != candidates.shape:
                raise ValueError('relative_mlp_v1 requires aligned [B,C,F] price features')
            if price_features.shape[-1] != 9:
                raise ValueError('relative_mlp_v1 requires the 9-feature schema')
            price_embedding = self.price_encoder(price_features)
            neutral_context = torch.zeros_like(item_preference)
            context_enabled = item_preference.new_zeros((item_preference.shape[0], 1))
            gate_input = torch.cat((item_preference, neutral_context, context_enabled), dim=-1)
            gate = torch.sigmoid(self.price_gate(gate_input))
            head_input = torch.cat((
                item_preference[:, None, :].expand_as(candidate_item),
                candidate_item, candidate_feature, price_embedding,
                neutral_context[:, None, :].expand_as(candidate_item),
                context_enabled[:, None, :].expand(*candidates.shape, 1),
            ), dim=-1)
            delta_price = price_features[..., 7] * gate * self.price_head(head_input).squeeze(-1)
            scores = scores + delta_price
        return scores.masked_fill(~candidate_mask.bool(), torch.finfo(scores.dtype).min)

    def project_feature_values(self, embedding_i):
        """Apply W1_feature field by field without [items, fields * dim].

        Every FL module, field order, weight block and value ID remains the
        same. The linear bias is added once after summing field contributions.
        """
        result = None
        for field_index in range(self.n_feature):
            neighbors = torch.as_tensor(
                self.adjacency_fi[field_index], device=embedding_i.device,
                dtype=torch.long,
            )
            values = self.all_feature_model[field_index](neighbors, embedding_i)
            start = field_index * self.emb_size
            weight = self.W1_feature.weight[:, start:start + self.emb_size]
            projected_values = values @ weight.t()
            value_ids = torch.as_tensor(
                self.adjacency_if[field_index], device=embedding_i.device,
                dtype=torch.long,
            )
            contribution = projected_values.index_select(0, value_ids)
            result = contribution if result is None else result + contribution
        if result is None:
            raise ValueError('At least one metadata field is required')
        if self.W1_feature.bias is not None:
            result = result + self.W1_feature.bias
        return result

    def encode_attributes(self, embedding_i, zeros, re_inputs, mask, re_groups,
                          re_neg_groups, mask_groups, max_len, batch_size):
        if not self.n_feature:
            return (None, embedding_i.new_zeros((batch_size, max_len, self.emb_size)),
                    embedding_i.new_zeros((batch_size, self.emb_size)), None,
                    embedding_i.new_zeros(()))
        # Attribute embedding learning
        # (1) Modeling global attribute-item relationship   (2) Concatenating multiple attribute values of an item
        if self.metadata_projection == 'fieldwise':
            embedding_f = self.project_feature_values(embedding_i)
        else:
            field_embeddings = []
            for i in range(self.n_feature):
                neighbors = torch.as_tensor(
                    self.adjacency_fi[i], device=embedding_i.device, dtype=torch.long,
                )
                field_ids = torch.as_tensor(
                    self.adjacency_if[i], device=embedding_i.device, dtype=torch.long,
                )
                new_embedding_f = self.all_feature_model[i](neighbors, embedding_i)
                field_embeddings.append(new_embedding_f[field_ids])
            # Preserve field order and gradients while avoiding repeated copies
            # of the growing [items, fields * embedding_dim] tensor.
            con_embedding_f = torch.cat(field_embeddings, dim=-1)
            del field_embeddings
            embedding_f = self.W1_feature(con_embedding_f)
        embedding_f = torch.cat([zeros, embedding_f], 0)
        # (3) Modeling attribute-attribute relationships within a session
        final_embedding_f = self.generate_embedding_f(embedding_f, re_inputs, mask, max_len, batch_size)
        # Stable user preference extraction
        sess_embedding_f = self.generate_sess_embedding_f(final_embedding_f, mask)
        current_stable_available = mask.bool().any(dim=1)
        sess_embedding_f = torch.where(current_stable_available[:, None], sess_embedding_f,
                                       torch.zeros_like(sess_embedding_f))
        # Neighbor-based contrastive self-supervised attribute learning task
        s_loss = 0
        one = embedding_i.new_ones(batch_size)
        valid_pair_count = embedding_i.new_zeros(())
        for i in torch.arange(self.m):
            mask_g = mask_groups[:,i,:].squeeze(1)
            pos_group = re_groups[:,i,:].squeeze(1)
            neg_group = re_neg_groups[:,i,:].squeeze(1)
            pos_mask = pos_group.ne(0).long()
            neg_mask = neg_group.ne(0).long()
            pos_sess_embedding_f = self.generate_sess_embedding_f(
                self.generate_embedding_f(embedding_f, pos_group, pos_mask, max_len, batch_size), pos_mask)
            pos = self.discrim_feature(sess_embedding_f, pos_sess_embedding_f)
            neg_sess_embedding_f = self.generate_sess_embedding_f(
                self.generate_embedding_f(embedding_f, neg_group, neg_mask, max_len, batch_size), neg_mask)
            neg = self.discrim_feature(sess_embedding_f, neg_sess_embedding_f)
            pair_loss = -torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg)))
            if self.valid_groups_only:
                # A pair contributes only when both source sessions exist.
                # Missing groups are padding, never synthetic positives/negatives.
                valid = (mask_g.any(dim=1) & re_neg_groups[:,i,:].ne(0).any(dim=1)
                         & current_stable_available)
                pair_loss = pair_loss * valid.to(pair_loss.dtype)
                valid_pair_count = valid_pair_count + valid.sum()
            s_loss = s_loss + torch.sum(pair_loss)
        if self.valid_groups_only:
            # Match CE's mean reduction. Padding and missing neighbor pairs
            # must not dilute the loss or change beta when batch size changes.
            s_loss = s_loss / valid_pair_count.clamp_min(1)

        return embedding_f, final_embedding_f, sess_embedding_f, current_stable_available, s_loss

    def forward(self, alias_re_inputs, items, mask, re_inputs, re_groups, re_neg_groups, mask_groups, edge_nodes_co, edge_weight_co, node_edges_co, f_adjacency_nodes_seq, f_adjacency_weight_seq, b_adjacency_nodes_seq, b_adjacency_weight_seq, re_actions=None, event_tokens=None, event_mask=None, candidates=None, candidate_mask=None, price_features=None):
        batch_size = mask.shape[0]
        max_len = mask.shape[1]
        if self.action_mode in ('item_event', 'event_graph_candidate'):
            if re_actions is None:
                raise ValueError(f'{self.action_mode} mode requires re_actions')
            if re_actions.shape != mask.shape:
                raise ValueError('re_actions must have the same [B,L] shape as mask')
        elif re_actions is not None:
            raise ValueError('re_actions supplied outside an action-aware mode')
        if self.action_mode == 'event_graph_candidate':
            if candidates is None or candidate_mask is None:
                raise ValueError('event_graph_candidate requires padded candidate IDs and a mask')
            if candidates.shape != candidate_mask.shape or candidates.shape[0] != batch_size:
                raise ValueError('Candidates and candidate mask must have aligned [B,C] shapes')
        elif candidates is not None or candidate_mask is not None:
            raise ValueError('Candidates supplied outside event_graph_candidate mode')
        if self.action_mode != 'all_event' and (event_tokens is not None or event_mask is not None):
            raise ValueError('Event tokens supplied outside all_event mode')
        if self.action_mode == 'all_event' and (event_tokens is None or event_tokens.shape[0] != batch_size):
            raise ValueError('Event histories must align with the item batch')
        zeros = self.embedding_i.weight.new_zeros((1, self.emb_size))
        sample_order = self.sample_order

        embedding_i = torch.cat([zeros, self.embedding_i.weight], 0)
        embedding_f, final_embedding_f, sess_embedding_f, current_stable_available, s_loss = (
            self.encode_attributes(embedding_i, zeros, re_inputs, mask, re_groups,
                                   re_neg_groups, mask_groups, max_len, batch_size)
        )

        # Item embedding learning
        tar_embedding_i = embedding_i[items]
        # Item co-occurrence relationship encoder
        # (1) Node-to-hyperedge propagation
        edge_embedding = embedding_i[edge_nodes_co]
        edge_nodes_mask_co = torch.where(edge_nodes_co != 0, 1, 0)
        edge_nodes_att_mask_co = (1.0 - edge_nodes_mask_co) * -10000.0
        num = torch.sum(edge_nodes_mask_co, 2)
        edge_embedding_query = torch.sum(edge_embedding.float(), 2) / torch.where(num == 0, float('inf'), num).unsqueeze(2)
        alpha = torch.matmul(edge_embedding * edge_embedding_query.unsqueeze(2), self.q1_co).squeeze(-1)
        alpha = alpha + edge_nodes_att_mask_co
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        edge_embedding = torch.sum(alpha * edge_embedding, 2)
        # (2) Hyperedge-to-node propagation
        node_embedding = torch.cat([edge_embedding[i][node_edges_co[i]].unsqueeze(0) for i in torch.arange(batch_size)])
        node_edges_weight_co = torch.cat([edge_weight_co[i][node_edges_co[i]].unsqueeze(0) for i in torch.arange(batch_size)])
        final_embedding_i_co = self.weighted_GAT(tar_embedding_i, node_embedding, node_edges_weight_co, self.q2_co, self.q3_co)
        # Item sequential relationship encoder
        # (1) Concatenating relative position embeddings
        f_n_neighbor = f_adjacency_weight_seq.shape[-1]
        f_adjacency_weight_seq = f_adjacency_weight_seq.reshape(batch_size, max_len, -1)
        f_adjacency_seq_mask = torch.where(f_adjacency_weight_seq != 0, 1, 0)
        f_embedding_pos_seq = self.f_embedding_pos.weight[:sample_order].reshape(1, 1, sample_order, 1, self.emb_size).repeat(batch_size, max_len, 1, f_n_neighbor, 1)
        neig_f_embedding_i_seq = self.W1_seq(torch.cat([f_embedding_pos_seq, embedding_i[f_adjacency_nodes_seq]], -1)).reshape(batch_size, max_len, -1, self.emb_size)
        neig_f_embedding_i_seq = torch.tanh(neig_f_embedding_i_seq)
        neig_f_embedding_i_seq = neig_f_embedding_i_seq * f_adjacency_seq_mask.unsqueeze(-1)
        b_n_neighbor = b_adjacency_weight_seq.shape[-1]
        b_adjacency_weight_seq = b_adjacency_weight_seq.reshape(batch_size, max_len, -1)
        b_adjacency_seq_mask = torch.where(b_adjacency_weight_seq != 0, 1, 0)
        b_embedding_pos_seq = self.b_embedding_pos.weight[:sample_order].reshape(1, 1, sample_order, 1, self.emb_size).repeat(batch_size, max_len, 1, b_n_neighbor, 1)
        neig_b_embedding_i_seq = self.W2_seq(torch.cat([b_embedding_pos_seq, embedding_i[b_adjacency_nodes_seq]], -1)).reshape(batch_size, max_len, -1, self.emb_size)
        neig_b_embedding_i_seq = torch.tanh(neig_b_embedding_i_seq)
        neig_b_embedding_i_seq = neig_b_embedding_i_seq * b_adjacency_seq_mask.unsqueeze(-1)
        # (2) learning the information propagated from high-order adjacent edges
        neig_f_embedding_i_seq = self.weighted_GAT(tar_embedding_i, neig_f_embedding_i_seq, f_adjacency_weight_seq, self.q1_seq, self.q2_seq)
        neig_b_embedding_i_seq = self.weighted_GAT(tar_embedding_i, neig_b_embedding_i_seq, b_adjacency_weight_seq, self.q3_seq, self.q4_seq)
        neig_embedding_i_seq = neig_f_embedding_i_seq + neig_b_embedding_i_seq
        # (3) Updating node embedding
        gate3 = torch.sigmoid(self.W3_seq(torch.cat([neig_embedding_i_seq, tar_embedding_i], -1)))
        final_embedding_i_seq = gate3 * neig_embedding_i_seq + (1 - gate3) * tar_embedding_i
        final_embedding_i_seq = self.dropout20(final_embedding_i_seq)

        # Dynamic user preference extraction
        # (2) Concatenating forward position embeddings
        embedding_pos_seq = self.f_embedding_pos.weight[:max_len].unsqueeze(0).repeat(batch_size, 1, 1)
        get = lambda index: final_embedding_i_co[index][alias_re_inputs[index]]
        embedding_i_co = torch.stack([get(i) for i in torch.arange(batch_size)])
        get = lambda index: final_embedding_i_seq[index][alias_re_inputs[index]]
        embedding_i_seq = torch.stack([get(i) for i in torch.arange(batch_size)])
        action_h = self.encode_action_sequence(re_actions, mask) if self.action_mode == 'item_event' else None
        event_states = None
        if self.action_mode == 'event_graph_candidate':
            event_states = self.encode_event_graph(
                embedding_i_seq, embedding_pos_seq, re_inputs, re_actions, mask,
            )
        co_parts = [embedding_i_co, embedding_pos_seq]
        if action_h is not None:
            co_parts.append(action_h)
        elif event_states is not None:
            co_parts.append(event_states)
        embedding_i_co = torch.tanh(self.W5_co(torch.cat(co_parts, -1)))
        embedding_i_co = embedding_i_co * mask.unsqueeze(-1)
        seq_parts = [embedding_i_seq, embedding_pos_seq]
        if action_h is not None:
            seq_parts.append(action_h)
        elif event_states is not None:
            seq_parts.append(event_states)
        embedding_i_seq = torch.tanh(self.W4_seq(torch.cat(seq_parts, -1)))
        embedding_i_seq = embedding_i_seq * mask.unsqueeze(-1)
        # (2) Learning session embeddings
        attention_mask = (1.0 - mask) * -10000.0
        f3 = self.W6_co(embedding_i_co)
        f4 = self.W7_co(final_embedding_f)
        f5 = self.W8_co(sess_embedding_f).unsqueeze(1)
        alpha2 = torch.matmul(self.leakyrelu(f3 + f4 + f5), self.q4_co).squeeze(-1)
        alpha2 = alpha2 + attention_mask
        alpha2 = torch.softmax(alpha2, -1).unsqueeze(-1)
        sess_embedding_i_co = torch.sum(alpha2 * embedding_i_co, 1)
        f6 = self.W5_seq(embedding_i_seq)
        f7 = self.W6_seq(final_embedding_f)
        f8 = self.W7_seq(sess_embedding_f).unsqueeze(1)
        alpha5 = torch.matmul(self.leakyrelu(f6 + f7 + f8), self.q5_seq).squeeze(-1)
        alpha5 = alpha5 + attention_mask
        alpha5 = torch.softmax(alpha5, -1).unsqueeze(-1)
        sess_embedding_i_seq = torch.sum(alpha5 * embedding_i_seq, 1)
        # (3) Integrating the two
        gate4 = torch.sigmoid(self.W1(torch.cat([sess_embedding_i_co, sess_embedding_i_seq], -1)))
        sess_embedding_i = gate4 * sess_embedding_i_co + (1 - gate4) * sess_embedding_i_seq
        sess_embedding_i = self.dropout20(self.LN(sess_embedding_i))
        if self.action_mode == 'all_event':
            event_context = self.encode_all_events(event_tokens, event_mask)
            gate_event = torch.sigmoid(self.event_gate(torch.cat([sess_embedding_i, event_context], -1)))
            sess_embedding_i = gate_event * sess_embedding_i + (1 - gate_event) * event_context
            current_dynamic_available = current_stable_available | event_mask.bool().any(dim=1)
            sess_embedding_i = torch.where(current_dynamic_available[:, None], sess_embedding_i,
                                           torch.zeros_like(sess_embedding_i))
        elif self.action_mode in ('item_event', 'event_graph_candidate'):
            current_dynamic_available = mask.bool().any(dim=1)

        if not self.n_feature:
            return sess_embedding_i @ embedding_i.T, s_loss

        # User preferences fusion
        sess_embedding_f = self.dropout20(self.LN(sess_embedding_f))
        sess_embedding_f = torch.where(current_stable_available[:, None], sess_embedding_f,
                                       torch.zeros_like(sess_embedding_f))
        m_c = torch.tanh(self.W2(sess_embedding_f * sess_embedding_i))
        m_j = torch.tanh(self.W3(sess_embedding_f + sess_embedding_i))
        r_i = torch.sigmoid(self.W4(m_c) + self.W5(m_j))
        r_f = torch.sigmoid(self.W6(m_c) + self.W7(m_j))
        m_f = torch.tanh(self.W8(sess_embedding_f * r_f) + self.W9((1 - r_i) * sess_embedding_i))
        m_i = torch.tanh(self.W10(sess_embedding_i * r_i) + self.W11((1 - r_f) * sess_embedding_f))
        f_pre = (sess_embedding_f + m_i) * m_f
        i_pre = (sess_embedding_i + m_f) * m_i
        if self.action_mode == 'event_graph_candidate':
            scores = self.score_event_candidates(
                embedding_i, embedding_f, i_pre, f_pre, event_states,
                candidates, candidate_mask, mask, price_features,
            )
        else:
            scores_f = torch.mm(f_pre, torch.transpose(embedding_f, 1, 0))
            scores_i = torch.mm(i_pre, torch.transpose(embedding_i, 1, 0))
            scores = scores_f + scores_i

        return scores, self.beta * s_loss

    def weighted_GAT(self, tar_embedding_i, neig_embedding_i, adjacency_weight, q1, q2):
        attention_adjacency_mask = torch.where(adjacency_weight != 0, 0, -10000)
        alpha = torch.matmul(torch.cat([tar_embedding_i.unsqueeze(2) * neig_embedding_i, adjacency_weight.unsqueeze(-1)], -1), q1)
        alpha = torch.matmul(self.leakyrelu(alpha), q2).squeeze(-1)
        alpha = alpha + attention_adjacency_mask
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        neig_embedding_i = torch.sum(alpha * neig_embedding_i, 2)
        return neig_embedding_i

    def generate_sess_embedding_f(self, final_embedding_f, mask):
        attention_mask = (1.0 - mask) * -10000.0
        alpha = torch.matmul(final_embedding_f, self.u_feature).squeeze(-1)
        alpha = alpha + attention_mask
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        sess_embedding_f = torch.sum(alpha * final_embedding_f, 1)
        return sess_embedding_f

    def generate_embedding_f(self, embedding_f, re_inputs, mask, max_len, batch_size):
        out_embedding_f = embedding_f[re_inputs]
        embedding_pos_feature = self.f_embedding_pos.weight[:max_len].unsqueeze(0).repeat(batch_size, 1, 1)
        h = self.W2_feature(torch.cat([embedding_pos_feature, out_embedding_f], -1))
        h = torch.tanh(h)
        final_embedding_f = self.multi_head_SSL(h, mask, batch_size, max_len)
        return final_embedding_f

    def discrim_feature(self, x1, x2):
        return torch.sum(x1 * x2, 1)

    def multi_head_SSL(self, h, mask, batch_size, max_len):
        attention_mask = mask.reshape(batch_size, 1, 1, max_len)
        attention_mask = (1.0 - attention_mask) * -10000.0
        mixed_query_layer = self.query(h)
        mixed_key_layer = self.key(h)
        mixed_value_layer = self.value(h)
        query_layer = self.transpose_layer(mixed_query_layer, batch_size, max_len)
        key_layer = self.transpose_layer(mixed_key_layer, batch_size, max_len)
        value_layer = self.transpose_layer(mixed_value_layer, batch_size, max_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = torch.softmax(attention_scores, -1)
        attention_probs = self.dropout20(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)
        sa_result = context_layer.reshape(batch_size, max_len, -1)
        sa_result = sa_result * mask.unsqueeze(-1)
        return sa_result

    def transpose_layer(self, x, batch_size, max_len):
        x = x.reshape(batch_size, max_len, self.num_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

def forward(model, data):
    if len(data) not in (15, 16, 17, 19, 20):
        raise ValueError(f"Expected 15, 16, 17, 19 or 20 batch fields, got {len(data)}")
    (alias_re_inputs, items, mask, targets, re_inputs, re_groups, re_neg_groups,
     mask_groups, edge_nodes_co, edge_weight_co, node_edges_co,
     f_adjacency_nodes_seq, f_adjacency_weight_seq,
     b_adjacency_nodes_seq, b_adjacency_weight_seq) = data[:15]
    re_actions = data[15] if len(data) in (16, 19, 20) else None
    event_tokens, event_mask = data[15:17] if len(data) == 17 else (None, None)
    candidates, candidate_mask = (data[16], data[17]) if len(data) in (19, 20) else (None, None)
    candidate_targets = data[18] if len(data) in (19, 20) else None
    price_features = data[19] if len(data) == 20 else None
    expected_size = {'none': 15, 'item_event': 16, 'all_event': 17,
                     'event_graph_candidate': 20 if model.price_mode != 'off' else 19}[model.action_mode]
    if model.action_mode in ('item_event', 'event_graph_candidate') and re_actions is None:
        raise ValueError(f'{model.action_mode} mode requires its action-history batch field')
    if model.action_mode == 'none' and re_actions is not None:
        raise ValueError('Batch includes actions but model action_mode is none')
    if len(data) != expected_size:
        raise ValueError(f'{model.action_mode} mode requires {expected_size} batch fields')
    device = model.embedding_i.weight.device
    alias_re_inputs = alias_re_inputs.to(device=device, dtype=torch.long)
    items = items.to(device=device, dtype=torch.long)
    mask = mask.to(device=device, dtype=torch.long)
    re_inputs = re_inputs.to(device=device, dtype=torch.long)
    re_groups = re_groups.to(device=device, dtype=torch.long)
    re_neg_groups = re_neg_groups.to(device=device, dtype=torch.long)
    mask_groups = mask_groups.to(device=device, dtype=torch.long)
    edge_nodes_co = edge_nodes_co.to(device=device, dtype=torch.long)
    edge_weight_co = edge_weight_co.to(device=device, dtype=torch.float32)
    node_edges_co = node_edges_co.to(device=device, dtype=torch.long)
    f_adjacency_nodes_seq = f_adjacency_nodes_seq.to(device=device, dtype=torch.long)
    f_adjacency_weight_seq = f_adjacency_weight_seq.to(device=device, dtype=torch.float32)
    b_adjacency_nodes_seq = b_adjacency_nodes_seq.to(device=device, dtype=torch.long)
    b_adjacency_weight_seq = b_adjacency_weight_seq.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.long)
    if re_actions is not None:
        re_actions = re_actions.to(device=device, dtype=torch.long)
    if event_tokens is not None:
        event_tokens = event_tokens.to(device=device, dtype=torch.long)
        event_mask = event_mask.to(device=device, dtype=torch.long)
    if candidates is not None:
        candidates = candidates.to(device=device, dtype=torch.long)
        candidate_mask = candidate_mask.to(device=device, dtype=torch.long)
        candidate_targets = candidate_targets.to(device=device, dtype=torch.long)
    if price_features is not None:
        price_features = price_features.to(
            device=device, dtype=model.embedding_i.weight.dtype,
        )
    scores, s_loss = model(
        alias_re_inputs, items, mask, re_inputs, re_groups, re_neg_groups,
        mask_groups, edge_nodes_co, edge_weight_co, node_edges_co,
        f_adjacency_nodes_seq, f_adjacency_weight_seq,
        b_adjacency_nodes_seq, b_adjacency_weight_seq,
        re_actions=re_actions, event_tokens=event_tokens, event_mask=event_mask,
        candidates=candidates, candidate_mask=candidate_mask,
        price_features=price_features,
    )
    if model.action_mode == 'event_graph_candidate':
        return candidate_targets, scores, s_loss
    return targets, scores, s_loss
