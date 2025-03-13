# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch
import torch.nn as nn


class MainPokerModuleFLAT(nn.Module):
    """
    Feeds parts of the observation through different fc layers before the RNN

    Structure (each branch merge is a concat):

    Table & Player state --> FC -> RE -> FCS -> RE ----------------------------.
    Board Cards ---> FC -> RE --> cat -> FC -> RE -> FCS -> RE -> FC -> RE --> cat --> FC -> RE -> FCS-> RE -> Normalize
    Private Cards -> FC -> RE -'


    where FCS refers to FC+Skip and RE refers to ReLU
    """

    def __init__(self,
                 env_bldr,
                 device,
                 mpm_args,
                 ):
        super().__init__()
        self.args = mpm_args

        self.env_bldr = env_bldr

        self.N_SEATS = self.env_bldr.N_SEATS
        self.device = device

        self.board_start = self.env_bldr.obs_board_idxs[0]
        self.board_stop = self.board_start + len(self.env_bldr.obs_board_idxs)

        self.pub_obs_size = self.env_bldr.pub_obs_size
        self.priv_obs_size = self.env_bldr.priv_obs_size

        self._relu = nn.ReLU(inplace=False)
        self.bucket_feature = BucketFeature()

        if mpm_args.use_pre_layers:
            # self._priv_cards = nn.Linear(in_features=self.env_bldr.priv_obs_size,
            #                              out_features=mpm_args.other_units)
            # self._board_cards = nn.Linear(in_features=self.env_bldr.obs_size_board,
            #                               out_features=mpm_args.other_units)

            self.cards_fc_1 = nn.Linear(in_features=39, out_features=mpm_args.card_block_units)
            self.cards_fc_2 = nn.Linear(in_features=mpm_args.card_block_units, out_features=mpm_args.card_block_units)
            self.cards_fc_3 = nn.Linear(in_features=mpm_args.card_block_units, out_features=mpm_args.other_units)

            self.hist_and_state_1 = nn.Linear(in_features=self.env_bldr.pub_obs_size - self.env_bldr.obs_size_board,
                                              out_features=mpm_args.other_units)
            self.hist_and_state_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

            self.final_fc_1 = nn.Linear(in_features=2 * mpm_args.other_units, out_features=mpm_args.other_units)
            self.final_fc_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

        else:
            self.final_fc_1 = nn.Linear(in_features=self.env_bldr.complete_obs_size, out_features=mpm_args.other_units)
            self.final_fc_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

        self.lut_range_idx_2_priv_o = torch.from_numpy(self.env_bldr.lut_holder.LUT_RANGE_IDX_TO_PRIVATE_OBS)
        self.lut_range_idx_2_priv_o = self.lut_range_idx_2_priv_o.to(device=self.device, dtype=torch.float32)

        self.to(device)

    @property
    def output_units(self):
        return self.args.other_units

    def forward(self, pub_obses, range_idxs):
        """
        1. do list -> padded
        2. feed through pre-processing fc layers
        3. PackedSequence (sort, pack)
        4. rnn
        5. unpack (unpack re-sort)
        6. cut output to only last entry in sequence

        Args:
            pub_obses (list):                 list of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (LongTensor):        range_idxs (one for each pub_obs) tensor([2, 421, 58, 912, ...])
        """

        # ____________________________________________ Packed Sequence _____________________________________________
        
        priv_obses = self.lut_range_idx_2_priv_o[range_idxs]

        if isinstance(pub_obses, list):
            pub_obses = torch.from_numpy(np.array(pub_obses)).to(self.device, torch.float32)

        if self.args.use_pre_layers:
            _board_obs = pub_obses[:, self.board_start:self.board_stop]
            _hist_and_state_obs = torch.cat([
                pub_obses[:, :self.board_start],
                pub_obses[:, self.board_stop:]
            ],
                dim=-1
            )
            y = self._feed_through_pre_layers(board_obs=_board_obs, priv_obs=priv_obses,
                                              hist_and_state_obs=_hist_and_state_obs)

        else:
            y = torch.cat((priv_obses, pub_obses,), dim=-1)

        final = self._relu(self.final_fc_1(y))
        final = self._relu(self.final_fc_2(final) + final)

        # Normalize last layer
        if self.args.normalize:
            final = final - final.mean(dim=-1).unsqueeze(-1)
            final = final / final.std(dim=-1).unsqueeze(-1)

        return final

    def _feed_through_pre_layers(self, priv_obs, board_obs, hist_and_state_obs):

        # """""""""""""""
        # Cards Body
        # """""""""""""""
        group_tensor = self.encode_card_group(priv_obs, board_obs)
        # print(priv_obs, board_obs, group_tensor)
        # _priv_1 = self._relu(self._priv_cards(priv_obs))
        # _board_1 = self._relu(self._board_cards(board_obs))

        # cards_out = self._relu(self.cards_fc_1(torch.cat([_priv_1, _board_1, group_id], dim=-1)))
        cards_out = self._relu(self.cards_fc_1(group_tensor))
        cards_out = self._relu(self.cards_fc_2(cards_out) + cards_out)
        cards_out = self.cards_fc_3(cards_out)

        hist_and_state_out = self._relu(self.hist_and_state_1(hist_and_state_obs))
        hist_and_state_out = self.hist_and_state_2(hist_and_state_out) + hist_and_state_out

        return self._relu(torch.cat([cards_out, hist_and_state_out], dim=-1))
    
    def encode_card_group(self, priv_obs, board_obs):
        RANK_DICT = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        SUIT_DICT = {'h': 0, 'd': 1, 's': 2, 'c': 3}
        RANK_REV_DICT = {v: k for k, v in RANK_DICT.items()}
        SUIT_REV_DICT = {v: k for k, v in SUIT_DICT.items()}

        def decode_card(card_obs):
            if torch.all(card_obs == 0):
                return None
            rank_bits = card_obs[:13]
            suit_bits = card_obs[13:]
            rank_index = (rank_bits == 1).nonzero(as_tuple=True)[0][0]
            suit_index = (suit_bits == 1).nonzero(as_tuple=True)[0][0]
            rank = RANK_REV_DICT[rank_index.item()]
            suit = SUIT_REV_DICT[suit_index.item()]
            return f"{rank}{suit}"
        
        results = []
        
        for priv, board in zip(priv_obs, board_obs):
            hole_cards, board_cards = [], []
            hole_cards_len, board_cards_len = 2, 5
            
            for i in range(hole_cards_len):
                card_obs = priv[i * 17:(i + 1) * 17]
                card = decode_card(card_obs)
                if card:
                    hole_cards.append(card)

            for j in range(board_cards_len):
                card_obs = board[j * 17:(j + 1) * 17]
                card = decode_card(card_obs)
                if card:
                    board_cards.append(card)
            
            group_tensor = self.bucket_feature.query_groups(hole_cards, board_cards)
            results.append(group_tensor)

        results_2d = [t.unsqueeze(0) for t in results]
        return torch.cat(results_2d, dim=0).to(device=self.device)

class MPMArgsFLAT:

    def __init__(self,
                 use_pre_layers=True,
                 card_block_units=192,
                 other_units=64,
                 normalize=True,
                 ):
        self.use_pre_layers = use_pre_layers
        self.other_units = other_units
        self.card_block_units = card_block_units
        self.normalize = normalize

    def get_mpm_cls(self):
        return MainPokerModuleFLAT


import pickle
import os

class BucketFeature:
    def __init__(self):
        self.db_folder_path = '/home/lanhou/Workspace/Deep-CFR/assets/db'
        self.value_to_char = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J'}
        self.char_to_value = {v: k for k, v in self.value_to_char.items()}
        self.load_bucket()

    def load_bucket(self):
        self.ref_data = [{} for i in range(8)]
        for i in range(5, 8):
            file_path = os.path.join(self.db_folder_path, f'bckt{i}-2.pkl')  
            with open(file_path, 'rb') as fp:
                self.ref_data[i] = pickle.load(fp)
        self.ref_data[2] = {
            '22': 4, '33': 5, '44': 5, '55': 6, '66': 6, '77': 7, '88': 7, '99': 8, 'TT': 9, 'JJ': 9, 'QQ': 10, 'KK': 10, 'AA': 10,
            '23': 1, '24': 1, '25': 1, '26': 1, '27': 1, '28': 2, '29': 2, '2T': 2, '2J': 3, '2Q': 3, '2K': 4, '2A': 5,
            '34': 1, '35': 1, '36': 1, '37': 1, '38': 2, '39': 2, '3T': 3, '3J': 3, '3Q': 4, '3K': 4, '3A': 5,
            '45': 2, '46': 2, '47': 2, '48': 2, '49': 2, '4T': 3, '4J': 3, '4Q': 4, '4K': 4, '4A': 5,
            '56': 2, '57': 2, '58': 2, '59': 3, '5T': 3, '5J': 3, '5Q': 4, '5K': 5, '5A': 6,
            '67': 3, '68': 3, '69': 3, '6T': 3, '6J': 4, '6Q': 4, '6K': 5, '6A': 5,
            '78': 3, '79': 3, '7T': 4, '7J': 4, '7Q': 4, '7K': 5, '7A': 6,
            '89': 4, '8T': 4, '8J': 4, '8Q': 5, '8K': 5, '8A': 6,
            '9T': 4, '9J': 5, '9Q': 5, '9K': 5, '9A': 6,
            'TJ': 5, 'TQ': 5, 'TK': 6, 'TA': 6,
            'JQ': 5, 'JK': 6, 'JA': 7,
            'QK': 6, 'QA': 7,
            'KA': 7, 
            '2s3s': 1, '2s4s': 2, '2s5s': 2, '2s6s': 2, '2s7s': 2, '2s8s': 2, '2s9s': 3, '2sTs': 3, '2sJs': 4, '2sQs': 4, '2sKs': 5, '2sAs': 5,
            '3s4s': 2, '3s5s': 2, '3s6s': 2, '3s7s': 2, '3s8s': 2, '3s9s': 3, '3sTs': 3, '3sJs': 4, '3sQs': 4, '3sKs': 5, '3sAs': 6,
            '4s5s': 2, '4s6s': 2, '4s7s': 3, '4s8s': 3, '4s9s': 3, '4sTs': 3, '4sJs': 4, '4sQs': 4, '4sKs': 5, '4sAs': 6,
            '5s6s': 3, '5s7s': 3, '5s8s': 3, '5s9s': 3, '5sTs': 3, '5sJs': 4, '5sQs': 5, '5sKs': 5, '5sAs': 6, 
            '6s7s': 3, '6s8s': 3, '6s9s': 4, '6sTs': 4, '6sJs': 4, '6sQs': 5, '6sKs': 5, '6sAs': 6,
            '7s8s': 4, '7s9s': 4, '7sTs': 4, '7sJs': 4, '7sQs': 5, '7sKs': 5, '7sAs': 6,
            '8s9s': 4, '8sTs': 4, '8sJs': 5, '8sQs': 5, '8sKs': 6, '8sAs': 6,
            '9sTs': 5, '9sJs': 5, '9sQs': 5, '9sKs': 6, '9sAs': 6,
            'TsJs': 5, 'TsQs': 6, 'TsKs': 6, 'TsAs': 7,
            'JsQs': 6, 'JsKs': 6, 'JsAs': 7,
            'QsKs': 6, 'QsAs': 7,
            'KsAs': 7
        }
        for key, value in self.ref_data[2].items():
                if value in self.value_to_char:
                        self.ref_data[2][key] = self.value_to_char[value]

    def transform(self, hole_cards, board_cards, idx):
        cards = hole_cards + board_cards
        num_dict = {
            "s": 0,
            "h": 0,
            "c": 0,
            "d": 0,
        }
        
        for card in cards:
            num_dict[card[-1]] += 1
        
        max_suit = max(num_dict, key=num_dict.get)
        max_suit_count = num_dict[max_suit]

        MIN_NUM = {2: 2, 5: 3, 6: 4, 7: 5}
        if max_suit_count < MIN_NUM[idx]:
            return ''.join([f"{card[:-1]}" for card in cards])
        else:
            result = []
            for card in cards:
                suit = card[-1]
                if suit == max_suit:
                    result.append(f"{card[:-1]}s") 
                else:
                    result.append(f"{card[:-1]}o") 
            return ''.join(result)


    def find(self, code, idx):
        def get_card_rank(card):
            ranks = '23456789TJQKA'
            return ranks.index(card)
        
        perm = ""
        if len(code) == idx:
            hand = ''.join(sorted(code[:2], key=get_card_rank))
            remaining = ''.join(sorted(code[2:], key=get_card_rank))
            perm = hand + remaining

        elif len(code) == idx*2:
            grouped = [code[i:i+2] for i in range(0, len(code), 2)]
            hand = ''.join(sorted(grouped[:2], key=lambda x: (get_card_rank(x[0]), x[1])))
            remaining = ''.join(sorted(grouped[2:], key=lambda x: (get_card_rank(x[0]), x[1])))
            perm = hand + remaining

        else:
            return None
        
        if perm in self.ref_data[idx]:
            return [perm, self.ref_data[idx][perm]]

        print(f"{code}, {perm} not found")
        
        return None

    def query_group(self, hole_cards, board_cards):
        idx = len(hole_cards) + len(board_cards)
        code = self.transform(hole_cards, board_cards, idx)
        res = self.find(code, idx)
        if res == None:
            return -1
        else:
            return self.char_to_value[res[-1]]
    
    def query_groups(self, hole_cards, board_cards):
        lengths = [10, 9, 10, 10]
        board_card_num = [0, 3, 4, 5]
        tensors = []
        for num, length in zip(board_card_num, lengths):
            curr_tensor = None
            if num > len(board_cards):
                curr_tensor = torch.zeros(length)
            else:
                curr_board_cards = board_cards[:num]
                curr_tensor = self.num_to_tensor(
                    self.query_group(hole_cards, curr_board_cards),
                    length
                )
            tensors.append(curr_tensor)
        
        return torch.cat(tensors)
    
    def num_to_tensor(self, num, length):
        tensor = torch.zeros(length)
        tensor[num - 1] = 1
        return tensor
