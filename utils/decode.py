import torch
import pickle
import ctcdecode
from itertools import groupby


class Decoder:
    def __init__(self, num_classes, gloss_dict, search_mode, blank_token='<BLANK>'):
        # search_mode: max/beam
        self.num_classes = num_classes
        with open(gloss_dict, 'rb') as f:
            self.gloss_dict = pickle.load(f)
        self.search_mode = search_mode
        self.blank_id = self.gloss_dict[blank_token]

        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=10, blank_id=self.blank_id,
                                                    num_processes=10)

    def decode(self, alignments, valid_len):
        if self.search_mode == 'max':
            return self.max_decode(alignments, valid_len)
        elif self.search_mode == 'beam':
            return self.beam_search_decode(alignments, valid_len)
        else:
            print('search mode not valid. Choose from max/beam')
            return

    def max_decode(self, alignments, valid_len):
        outputs = []
        for batch_id in range(alignments.shape[0]):
            alignment = alignments[batch_id, :valid_len[batch_id], :]

            # alignment: length * num_classes
            _, max_alignment = torch.max(alignment, dim=1)

            # max_alignment: shape = length
            output = []
            if max_alignment[0].item() != self.blank_id:
                output.append(max_alignment[0].item())
            for i in range(1, max_alignment.shape[0]):
                if max_alignment[i].item() != self.blank_id and max_alignment[i].item() != max_alignment[i - 1].item():
                    output.append(max_alignment[i].item())
                else:
                    continue
            outputs.append(torch.Tensor(output))
        return outputs  # list of tensors

    def beam_search_decode(self, alignments, valid_len):
        alignments = alignments.softmax(-1).cpu()
        valid_len = valid_len.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(alignments, valid_len)
        outputs = []
        for i in range(alignments.shape[0]):
            best_output = beam_result[i][0][:out_seq_len[i][0]]
            if len(best_output) != 0:
                best_output = torch.stack([x[0] for x in groupby(best_output)])
            outputs.append(best_output)
        return outputs
