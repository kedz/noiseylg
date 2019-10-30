import plum
from plum.types import register, PlumObject, HP, props, Variable
import torch
import numpy as np


@register("seq2seq.search.beam")
class BeamSearch(PlumObject):

    max_steps = HP(default=999999, type=props.INTEGER)
    beam_size = HP(default=4, type=props.INTEGER)
    vocab = HP()
    
    def __pluminit__(self):
        self.reset()

    def reset(self):
        self.is_finished = False
        self.steps = 0
        self._states = []
        self._outputs = []

    def init_state(self, batch_size, encoder_state):

        beam_state = {}

        n, bs, ds = encoder_state.size()
        assert batch_size == bs

        beam_state["decoder_state"] = encoder_state.unsqueeze(2)\
            .repeat(1, 1, self.beam_size, 1)\
            .view(n, batch_size * self.beam_size, ds)

        beam_state["output"] = Variable(
            torch.LongTensor(
                [self.vocab.start_index] * batch_size * self.beam_size)\
                .view(1, -1),
            lengths=torch.LongTensor([1] * batch_size * self.beam_size),
            length_dim=0, batch_dim=1, pad_value=self.vocab.pad_index)

        if str(encoder_state.device) != "cpu":
            beam_state["output"] = beam_state["output"].cuda(
                encoder_state.device)

        # Start the first beam of each batch with 0 log prob, and all others 
        # with -inf.
        beam_state["accum_log_prob"] = (
            self._init_accum_log_probs(batch_size, encoder_state.device)
        )

        # At the first time step no sequences have been terminated so this mask
        # is all 0s. 
        beam_state["terminal_mask"] = (
            encoder_state.new().byte().new(
                1, batch_size * self.beam_size, 1).fill_(0)
        )   

        return beam_state

    def _init_accum_log_probs(self, batch_size, device):
        lp = torch.FloatTensor(1, batch_size, self.beam_size, 1)
        if "cuda" in str(device):
            lp = lp.cuda(device)
        lp.data.fill_(0)
        lp.data[:,:,1:].fill_(float("-inf"))
        return lp.view(1, batch_size * self.beam_size, 1)


    def init_context(self, encoder_state):
        n, bs, ds = encoder_state["output"].size()
        beam_encoder_output = encoder_state["output"].repeat_batch_dim(
            self.beam_size)
            #.repeat(1, 1, self.beam_size, 1)\
            #.view(n, bs * self.beam_size, ds)
        return {"encoder_output": beam_encoder_output}

    def __call__(self, decoder, encoder_state, controls=None):
        self.reset()

        # TODO get batch size in a more reliable way. This will probably break
        # for cnn or transformer based models.
        batch_size = encoder_state["state"].size(1)
        search_state = self.init_state(batch_size, encoder_state["state"])
        search_context = self.init_context(encoder_state)
        active_items = search_state["decoder_state"].new(batch_size).byte() \
            .fill_(1)

        if controls is not None:
            controls = controls.repeat_batch_dim(self.beam_size)

        self._beam_scores = [list() for _ in range(batch_size)]
        self._num_complete = search_state["decoder_state"].new()\
            .long().new(batch_size).fill_(0)
        self._terminal_info = [list() for _ in range(batch_size)]

        # Perform search until we either trigger a termination condition for
        # each batch item or we reach the maximum number of search steps.
        while self.steps < self.max_steps and not self.is_finished:
            
            search_state = self.next_state(
                decoder, batch_size, search_state, search_context, 
                active_items, controls)        
            active_items = self.check_termination(search_state, active_items)
            self._is_finished = torch.all(~active_items)

            self._states.append(search_state)
            self.steps += 1

        # Finish the search by collecting final sequences, and other 
        # stats. 
        self._collect_search_states(active_items)
        self._incomplete_items = active_items
        self._is_finished = True

        return self

    def next_state(self, decoder, batch_size, prev_state, context, 
                   active_items, controls):

        # Get next state from the decoder.
        next_state = decoder.next_state(prev_state, context, controls=controls)

        # Compute the top beam_size next outputs for each beam item.
        # topk_lps (1 x batch size x beam size x beam size)
        # candidate_outputs (1 x batch size x beam size x beam size)
        topk_lps, candidate_outputs = torch.topk(
            next_state["log_probs"].data \
                .view(1, batch_size, self.beam_size, -1),
            k=self.beam_size, dim=3)

        # If any sequence was completed last step, we should mask it's log
        # prob so that we don't generate from the terminal token.
        # slp (1 x batch_size x beam size x 1) 
        slp = prev_state["accum_log_prob"] \
            .masked_fill(prev_state["terminal_mask"], float("-inf")) \
            .view(1, batch_size, self.beam_size, 1)

        # Combine next step log probs with the previous sequences cumulative
        # log probs, i.e.
        #     log P(y_t) = log P(y_<t) + log P(y_t)
        # candidate_log_probs (1 x batch size x beam size x beam size)
        candidate_log_probs = slp + topk_lps

        # Rerank and select the beam_size best options from the available 
        # beam_size ** 2 candidates.
        # b_seq_lps (1 x (batch size * beam size) x 1)
        # b_scores (1 x (batch size * beam size) x 1)
        # b_output (1 x (batch size * beam size))
        # b_indices ((batch size * beam size))
        b_seq_lps, b_scores, b_output, b_indices = self._next_candidates(
            batch_size, candidate_log_probs, candidate_outputs)

        # TODO re-implement this behavior
        #next_state.stage_indexing("batch", b_indices)

        next_state = {
            "decoder_state": next_state["decoder_state"]\
                .index_select(1, b_indices),
            "output": b_output,
            "accum_log_prob": b_seq_lps,
            "beam_score": b_scores,
            "beam_indices": b_indices,
        }
        return next_state
        #exit()
        #next_state = {"decoder_state": next_state["decoder_state"]
        #print(next_state.keys())


        next_state["output"] = (b_output, ("batch", "sequence"))
        next_state["cumulative_log_probability"] = (
            b_seq_lps, ("sequence", "batch", "placeholder")
        )
        next_state["beam_score"] = (
            b_scores, ("sequence", "batch", "placeholder")
        )
        next_state["beam_indices"] = (b_indices, ("batch"))

        return next_state

    def _next_candidates(self, batch_size, log_probs, candidates):
        # TODO seq_lps should really be called cumulative log probs.

        # flat_beam_lps (batch size x (beam size ** 2))
        flat_beam_lps = log_probs.view(batch_size, -1)

        flat_beam_scores = flat_beam_lps / (self.steps + 1)

        # beam_seq_scores (batch size x beam size)
        # relative_indices (batch_size x beam size)
        beam_seq_scores, relative_indices = torch.topk(
            flat_beam_scores, k=self.beam_size, dim=1)

        # beam_seq_lps (batch size x beam size)
        beam_seq_lps = flat_beam_lps.gather(1, relative_indices)

        # TODO make these ahead of time. 
        offset1 = (
            torch.arange(batch_size, device=beam_seq_lps.device) 
                * self.beam_size
        ).view(batch_size, 1)
        
        offset2 = offset1 * self.beam_size
       
        beam_indexing = ((relative_indices // self.beam_size) + offset1) \
            .view(-1)

        # beam_seq_lps (1 x (batch_size * beam_size) x 1)
        beam_seq_lps = beam_seq_lps \
            .view(1, batch_size * self.beam_size, 1)
        
        # beam_seq_scores (1 x (batch_size * beam_size) x 1)
        beam_seq_scores = beam_seq_scores \
            .view(1, batch_size * self.beam_size, 1)

        # next_output (1 x (batch size * beam size))
        next_candidate_indices = (relative_indices + offset2).view(-1)
        next_output = Variable(
            candidates.view(-1)[next_candidate_indices].view(1, -1),
            lengths=candidates.new().long().new(batch_size * self.beam_size)\
                .fill_(1),
            length_dim=0, batch_dim=1, pad_value=self.vocab.pad_index)


        return beam_seq_lps, beam_seq_scores, next_output, beam_indexing

    def check_termination(self, next_state, active_items):
        
        # view as batch size x beam size 
        next_output = next_state["output"].data \
            .view(-1, self.beam_size)
        batch_size = next_output.size(0)

        is_complete = next_output.eq(self.vocab.stop_index)
        complete_indices = np.where(is_complete.cpu().data.numpy())

        for batch, beam in zip(*complete_indices):
            if self._num_complete[batch] == self.beam_size:
                continue
            else:
                self._num_complete[batch] += 1

                # Store step and beam that finished so we can retrace it
                # later and recover arbitrary search state item.
                self._terminal_info[batch].append(
                    (self.steps, beam + batch * self.beam_size))
                
                IDX = batch * self.beam_size + beam
                self._beam_scores[batch].append(
                    next_state["beam_score"][0, IDX, 0].view(1))
        
        next_state["terminal_mask"] = (
            is_complete.view(1, batch_size * self.beam_size, 1)
        )   
        active_items = self._num_complete < self.beam_size

        return active_items

    def _collect_search_states(self, active_items):

        batch_size = active_items.size(0)

        last_state = self._states[-1]
        last_step = self.steps - 1
        for batch in range(batch_size):
            beam = 0 
            while len(self._beam_scores[batch]) < self.beam_size:
                IDX = batch * self.beam_size + beam
                self._beam_scores[batch].append(
                    last_state["beam_score"][0, IDX, 0].view(1))
                self._terminal_info[batch].append(
                    (last_step, beam + batch * self.beam_size))
                beam += 1

        # TODO consider removing beam indices from state
        beam_indices = torch.stack([state["beam_indices"] 
                                    for state in self._states])

        self._beam_scores = torch.stack([torch.cat(bs)
                                         for bs in self._beam_scores])
        
        lengths = self._states[0]["output"].new(
            [[step + 1 for step, beam in self._terminal_info[batch]]
             for batch in range(batch_size)])
        
        selector = self._states[0]["output"].new(
            batch_size, self.beam_size, lengths.max())
        mask = selector.new().byte().new(selector.size()).fill_(1)

        for batch in range(batch_size):
            for beam in range(self.beam_size):
                step, real_beam = self._terminal_info[batch][beam]
                mask[batch, beam,:step + 1].fill_(0)
                self._collect_beam(batch, real_beam, step, 
                                   beam_indices,
                                   selector[batch, beam])
        selector = selector.view(batch_size * self.beam_size, -1)

        ## RESORTING HERE ##
        #if self.sort_by_score:
        # TODO make this an option again
        if True:
            self._beam_scores, I = torch.sort(self._beam_scores, dim=1,
                                              descending=True)
            offset1 = (
                torch.arange(batch_size, device=I.device) * self.beam_size
            ).view(batch_size, 1)
            II = I + offset1
            selector = selector[II.view(-1)]
            mask = mask.view(batch_size * self.beam_size,-1)[II]\
                .view(batch_size, self.beam_size, -1)
            lengths = lengths.gather(1, I)
        ## 

        # TODO reimplement staged indexing         
#        for step, sel_step in enumerate(selector.split(1, dim=1)):
#            self._states[step].stage_indexing("batch", sel_step.view(-1))

        self._output = []
        for step, sel_step in enumerate(selector.split(1, dim=1)):
            self._output.append(
                self._states[step]["output"].index_select(1, sel_step.view(-1))
            )
        #print(self._states[0]["output"].size())
        self._output = plum.cat([o.data for o in self._output], 0).t()\
            .view(batch_size, self.beam_size, -1)
        
        for i in range(batch_size):
            for j in range(self.beam_size):
                self._output[i,j,lengths[i,j]:].fill_(self.vocab.pad_index)
        
        self._lengths = lengths

        return self        


#        for batch_out in self._output:
#        #print(self._output.t().view(batch_size)
#        #for batch_out in self._output.t().view(batch_size, self.beam_size, -1):
#            for row in batch_out:
#                print(" ".join([self.vocab[t] for t in row if t != self.vocab.pad_index]))
#            print()
#        print(lengths)
#        print(lengths.size())
#        print(batch_size)
#        exit()
#
#        states = self._states[0]
#        for state in self._states[1:]:
#            states.append(state)
#
#        self._states = states
#        self._selector = selector
#        self._lengths = lengths
#        self._selector_mask = mask.view(self.batch_size * self.beam_size, -1)
#        self._selector_mask_T = self._selector_mask.t().contiguous()

    def _collect_beam(self, batch, beam, step, beam_indices,
                      selector_out):        
        selection = [0] * beam_indices.size(0)
        selector_out[step + 1:].fill_(0) 
        while step >= 0:
            selection[step] = beam
            selector_out[step].fill_(beam)
            next_beam = beam_indices[step, beam].item()
            beam = next_beam
            step -= 1

    def output(self, as_indices=False, n_best=-1):
        if n_best < 1:
            o = self._output[:,0]
            if as_indices:
                return o
            tokens = []
            for row in o:
                tokens.append([self.vocab[t] for t in row
                               if t != self.vocab.pad_index])
            return tokens

        elif n_best < self.beam_size:
            o = self._output[:,:n_best]
        else:
            o = self._output

        if as_indices:
            return o

        beams = []
        for beam in o:
            tokens = []
            for row in beam:
                tokens.append([self.vocab[t] for t in row
                               if t != self.vocab.pad_index])
            beams.append(tokens)
        return beams
