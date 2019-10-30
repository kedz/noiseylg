from plum.types import register, PlumObject, HP, props, Variable
import torch


@register("seq2seq.search.ancestral_sampler")
class AncestralSampler(PlumObject):
    # Decode with ancestral sampling

    max_steps = HP(default=999999, type=props.INTEGER)
    samples = HP(default=25, type=props.INTEGER)
    vocab = HP()
    
    def __pluminit__(self):
        self.reset()

    def reset(self):
        self.is_finished = False
        self.steps = 0
        self._states = []
        self._outputs = []

    def init_state_context(self, encoder_state):
        batch_size = encoder_state["state"].size(1) * self.samples
        

        output = Variable(
            torch.LongTensor(
                [self.vocab.start_index] * batch_size
            ).view(1, -1),
            lengths=torch.LongTensor([1] * batch_size),
            length_dim=0, batch_dim=1, pad_value=self.vocab.pad_index)
            
        if str(encoder_state["state"].device) != "cpu":
            output = output.cuda(encoder_state["state"].device)

        layers = encoder_state["state"].size(0)
        decoder_state = encoder_state["state"].unsqueeze(2)\
            .repeat(1, 1, self.samples, 1).view(layers, batch_size, -1)
        search_state = {"output": output, "decoder_state": decoder_state}

        encoder_output = encoder_state["output"].repeat_batch_dim(self.samples)
        context = {
            "encoder_output": encoder_output
        }

        return search_state, context

    def next_state(self, decoder, prev_state, context, active_items,
                   controls):

        # Get next state from the decoder.
        next_state = decoder.next_state(prev_state, context, controls=controls)
        
        log_probs = next_state["log_probs"].tensor
        samples = torch.distributions.Categorical(
            logits=next_state["target_logits"].tensor).sample()
            
        sample_output = Variable(
            samples,
            lengths=samples.new(samples.size(1)).fill_(1),
            length_dim=0, batch_dim=1, pad_value=self.vocab.pad_index)
        
        output_log_probs = log_probs.gather(2, samples.unsqueeze(-1))\
            .squeeze(2)

        next_state["output"] = sample_output
        next_state["output_log_probs"] = output_log_probs

        # Mask outputs if we have already completed that batch item. 
        next_state["output"].data.view(-1).masked_fill_(
            ~active_items, self.vocab.pad_index)

        return next_state

    def check_termination(self, next_state, active_items):

        # Check for stop tokens and batch item inactive if so.
        nonstop_tokens = next_state["output"].data.view(-1).ne(
            self.vocab.stop_index)
        active_items = active_items.data.mul_(nonstop_tokens)

        return active_items

    def _collect_search_states(self, active_items):
        # TODO implement search states api.
        #search_state = self._states[0]
        #for next_state in self._states[1:]:
        #    search_state.append(next_state)
        #self._states = search_state
        self._outputs = torch.cat([o.data for o in self._outputs], dim=0)
        self._output_log_probs = torch.cat(
            [state["output_log_probs"].data 
             for state in self._states],
            0)
        self._output_log_probs = self._output_log_probs.masked_fill(
            self._mask_T, 0)
        avg_log_probs = (
            self._output_log_probs.sum(0) / (~self._mask_T).float().sum(0)
        ).view(-1, self.samples)
        avg_log_probs, argsort = avg_log_probs.sort(1, descending=True)

        batch_size = avg_log_probs.size(0)
        offsets = (torch.arange(0, batch_size, device=argsort.device) \
            * self.samples).view(-1, 1)
        reindex = argsort + offsets

        self._outputs = self._outputs.index_select(1, reindex.view(-1))\
            .view(-1, batch_size, self.samples).permute(1, 2, 0)
        self._output_log_probs = self._output_log_probs.index_select(
            1, reindex.view(-1)).view(-1, batch_size, self.samples)\
            .permute(1, 2, 0)
        self._mask_T = None
        self._mask = None
        self._avg_log_probs = avg_log_probs
 
    def __call__(self, decoder, encoder_state, controls=None):

        self.reset()
        # TODO get batch size in a more reliable way. This will probably break
        # for cnn or transformer based models.
        batch_size = encoder_state["state"].size(1)
        search_state, context = self.init_state_context(encoder_state)

        active_items = search_state["decoder_state"]\
            .new(batch_size * self.samples).byte().fill_(1)

        step_masks = []
        # Perform search until we either trigger a termination condition for
        # each batch item or we reach the maximum number of search steps.
        while self.steps < self.max_steps and not self.is_finished:
            
            inactive_items = ~active_items

            # Mask any inputs that are finished, so that greedy would 
            # be identitcal to forward passes. 
            search_state["output"].data.view(-1).masked_fill_(
                inactive_items, self.vocab.pad_index)

            step_masks.append(inactive_items)
            self.steps += 1
            search_state = self.next_state(
                decoder, search_state, context, active_items, controls)        
            
            self._states.append(search_state)
            self._outputs.append(search_state["output"].clone())

            active_items = self.check_termination(search_state, active_items)
            self.is_finished = torch.all(~active_items)

        # Finish the search by collecting final sequences, and other 
        # stats. 

        self._mask_T = torch.stack(step_masks)
        self._mask = self._mask_T.t().contiguous()
        self._collect_search_states(active_items)
        self._incomplete_items = active_items
        self._is_finished = True

        return self
        
    def __getitem__(self, key):
        if key == "output":
            return self._outputs

    def output(self, as_indices=False, n_best=-1):
        if n_best < 1:
            o = self._outputs[:,0]
            if as_indices:
                return o
            tokens = []
            for row in o:
                tokens.append([self.vocab[t] for t in row
                               if t != self.vocab.pad_index])
            return tokens

        elif n_best < self.samples:
            o = self._outputs[:,:n_best]
        else:
            o = self._outputs

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
