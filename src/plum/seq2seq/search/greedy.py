from plum.types import register, PlumObject, HP, props, Variable
import torch


@register("seq2seq.search.greedy")
class GreedySearch(PlumObject):

    max_steps = HP(default=999999, type=props.INTEGER)
    vocab = HP()
    
    def __pluminit__(self):
        self.reset()

    def reset(self):
        self.is_finished = False
        self.steps = 0
        self._states = []
        self._outputs = []

    def init_state(self, batch_size, encoder_state):
        output = Variable(
            torch.LongTensor([self.vocab.start_index] * batch_size)\
                .view(1, -1),
            lengths=torch.LongTensor([1] * batch_size),
            length_dim=0, batch_dim=1, pad_value=self.vocab.pad_index)
            
        if str(encoder_state.device) != "cpu":
            output = output.cuda(encoder_state.device)

        return {"output": output, "decoder_state": encoder_state}

    def next_state(self, decoder, prev_state, context, active_items,
                   controls):

        # Get next state from the decoder.
        next_state = decoder.next_state(prev_state, context, controls=controls)

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
 
    def __call__(self, decoder, encoder_state, controls=None):

        self.reset()
        # TODO get batch size in a more reliable way. This will probably break
        # for cnn or transformer based models.
        batch_size = encoder_state["state"].size(1)
        search_state = self.init_state(batch_size, encoder_state["state"])
        context = {
            "encoder_output": encoder_state["output"],
        }
        active_items = search_state["decoder_state"].new(batch_size).byte() \
            .fill_(1)

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
        self._collect_search_states(active_items)
        self._incomplete_items = active_items
        self._is_finished = True

        self._mask_T = torch.stack(step_masks)
        self._mask = self._mask_T.t().contiguous()
        return self
        
    def __getitem__(self, key):
        if key == "output":
            return self._outputs

    def output(self, as_indices=False):
        
        if as_indices:
            return self._outputs.t()

        tokens = []
        for output in self._outputs.t():
            tokens.append(
                [self.vocab[index] for index in output 
                 if index != self.vocab.pad_index]
            )

        return tokens
