from ..types import register, PlumModule, HP, SM, LazyDict, Variable
from .plum_model import PlumModel


@register("plum.models.encoder_decoder")
class EncoderDecoder(PlumModel):
    
    encoder_inputs = HP(default=["source_inputs"])
    encoder = SM()

    decoder_inputs = HP(default="target_inputs")
    decoder = SM()

    control_inputs = HP(default=[])
    controls = SM(required=False)

    search_algos = HP(required=False, default={})

    def forward(self, batch):
        # TODO make everything lazy.
        
        ctrl_args = [batch[input] for input in self.control_inputs]
        ctrls = self.controls(ctrl_args) if self.controls else None
        
        encoder_args = [batch[input] for input in self.encoder_inputs]
        encoder_state = self.encoder(encoder_args)

        decoder_args = batch[self.decoder_inputs]
        if "num_references" in batch:  
            if ctrls is not None:
                raise Exception("Multiref not implmeneted with controls yet.")
            num_refs = batch["num_references"].max()
            multiref_encoder_states = self._multiref_state(
                encoder_state, num_refs)
            decoder_state = self.decoder(decoder_args, 
                                         multiref_encoder_states)

        else:        
            decoder_state = self.decoder(decoder_args, encoder_state,
                                         controls=ctrls)

        # Optionally we may want to search instead of evaluate likelihoods
        # on gold data so we set up the searches here, to only be evaluated
        # if they are needed later by looking them up in decoder_state.
        decoder_state["search"] = LazyDict()
        for name, algo in self.search_algos.items():
            def make_search(algo, decoder, encoder_state, ctrls):
                def search_func():
                    return algo(decoder, encoder_state, controls=ctrls)
                return search_func
            decoder_state["search"].lazy_set(
                name, make_search(algo, self.decoder, encoder_state, ctrls))

        return decoder_state

    def _multiref_state(self, encoder_state, num_refs):
        new_state = {}

        new_state["output"] = encoder_state["output"].repeat_batch_dim(
            num_refs)

        parts, batch_size, hidden_size = encoder_state["state"].size()
        new_state["state"] = encoder_state["state"].unsqueeze(2)\
            .contiguous()\
            .repeat(1, 1, num_refs, 1)\
            .view(parts, batch_size * num_refs, hidden_size)

        return new_state

    def encode(self, batch):

        encoder_args = [batch[input] for input in self.encoder_inputs]
        encoder_state = self.encoder(encoder_args)
        
        if self.controls:
            ctrl_args = [batch[input] for input in self.control_inputs]
            ctrls = self.controls(ctrl_args) if self.controls else None
            return encoder_state, ctrls
        else:
            return encoder_state

    def log_likelihood(self, inputs, targets, reduction='none'):
        forward_state = self.forward(inputs)
        log_probs = forward_state["log_probs"]
        log_probs = log_probs.permute_as_batch_sequence_features()
        targets = targets.permute_as_batch_sequence_features()
        
        el_log_probs = Variable(
            log_probs.tensor.gather(2, targets.tensor.unsqueeze(2)).squeeze(2),
            lengths=targets.lengths,
            batch_dim=0, length_dim=1,
            pad_value=float("-inf")).apply_sequence_mask()
        
        if reduction == "none":
            return el_log_probs
        else:
            return el_log_probs.reduce_sequence(reduction)
