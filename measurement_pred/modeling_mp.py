from typing import Optional, Tuple, Union

import torch
from torch.nn import BCEWithLogitsLoss
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast

class MeasurementPredictorMixin(PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.sensor_token = config.sensor_token
        self.sensor_token_id = config.sensor_token_id
        self.n_sensors = config.n_sensors
        self.sensor_probes = torch.nn.ModuleList([
            torch.nn.Linear(config.n_embd, 1) for _ in range(config.n_sensors)
        ])
        self.use_aggregated = config.use_aggregated
        if config.use_aggregated:
            self.aggregate_probe = torch.nn.Linear(config.n_embd, 1)
        self.sensors_weight = config.sensors_weight
        self.aggregate_weight = config.aggregate_weight
    
    def check_tokenizer(self, tokenizer: PreTrainedTokenizer):
        sensor_token_id = tokenizer.tokenize(self.sensor_token)[0]
        assert sensor_token_id == self.sensor_token_id
    
    def set_sensor_token(self, sensor_token: str, tokenizer: PreTrainedTokenizer):
        sensor_token_id = tokenizer.tokenize(sensor_token)[0]
        self.sensor_token = sensor_token
        self.sensor_token_id = sensor_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        base_model_output: BaseModelOutputWithPast = self.base_model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        tensor_token_mask = torch.where(input_ids == self.sensor_token_id)
        sensor_embs = base_model_output.last_hidden_state[:, tensor_token_mask, :] # TODO: check (should probably write this outside definition)
        sensor_logits = torch.concat([self.sensor_probes[i](sensor_embs[:, i, :]) 
                               for i in range(self.n_sensors)])
        logits = sensor_logits

        if self.use_aggregated:
            last_emb = base_model_output[:, -1, :]
            aggregate_logits = self.aggregate_probe(last_emb)
            logits = torch.concat([logits, aggregate_logits])
        
        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            sensor_loss = loss_fct(sensor_logits, labels[:, :, self.n_sensors]) * self.sensors_weight
            loss = sensor_loss
            if self.use_aggregated: #TOOD: should be use aggregate
                aggregate_loss = loss_fct(aggregate_logits, labels[:, :, -1]) * self.aggregate_weight
                loss += aggregate_loss

        if not return_dict:
            output = (logits, ) + base_model_output[1:]
            return ((loss,) + output) if loss is not None else output 
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_model_output.past_key_values,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )

