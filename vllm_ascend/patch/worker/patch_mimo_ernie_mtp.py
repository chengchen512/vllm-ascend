import torch

try:
    import vllm.model_executor.models.ernie_mtp as ernie_mtp
except ImportError:
    ernie_mtp = None

try:
    import vllm.model_executor.models.mimo_mtp as mimo_mtp
except ImportError:
    mimo_mtp = None


def _mask_first_mtp_position(
    inputs_embeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    return torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)


if mimo_mtp is not None:

    class AscendMiMoMultiTokenPredictorLayer(mimo_mtp.MiMoMultiTokenPredictorLayer):
        def forward(
            self,
            inputs_embeds: torch.Tensor,
            positions: torch.Tensor,
            previous_hidden_states: torch.Tensor,
            spec_step_index: int = 0,
        ) -> torch.Tensor:
            assert inputs_embeds is not None
            # Avoid boolean assignment because it lowers to NonZero, which cannot
            # synchronize streams during ACLGraph capture on Ascend.
            inputs_embeds = _mask_first_mtp_position(inputs_embeds, positions)
            inputs_embeds = self.token_layernorm(inputs_embeds)
            previous_hidden_states = self.hidden_layernorm(previous_hidden_states)

            hidden_states = self.input_proj(
                torch.cat([previous_hidden_states, inputs_embeds], dim=-1)
            )

            hidden_states, residual = self.mtp_block(
                positions=positions, hidden_states=hidden_states, residual=None
            )
            hidden_states = residual + hidden_states
            return self.final_layernorm(hidden_states)

    mimo_mtp.MiMoMultiTokenPredictorLayer = AscendMiMoMultiTokenPredictorLayer


if ernie_mtp is not None:

    class AscendErnieMultiTokenPredictorLayer(ernie_mtp.ErnieMultiTokenPredictorLayer):
        def forward(
            self,
            inputs_embeds: torch.Tensor,
            positions: torch.Tensor,
            previous_hidden_states: torch.Tensor,
            spec_step_index: int = 0,
        ) -> torch.Tensor:
            assert inputs_embeds is not None
            # Avoid boolean assignment because it lowers to NonZero, which cannot
            # synchronize streams during ACLGraph capture on Ascend.
            inputs_embeds = _mask_first_mtp_position(inputs_embeds, positions)

            inputs_embeds = self.mtp_emb_norm(inputs_embeds)
            previous_hidden_states = self.mtp_hidden_norm(previous_hidden_states)

            hidden_states = self.mtp_linear_proj(
                torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
            )

            hidden_states, residual = self.mtp_block(
                positions=positions, hidden_states=hidden_states, residual=None
            )
            hidden_states = residual + hidden_states

            return hidden_states

    ernie_mtp.ErnieMultiTokenPredictorLayer = AscendErnieMultiTokenPredictorLayer
