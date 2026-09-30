import torch

from Telugu.config.model_config import ModelConfig
from Telugu.model.lm import TeluguLanguageModel


def main():

    print("=" * 80)
    print("TELUGU MODEL H — LOCAL MODEL TEST")
    print("=" * 80)

    config = ModelConfig()

    print("\nConfiguration:")
    print(f"Vocabulary size : {config.vocab_size}")
    print(f"d_model         : {config.d_model}")
    print(f"Layers          : {config.n_layers}")
    print(f"Heads           : {config.n_heads}")
    print(f"Head dimension  : {config.head_dim}")
    print(f"FFN dimension   : {config.d_ff}")
    print(f"Context length  : {config.max_seq_len}")

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = TeluguLanguageModel(config).to(device)

    # ---------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------

    num_parameters = model.count_parameters()

    print(
        f"\nTrainable parameters: "
        f"{num_parameters:,}"
    )

    expected = 23_042_432

    assert num_parameters == expected, (
        f"Expected {expected:,} parameters, "
        f"but got {num_parameters:,}."
    )

    print("✓ Parameter count correct")

    # ---------------------------------------------------------
    # Construct explicitly aligned causal-LM example
    #
    # Full sequence:
    #
    # [101, 202, 303, 404, 505, 606]
    #
    # Input:
    #
    # [101, 202, 303, 404, 505]
    #
    # Target:
    #
    # [202, 303, 404, 505, 606]
    # ---------------------------------------------------------

    full_sequence = torch.tensor(
        [[
            101,
            202,
            303,
            404,
            505,
            606,
        ]],
        dtype=torch.long,
        device=device,
    )

    input_ids = full_sequence[:, :-1]
    targets = full_sequence[:, 1:]

    print(
        f"\nFull sequence : {tuple(full_sequence.shape)}"
    )

    print(
        f"Input shape   : {tuple(input_ids.shape)}"
    )

    print(
        f"Target shape  : {tuple(targets.shape)}"
    )

    assert input_ids.shape == targets.shape

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    logits, loss, _ = model(
        input_ids=input_ids,
        targets=targets,
    )

    print(
        f"Logits shape  : {tuple(logits.shape)}"
    )

    print(
        f"Loss          : {loss.item():.6f}"
    )

    assert logits.shape == (
        1,
        5,
        config.vocab_size,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print("✓ Forward pass correct")

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------

    loss.backward()

    gradients_found = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all()

            gradients_found += 1

    assert gradients_found > 0

    print("✓ Backward pass correct")

    # ---------------------------------------------------------
    # Weight tying
    # ---------------------------------------------------------

    assert (
        model.token_embedding.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    )

    print("✓ Input embedding and LM head are tied")

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("ALL TELUGU MODEL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()