import torch

from Telugu.config.model_config import ModelConfig
from Telugu.model.lm import TeluguLanguageModel


def main():

    print("=" * 80)
    print("TELUGU — CAUSAL MASK VERIFICATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    torch.manual_seed(42)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    config = ModelConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    model = TeluguLanguageModel(config).to(device)

    # Evaluation mode disables dropout so that the comparison
    # is affected only by the changed input token.
    model.eval()

    # ---------------------------------------------------------
    # Original sequence
    # ---------------------------------------------------------

    sequence = torch.tensor(
        [[
            101,
            202,
            303,
            404,
            505,
            606,
            707,
            808,
        ]],
        dtype=torch.long,
        device=device,
    )

    # ---------------------------------------------------------
    # Modified sequence
    #
    # Only position 6 is changed.
    #
    # If causal masking works correctly:
    #
    #   logits at positions 0-5
    #
    # must remain unchanged.
    # ---------------------------------------------------------

    modified_sequence = sequence.clone()

    changed_position = 6

    modified_sequence[0, changed_position] = 999

    print(
        f"Changed token position: {changed_position}"
    )

    print(
        f"Original token: "
        f"{sequence[0, changed_position].item()}"
    )

    print(
        f"Modified token: "
        f"{modified_sequence[0, changed_position].item()}"
    )

    # ---------------------------------------------------------
    # Forward passes
    # ---------------------------------------------------------

    with torch.no_grad():

        original_logits, _, _ = model(
            input_ids=sequence,
        )

        modified_logits, _, _ = model(
            input_ids=modified_sequence,
        )

    # ---------------------------------------------------------
    # Verify positions BEFORE the changed token
    #
    # Position i must not be able to attend to position 6
    # when i < 6.
    # ---------------------------------------------------------

    print("\nChecking positions before changed token:")

    max_difference_before = 0.0

    for position in range(changed_position):

        difference = torch.max(
            torch.abs(
                original_logits[:, position, :]
                - modified_logits[:, position, :]
            )
        ).item()

        max_difference_before = max(
            max_difference_before,
            difference,
        )

        print(
            f"Position {position}: "
            f"max logit difference = "
            f"{difference:.12e}"
        )

        # Numerical tolerance for floating-point computation.
        assert difference < 1e-6, (
            f"Causal masking failed at position "
            f"{position}: difference = {difference}"
        )

    # ---------------------------------------------------------
    # Verify the changed position itself
    #
    # Position 6 IS allowed to see its own token, so its logits
    # are expected to potentially change.
    # ---------------------------------------------------------

    same_position_difference = torch.max(
        torch.abs(
            original_logits[:, changed_position, :]
            - modified_logits[:, changed_position, :]
        )
    ).item()

    print(
        f"\nPosition {changed_position}: "
        f"max logit difference = "
        f"{same_position_difference:.12e}"
    )

    # We don't require this difference to be non-zero because
    # that depends on the model's learned parameters.
    # The important requirement is that earlier positions
    # do not change.

    # ---------------------------------------------------------
    # Verify a later position
    #
    # Position 7 CAN see position 6, so it is allowed to change.
    # ---------------------------------------------------------

    later_position = changed_position + 1

    later_position_difference = torch.max(
        torch.abs(
            original_logits[:, later_position, :]
            - modified_logits[:, later_position, :]
        )
    ).item()

    print(
        f"Position {later_position}: "
        f"max logit difference = "
        f"{later_position_difference:.12e}"
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print(
        f"\nMaximum difference before changed token: "
        f"{max_difference_before:.12e}"
    )

    print("\n✓ Future-token masking verified")

    print("=" * 80)
    print("CAUSAL MASK TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()