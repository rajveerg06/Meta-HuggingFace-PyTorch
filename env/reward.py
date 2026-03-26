from __future__ import annotations

from models.schemas import ActionType, DocumentSample


def step_reward_for_action(
    action: ActionType,
    action_correct: bool,
    loop_penalty: float,
    sample: DocumentSample,
) -> float:
    base_rewards = {
        ActionType.CLASSIFY_DOCUMENT: 0.25,
        ActionType.EXTRACT_FIELDS: 0.45,
        ActionType.VALIDATE_FIELDS: 0.2,
        ActionType.FINISH: 0.1,
    }
    base = base_rewards.get(action, 0.0)

    if action == ActionType.FINISH:
        return max(-0.15, base - loop_penalty)

    if action_correct:
        difficulty_boost = {"easy": 1.0, "medium": 1.05, "hard": 1.1}[sample.difficulty]
        return max(-0.2, (base * difficulty_boost) - loop_penalty)

    return max(-0.2, (-0.1 * (1.0 + loop_penalty)))
