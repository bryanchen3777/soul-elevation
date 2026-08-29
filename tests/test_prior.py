"""先验映射表（trigger_type → node_type 先验）单元测试。"""

from soul_elevation.prior import (
    CATEGORY_PRIOR_TABLE,
    CATEGORY_TRIGGER_TYPES,
    DEFAULT_PRIOR,
    PRIOR_TABLE,
    resolve_prior,
)


def test_news_maps_to_belief():
    assert resolve_prior("world:news_event") == ("belief",)


def test_leisure_maps_to_essence_trait():
    assert resolve_prior("world:calendar_event") == ("essence", "trait")
    assert resolve_prior("user_going_outside") == ("essence", "trait")


def test_diary_maps_to_value_trait_belief():
    assert resolve_prior("diary:morning") == ("value", "trait", "belief")
    assert resolve_prior("diary:night") == ("value", "trait", "belief")


def test_dream_maps_to_essence_trait():
    assert resolve_prior("dream:dream") == ("essence", "trait")
    assert resolve_prior("dream:event") == ("essence", "trait")


def test_conversation_by_category():
    assert resolve_prior(
        "conversation:user_message",
        {"llm_judge": {"category": "preference_plan_event_fact"}},
    ) == ("belief",)
    assert resolve_prior(
        "conversation:user_message",
        {"llm_judge": {"category": "milestone"}},
    ) == ("essence",)
    assert resolve_prior(
        "conversation:user_message",
        {"llm_judge": {"category": "diary"}},
    ) == ("value", "trait")


def test_memory_fact_by_category():
    assert resolve_prior("memory_fact", {"category": "preference_plan_event_fact"}) == ("belief",)
    assert resolve_prior("memory_fact", {"category": "milestone"}) == ("essence",)


def test_category_triggers_are_category_dependent():
    assert CATEGORY_TRIGGER_TYPES == frozenset({"conversation:user_message", "memory_fact"})


def test_unknown_trigger_type_falls_back_to_default():
    assert resolve_prior("unknown:type") == DEFAULT_PRIOR
    assert resolve_prior("unknown:type") == ("belief",)


def test_unknown_or_missing_category_falls_back_to_default():
    assert resolve_prior(
        "conversation:user_message",
        {"llm_judge": {"category": "weird_category"}},
    ) == DEFAULT_PRIOR
    assert resolve_prior("conversation:user_message", {}) == DEFAULT_PRIOR
    assert resolve_prior("conversation:user_message", None) == DEFAULT_PRIOR


def test_prior_table_is_explicit_mapping_not_ifelse():
    # 显式映射表（可扩展），不是散落的 if-else。
    assert isinstance(PRIOR_TABLE, dict)
    assert isinstance(CATEGORY_PRIOR_TABLE, dict)
    assert PRIOR_TABLE["world:news_event"] == ("belief",)
    assert PRIOR_TABLE["diary:night"] == ("value", "trait", "belief")
    assert CATEGORY_PRIOR_TABLE["preference_plan_event_fact"] == ("belief",)
    assert CATEGORY_PRIOR_TABLE["milestone"] == ("essence",)
    assert CATEGORY_PRIOR_TABLE["diary"] == ("value", "trait")
