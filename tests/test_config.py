"""Tests for the config layer.

The claim under test is "a new project is a config file, not a fork". These
check the mechanics that make it true: overlays, name-addressed overrides, open
params, secret handling, and the cross-stage checks that catch a config which is
structurally valid but violates an invariant.
"""

from __future__ import annotations

import os

import pytest

from indexer.config import load
from indexer.config.loader import config_hash, load_mapping, redact, with_overrides
from indexer.core.errors import ConfigError

REFERENCE = "configs/reference.yaml"
FULL = "configs/full.yaml"


@pytest.fixture(autouse=True)
def _fake_secrets() -> None:
    os.environ.setdefault("LLAMAPARSE_API_KEY", "sk-test")


class TestReferenceConfig:
    def test_loads(self) -> None:
        cfg, _ = load(REFERENCE)
        assert cfg.project.name == "reference"

    def test_has_the_three_mandatory_route_paths(self) -> None:
        cfg, _ = load(REFERENCE)
        assert {"structured", "lookup", "iterative"} <= set(cfg.query.route.paths)

    def test_ships_an_ablation_ladder_and_the_two_sanity_checks(self) -> None:
        cfg, _ = load(REFERENCE)
        assert len(cfg.eval.ablations) >= 5
        assert len(cfg.eval.sanity_checks) == 2

    def test_warns_when_rerank_is_off(self) -> None:
        cfg, _ = load(REFERENCE)
        assert any("rerank is off" in w for w in cfg.warnings())


class TestOverlays:
    def test_full_inherits_from_reference(self) -> None:
        cfg, _ = load(FULL)
        assert cfg.paths.store == "./var/index"  # inherited
        assert len(cfg.eval.ablations) == 6  # inherited
        assert cfg.query.rerank.enabled is True  # overridden

    def test_lists_replace_rather_than_append(self) -> None:
        """Otherwise 'run without the contextualiser' is inexpressible."""
        cfg, _ = load(FULL)
        impls = [e.impl for e in cfg.ingestion.enrich.enrichers]
        assert "section_prefix" not in impls  # reference's list is replaced, not merged

    def test_circular_extends_is_caught(self, tmp_path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("extends: b.yaml\n")
        b.write_text("extends: a.yaml\n")
        with pytest.raises(ConfigError, match="circular"):
            load_mapping(a)


class TestInterpolation:
    def test_env_default_is_used_when_unset(self) -> None:
        os.environ.pop("EMBED_DEVICE", None)
        cfg, _ = load(FULL)
        dense = next(i for i in cfg.ingestion.index.indexes if i.name == "dense")
        assert dense.params["device"] == "cpu"  # ${env:EMBED_DEVICE:cpu}

    def test_missing_secret_fails_at_load_not_mid_build(self) -> None:
        saved = os.environ.pop("LLAMAPARSE_API_KEY")
        try:
            with pytest.raises(ConfigError, match="unset and has no default"):
                load(FULL)
        finally:
            os.environ["LLAMAPARSE_API_KEY"] = saved

    def test_internal_reference_resolves(self) -> None:
        cfg, _ = load(FULL)
        fields = next(i for i in cfg.ingestion.index.indexes if i.name == "fields")
        assert fields.params["path"] == "./var/index/fields.db"  # ${paths.store}


class TestSecrets:
    def test_redaction_strips_secret_shaped_keys(self) -> None:
        red = redact({"params": {"api_key": "sk-live", "url": "http://x"}})
        assert red["params"]["api_key"] == "<redacted>"
        assert red["params"]["url"] == "http://x"

    def test_key_rotation_does_not_invalidate_the_index(self) -> None:
        """A secret is not part of what a config means."""
        os.environ["LLAMAPARSE_API_KEY"] = "sk-one"
        _, r1 = load(FULL)
        os.environ["LLAMAPARSE_API_KEY"] = "sk-two"
        _, r2 = load(FULL)
        assert config_hash(r1) == config_hash(r2)


class TestOverrides:
    def test_named_list_elements_are_addressable(self) -> None:
        """Ablation arms cannot depend on an index's position in a list."""
        raw = load_mapping(REFERENCE)
        out = with_overrides(raw, {"ingestion.index.indexes[lexical].enabled": False})
        by_name = {i["name"]: i for i in out["ingestion"]["index"]["indexes"]}
        assert by_name["lexical"]["enabled"] is False
        assert by_name["dense"].get("enabled", True) is True

    def test_unknown_name_is_an_error(self) -> None:
        raw = load_mapping(REFERENCE)
        with pytest.raises(ConfigError, match="no list element named"):
            with_overrides(raw, {"ingestion.index.indexes[nope].enabled": False})

    def test_ablation_arm_applies_end_to_end(self) -> None:
        cfg, _ = load(REFERENCE, overrides={"ingestion.enrich.enabled": False})
        assert cfg.enrich_enabled is False
        assert any("contextualisation" in w for w in cfg.warnings())


class TestCrossStageValidation:
    """Checks no single stage could make for itself."""

    def test_unknown_key_is_rejected(self) -> None:
        """A misspelled key that is accepted and does nothing is the worst bug."""
        raw = load_mapping(REFERENCE)
        raw["query"]["rerank"]["inputtopk"] = 10
        with pytest.raises(ConfigError, match=r"[Ee]xtra"):
            from indexer.config.schema import Config

            try:
                Config.model_validate(raw)
            except Exception as exc:
                raise ConfigError(str(exc)) from exc

    def test_structured_path_without_a_structured_index_is_rejected(self) -> None:
        """Invariant 5 is a property of the configuration, not just the router."""
        raw = load_mapping(REFERENCE)
        raw["query"]["route"]["paths"]["structured"]["targets"] = ["dense"]
        with pytest.raises(ConfigError, match="would reach vector"):
            from indexer.config.schema import Config

            try:
                Config.model_validate(raw)
            except Exception as exc:
                raise ConfigError(str(exc)) from exc

    def test_all_indexes_disabled_is_rejected(self) -> None:
        raw = load_mapping(REFERENCE)
        for i in raw["ingestion"]["index"]["indexes"]:
            i["enabled"] = False
        with pytest.raises(ConfigError, match="nothing to retrieve"):
            from indexer.config.schema import Config

            try:
                Config.model_validate(raw)
            except Exception as exc:
                raise ConfigError(str(exc)) from exc

    def test_lookup_path_must_be_one_pass(self) -> None:
        raw = load_mapping(REFERENCE)
        raw["query"]["route"]["paths"]["lookup"]["step_budget"] = 3
        with pytest.raises(ConfigError, match="one retrieval pass"):
            from indexer.config.schema import Config

            try:
                Config.model_validate(raw)
            except Exception as exc:
                raise ConfigError(str(exc)) from exc


class TestOpenParams:
    def test_unknown_impl_params_pass_phase_one(self) -> None:
        """Phase 1 must not enumerate implementations, or adding one is a fork."""
        raw = load_mapping(REFERENCE)
        raw["ingestion"]["index"]["indexes"].append(
            {
                "name": "experimental",
                "kind": "graph",  # a kind the frame has never heard of
                "impl": "nobody_has_written_this_yet",
                "params": {"hops": 3, "whatever": {"nested": True}},
            }
        )
        from indexer.config.schema import Config

        cfg = Config.model_validate(raw)
        assert cfg.ingestion.index.indexes[-1].params["hops"] == 3


class TestOverridePurity:
    """An ablation runner applies many arms to one snapshot of a config.

    If applying an arm mutates that snapshot, every later arm inherits the
    earlier one's overrides -- so the arms differ in ways their override lists
    do not state, and every row of the delta table becomes unattributable. The
    failure is silent: the numbers still look like numbers.
    """

    def test_overrides_do_not_mutate_the_input(self) -> None:
        raw = load_mapping(REFERENCE)
        before = json_snapshot(raw)
        with_overrides(
            raw,
            {
                "ingestion.index.indexes[lexical].enabled": False,
                "query.route.paths.lookup.targets": ["dense"],
                "ingestion.segment.max_tokens": 99,
            },
        )
        assert json_snapshot(raw) == before

    def test_two_arms_from_one_snapshot_are_independent(self) -> None:
        raw = load_mapping(REFERENCE)
        arm_a = with_overrides(raw, {"ingestion.index.indexes[dense].enabled": False})
        arm_b = with_overrides(raw, {"ingestion.segment.max_tokens": 128})
        a_dense = next(i for i in arm_a["ingestion"]["index"]["indexes"] if i["name"] == "dense")
        b_dense = next(i for i in arm_b["ingestion"]["index"]["indexes"] if i["name"] == "dense")
        assert a_dense["enabled"] is False
        assert b_dense.get("enabled", True) is True  # arm A did not leak into arm B
        assert arm_a["ingestion"]["segment"]["max_tokens"] != 128


def json_snapshot(obj: object) -> str:
    import json

    return json.dumps(obj, sort_keys=True, default=str)


class TestInertStructuredPath:
    """A structured index with nothing in it is the quietest way to lose
    invariant 5: the config validates, but the router has no field vocabulary,
    so every structured question is classified as prose and sent to vector
    search. Nothing errors -- the questions just fail."""

    def test_warns_when_no_enricher_extracts_fields(self) -> None:
        from indexer.config.schema import Config

        raw = load_mapping(REFERENCE)
        raw["ingestion"]["enrich"]["enrichers"] = [{"impl": "section_prefix", "scope": "unit"}]
        cfg = Config.model_validate(raw)
        assert cfg.extracted_field_names() == []
        assert any("no enabled enricher declares any field" in w for w in cfg.warnings())

    def test_no_warning_once_extraction_is_configured(self) -> None:
        cfg, _ = load(REFERENCE)
        assert cfg.extracted_field_names()
        assert not any("no enabled enricher declares any field" in w for w in cfg.warnings())

    def test_the_lexicon_the_router_gets_is_the_one_the_check_reads(self) -> None:
        """One derivation, so the warning and the behaviour cannot drift apart."""
        from indexer.pipeline.build import assemble

        cfg, _ = load(REFERENCE)
        assembly = assemble(REFERENCE)
        assert assembly._field_lexicon() == cfg.extracted_field_names()


def test_checkpoint_interval_is_configurable() -> None:
    """Resumability granularity is an operational choice, not a constant."""
    from indexer.pipeline.build import assemble

    cfg, _ = load(REFERENCE)
    assert cfg.ingestion.checkpoint_every == 200
    a = assemble(REFERENCE, overrides={"ingestion.checkpoint_every": 25})
    assert a.ingestion().checkpoint_every == 25
