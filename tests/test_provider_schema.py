# pyright: reportMissingTypeStubs=false, reportMissingImports=false

from pathlib import Path
from typing import TypedDict, cast

import yaml
from dify_plugin.entities.model import ModelFeature, ModelPropertyKey

from models.llm.llm import Sub2apiPluginLargeLanguageModel


REPO_ROOT = Path(__file__).resolve().parents[1]
PROVIDER_SCHEMA_PATH = REPO_ROOT / "provider" / "sub2api-plugin.yaml"
PREDEFINED_MODEL_SCHEMA_PATH = REPO_ROOT / "models" / "llm" / "llm.yaml"
MANIFEST_PATH = REPO_ROOT / "manifest.yaml"
LLM_IMPLEMENTATION_PATH = REPO_ROOT / "models" / "llm" / "llm.py"
README_PATH = REPO_ROOT / "README.md"
GUIDE_PATH = REPO_ROOT / "GUIDE.md"


class I18nSchema(TypedDict):
    en_US: str
    zh_Hans: str
    ja_JP: str


class CredentialFormSchema(TypedDict):
    variable: str


class ModelCredentialSchema(TypedDict):
    credential_form_schemas: list[CredentialFormSchema]


class PythonExtraSchema(TypedDict):
    provider_source: str


class ExtraSchema(TypedDict):
    python: PythonExtraSchema


class ProviderSchemaRequired(TypedDict):
    name: str
    description: I18nSchema
    supported_model_types: list[str]
    configurate_methods: list[str]
    model_credential_schema: ModelCredentialSchema
    help: dict[str, I18nSchema]
    extra: ExtraSchema


class ProviderSchema(ProviderSchemaRequired, total=False):
    provider_credential_schema: object
    models: object


def load_provider_schema() -> ProviderSchema:
    with PROVIDER_SCHEMA_PATH.open("r", encoding="utf-8") as file:
        return cast(ProviderSchema, yaml.safe_load(file))


def test_provider_schema_only_keeps_responses_customizable_contract() -> None:
    provider_schema = load_provider_schema()
    manifest_schema = cast(dict[str, object], yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")))

    assert manifest_schema["name"] == "openai-responseapi-dify-plugin"
    assert provider_schema["supported_model_types"] == ["llm"]
    assert provider_schema["configurate_methods"] == ["customizable-model"]
    assert provider_schema["extra"]["python"]["provider_source"] == (
        "provider/sub2api-plugin.py"
    )
    assert "provider_credential_schema" not in provider_schema
    assert "models" not in provider_schema

    model_schema = provider_schema["model_credential_schema"]
    credential_variables = [
        schema["variable"] for schema in model_schema["credential_form_schemas"]
    ]

    assert "openai_api_key" not in credential_variables
    assert "openai_api_base" not in credential_variables
    assert credential_variables == ["api_key", "endpoint_url", "context_size"]
    assert provider_schema["help"]["title"]["zh_Hans"] == "配置 Responses 接口"
    assert provider_schema["help"]["url"]["en_US"] == "/v1/responses"


def test_provider_schema_docs_copy_mentions_responses_only() -> None:
    provider_schema = load_provider_schema()
    readme_text = README_PATH.read_text(encoding="utf-8")
    guide_text = GUIDE_PATH.read_text(encoding="utf-8")
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")

    assert provider_schema["description"]["zh_Hans"] == (
        "面向 OpenAI 兼容 /v1/responses 接口的 Dify LLM 插件桥接。"
    )
    assert provider_schema["help"]["title"]["zh_Hans"] == "配置 Responses 接口"
    assert provider_schema["help"]["url"]["zh_Hans"] == "/v1/responses"
    assert "/v1/responses" in readme_text
    assert "/v1/responses" in guide_text
    assert "/v1/responses" in manifest_text
    assert "/v1/messages" not in readme_text
    assert "/v1/messages" not in guide_text
    assert "/v1/messages" not in manifest_text
    assert "/v1/chat/completions" not in readme_text
    assert "/v1/chat/completions" not in guide_text
    assert "/v1/chat/completions" not in manifest_text


def test_predefined_llm_schema_is_removed() -> None:
    assert not PREDEFINED_MODEL_SCHEMA_PATH.exists()


def test_customizable_runtime_schema_declares_chat_context_and_tool_features() -> None:
    """确认 runtime schema 会显式声明 Responses 所需的 chat/context/tool-call 能力。"""

    llm = Sub2apiPluginLargeLanguageModel([])
    schema = llm.get_customizable_model_schema(
        model="gpt-4.1-mini",
        credentials={"context_size": "32768"},
    )

    assert schema.features == [ModelFeature.MULTI_TOOL_CALL, ModelFeature.STREAM_TOOL_CALL]
    assert schema.model_properties == {
        ModelPropertyKey.CONTEXT_SIZE: 32768,
        ModelPropertyKey.MODE: "chat",
    }


def test_default_runtime_schema_exposes_structured_output_parameter_rules() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    schema = llm.get_customizable_model_schema(
        model="gpt-4.1-mini",
        credentials={"context_size": "32768"},
    )

    parameter_rules_by_name = {
        parameter_rule.name: parameter_rule for parameter_rule in schema.parameter_rules
    }

    assert list(parameter_rules_by_name.keys()) == [
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
        "response_format",
        "json_schema",
    ]
    assert parameter_rules_by_name["temperature"].use_template == "temperature"
    assert parameter_rules_by_name["top_p"].use_template == "top_p"
    assert parameter_rules_by_name["frequency_penalty"].use_template == "frequency_penalty"
    assert parameter_rules_by_name["presence_penalty"].use_template == "presence_penalty"
    assert parameter_rules_by_name["max_tokens"].use_template == "max_tokens"
    assert parameter_rules_by_name["response_format"].options == ["text", "json_object", "json_schema"]
    assert parameter_rules_by_name["json_schema"].use_template == "json_schema"


def test_gpt_5_4_runtime_schema_exposes_supported_parameter_rules() -> None:
    llm = Sub2apiPluginLargeLanguageModel([])
    schema = llm.get_customizable_model_schema(
        model="gpt-5.4",
        credentials={"context_size": "32768"},
    )

    parameter_rules_by_name = {
        parameter_rule.name: parameter_rule for parameter_rule in schema.parameter_rules
    }
    temperature_help = parameter_rules_by_name["temperature"].help
    top_p_help = parameter_rules_by_name["top_p"].help
    frequency_penalty_help = parameter_rules_by_name["frequency_penalty"].help
    presence_penalty_help = parameter_rules_by_name["presence_penalty"].help
    reasoning_effort_help = parameter_rules_by_name["reasoning_effort"].help

    assert list(parameter_rules_by_name.keys()) == [
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
        "response_format",
        "json_schema",
        "reasoning_effort",
        "verbosity",
    ]
    assert parameter_rules_by_name["temperature"].use_template == "temperature"
    assert parameter_rules_by_name["top_p"].use_template == "top_p"
    assert parameter_rules_by_name["frequency_penalty"].use_template == "frequency_penalty"
    assert parameter_rules_by_name["presence_penalty"].use_template == "presence_penalty"
    assert parameter_rules_by_name["max_tokens"].use_template == "max_tokens"
    assert parameter_rules_by_name["response_format"].options == ["text", "json_schema"]
    assert parameter_rules_by_name["json_schema"].use_template == "json_schema"
    assert temperature_help is not None
    assert temperature_help.zh_Hans == (
        "仅在推理努力程度为 none 时建议设置。若推理努力程度不是 none，OpenAI GPT-5.4 Responses API 不支持 temperature。"
    )
    assert top_p_help is not None
    assert top_p_help.zh_Hans == (
        "仅在推理努力程度为 none 时建议设置。若推理努力程度不是 none，OpenAI GPT-5.4 Responses API 不支持 top_p。"
    )
    assert frequency_penalty_help is not None
    assert frequency_penalty_help.zh_Hans == (
        "当前插件会按顶层字段透传该参数；OpenAI 官方文档尚未明确说明它会像 temperature/top_p 一样受推理努力程度限制。"
    )
    assert presence_penalty_help is not None
    assert presence_penalty_help.zh_Hans == (
        "当前插件会按顶层字段透传该参数；OpenAI 官方文档尚未明确说明它会像 temperature/top_p 一样受推理努力程度限制。"
    )
    assert parameter_rules_by_name["reasoning_effort"].default == "none"
    assert parameter_rules_by_name["reasoning_effort"].options == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    ]
    assert reasoning_effort_help is not None
    assert reasoning_effort_help.zh_Hans == (
        "约束 gpt-5.4 的推理努力程度。支持 none、low、medium、high、xhigh；none 为低延迟模式。若取值不是 none，temperature 与 top_p 不应设置；frequency_penalty 与 presence_penalty 当前仍按顶层透传。"
    )
    assert parameter_rules_by_name["verbosity"].default == "medium"
    assert parameter_rules_by_name["verbosity"].options == ["low", "medium", "high"]


def test_provider_schema_scope_audit_responses_only_runtime_surface() -> None:
    provider_schema = load_provider_schema()
    manifest_schema = cast(dict[str, object], yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")))
    runtime_surface_texts = [
        PROVIDER_SCHEMA_PATH.read_text(encoding="utf-8"),
        MANIFEST_PATH.read_text(encoding="utf-8"),
        LLM_IMPLEMENTATION_PATH.read_text(encoding="utf-8"),
    ]

    permission_schema = cast(
        dict[str, object],
        cast(dict[str, object], cast(dict[str, object], manifest_schema["resource"])["permission"])["model"],
    )

    assert provider_schema["supported_model_types"] == ["llm"]
    assert permission_schema == {
        "enabled": True,
        "llm": True,
        "text_embedding": False,
        "rerank": False,
        "tts": False,
        "speech2text": False,
        "moderation": False,
    }
    assert all("/v1/messages" not in text for text in runtime_surface_texts)
    assert all("/v1/chat/completions" not in text for text in runtime_surface_texts)


def test_readme_mentions_gpt_5_4_parameter_compatibility() -> None:
    readme_text = README_PATH.read_text(encoding="utf-8")

    assert "## gpt-5.4 参数兼容说明" in readme_text
    assert "当 `reasoning_effort` 为 `none` 时，可以同时使用 `temperature` 与 `top_p`。" in readme_text
    assert "当 `reasoning_effort` 不是 `none` 时，不应再设置 `temperature` 与 `top_p`。" in readme_text
    assert "`frequency_penalty` 与 `presence_penalty` 当前仍按顶层字段透传。" in readme_text


def test_guide_mentions_gpt_5_4_parameter_compatibility() -> None:
    guide_text = GUIDE_PATH.read_text(encoding="utf-8")

    assert "## gpt-5.4 参数兼容说明" in guide_text
    assert "当 `reasoning_effort` 为 `none` 时，可以同时使用 `temperature` 与 `top_p`。" in guide_text
    assert "当 `reasoning_effort` 不是 `none` 时，不应再设置 `temperature` 与 `top_p`。" in guide_text
