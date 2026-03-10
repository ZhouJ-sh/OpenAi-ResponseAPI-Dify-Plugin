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
