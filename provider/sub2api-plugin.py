# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportImplicitOverride=false

from collections.abc import Mapping

from dify_plugin import ModelProvider


class Sub2apiPluginModelProvider(ModelProvider):
    """Sub2api 的最小 provider 桥接实现。"""

    def validate_provider_credentials(self, credentials: Mapping[str, object]) -> None:
        """保持 provider 层无独立凭据校验，实际配置完全由 customizable model 表单承载。"""

        # 当前插件只暴露 customizable-model 合同，provider 自身没有额外凭据表单。
        del credentials
        return None
