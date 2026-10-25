"""Claude client construction. The API key comes from Azure Key Vault when AZURE_KEY_VAULT_URL is set."""
from __future__ import annotations

from functools import lru_cache

import anthropic

from .config import LLMSettings


@lru_cache(maxsize=4)
def _key_vault_secret(vault_url: str, secret_name: str) -> str:
    # Imported here so the rest of the package works without the Azure SDK configured.
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    # DefaultAzureCredential tries managed identity, environment variables, `az login`, etc.
    client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())
    value = client.get_secret(secret_name).value
    if not value:
        raise RuntimeError(f"Secret {secret_name!r} in {vault_url} is empty")
    return value


def api_key(settings: LLMSettings) -> str | None:
    """The key from Key Vault, or None to let the SDK use its usual environment variables."""
    if settings.key_vault_url:
        return _key_vault_secret(settings.key_vault_url, settings.key_secret_name)
    return None


def async_client(settings: LLMSettings) -> anthropic.AsyncAnthropic:
    key = api_key(settings)
    if settings.provider == "foundry":
        # Without explicit values the SDK reads ANTHROPIC_FOUNDRY_API_KEY / ANTHROPIC_FOUNDRY_RESOURCE.
        return anthropic.AsyncAnthropicFoundry(api_key=key, resource=settings.foundry_resource or None)
    if settings.provider == "anthropic":
        return anthropic.AsyncAnthropic(api_key=key) if key else anthropic.AsyncAnthropic()
    raise ValueError(f"Unknown LLM_PROVIDER {settings.provider!r}; use 'anthropic' or 'foundry'")
