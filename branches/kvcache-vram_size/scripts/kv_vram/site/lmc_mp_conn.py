"""D9: LMCache 0.4.4's own MP connector under a name vLLM's registry does not claim.
vLLM 0.15.1's built-in LMCacheMPConnector keys lookups by block hashes, which the 0.4.4
adapter/server reject (they expect token ids); LMCache's copy keys by token ids."""
from lmcache.integration.vllm.lmcache_mp_connector_0180 import LMCacheMPConnector as LMCacheMP044Connector  # noqa: F401
