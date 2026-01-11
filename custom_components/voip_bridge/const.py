"""Constants for VoIP Bridge integration."""
from typing import Final

DOMAIN: Final = "voip_bridge"

# Configuration keys
CONF_SIP_SERVER: Final = "sip_server"
CONF_SIP_PORT: Final = "sip_port"
CONF_SIP_USERNAME: Final = "sip_username"
CONF_SIP_PASSWORD: Final = "sip_password"
CONF_SIP_EXTENSION: Final = "sip_extension"
CONF_ASSIST_PIPELINE: Final = "assist_pipeline_id"
CONF_REQUIRE_PIN_INTERNAL: Final = "require_pin_internal"
CONF_REQUIRE_PIN_EXTERNAL: Final = "require_pin_external"
CONF_PIN: Final = "pin"
CONF_TRUSTED_CALLER_IDS: Final = "trusted_caller_ids"
CONF_OUTBOUND_DESTINATIONS: Final = "outbound_destinations"
CONF_CODEC: Final = "codec"
CONF_SAMPLE_RATE: Final = "sample_rate"
CONF_VAD_AGGRESSIVENESS: Final = "vad_aggressiveness"
CONF_SILENCE_TIMEOUT: Final = "silence_timeout"

# Defaults
DEFAULT_SIP_PORT: Final = 5060
DEFAULT_CODEC: Final = "PCMU"  # G.711 μ-law
DEFAULT_SAMPLE_RATE: Final = 8000
DEFAULT_VAD_AGGRESSIVENESS: Final = 2
DEFAULT_SILENCE_TIMEOUT: Final = 1.5
DEFAULT_STARTUP_CALL_DELAY: Final = 30

# Event priorities
PRIORITY_CRITICAL: Final = 1
PRIORITY_WARNING: Final = 2
PRIORITY_INFO: Final = 3

# Call states
STATE_IDLE: Final = "idle"
STATE_RINGING: Final = "ringing"
STATE_AUTH_REQUIRED: Final = "auth_required"
STATE_AUTHENTICATED: Final = "authenticated"
STATE_PRESENTING_EVENTS: Final = "presenting_events"
STATE_IN_EVENT: Final = "in_event"
STATE_GENERAL_QUERY: Final = "general_query"
STATE_CLOSING: Final = "closing"
STATE_ENDED: Final = "ended"

# Services
SERVICE_ADD_EVENT: Final = "add_event"
SERVICE_CLEAR_EVENT: Final = "clear_event"
SERVICE_CLEAR_ALL_EVENTS: Final = "clear_all_events"
SERVICE_TRIGGER_CALL: Final = "trigger_call"
SERVICE_LIST_EVENTS: Final = "list_events"

# Storage
STORAGE_KEY: Final = "voip_bridge_events"
STORAGE_VERSION: Final = 1