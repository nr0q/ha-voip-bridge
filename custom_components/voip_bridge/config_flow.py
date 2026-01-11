"""Config flow for VoIP Bridge integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components.assist_pipeline import DOMAIN as ASSIST_DOMAIN
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_SIP_SERVER,
    CONF_SIP_PORT,
    CONF_SIP_USERNAME,
    CONF_SIP_PASSWORD,
    CONF_SIP_EXTENSION,
    CONF_ASSIST_PIPELINE,
    CONF_REQUIRE_PIN_INTERNAL,
    CONF_REQUIRE_PIN_EXTERNAL,
    CONF_PIN,
    CONF_TRUSTED_CALLER_IDS,
    CONF_OUTBOUND_DESTINATIONS,
    CONF_CODEC,
    CONF_SAMPLE_RATE,
    CONF_VAD_AGGRESSIVENESS,
    CONF_SILENCE_TIMEOUT,
    DEFAULT_SIP_PORT,
    DEFAULT_CODEC,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VAD_AGGRESSIVENESS,
    DEFAULT_SILENCE_TIMEOUT,
)

_LOGGER = logging.getLogger(__name__)


def get_assist_pipelines(hass: HomeAssistant) -> dict[str, str]:
    """Get available Assist pipelines."""
    if ASSIST_DOMAIN not in hass.data:
        return {}
    
    pipelines = {}
    # TODO: Get actual pipelines from assist_pipeline integration
    # For now, return empty dict - user will need to configure manually
    return pipelines


class VoipBridgeConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for VoIP Bridge."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._data: dict[str, Any] = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - FreePBX connection."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate connection (basic checks)
            if not user_input[CONF_SIP_SERVER]:
                errors["base"] = "invalid_host"
            else:
                # Store and move to next step
                self._data.update(user_input)
                return await self.async_step_security()

        data_schema = vol.Schema({
            vol.Required(CONF_SIP_SERVER): str,
            vol.Required(CONF_SIP_PORT, default=DEFAULT_SIP_PORT): int,
            vol.Required(CONF_SIP_USERNAME): str,
            vol.Required(CONF_SIP_PASSWORD): str,
            vol.Required(CONF_SIP_EXTENSION): str,
        })

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={
                "name": "FreePBX Connection",
            },
        )

    async def async_step_security(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle security settings."""
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_outbound()

        data_schema = vol.Schema({
            vol.Required(CONF_REQUIRE_PIN_INTERNAL, default=False): bool,
            vol.Required(CONF_REQUIRE_PIN_EXTERNAL, default=True): bool,
            vol.Optional(CONF_PIN, default="1234"): str,
            vol.Optional(CONF_TRUSTED_CALLER_IDS, default=""): str,
        })

        return self.async_show_form(
            step_id="security",
            data_schema=data_schema,
            description_placeholders={
                "name": "Security Settings",
            },
        )

    async def async_step_outbound(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle outbound destination settings."""
        if user_input is not None:
            # Parse destinations from text input
            destinations = []
            
            destinations_input = user_input[CONF_OUTBOUND_DESTINATIONS]
            if isinstance(destinations_input, str):
                lines = destinations_input.split("\n")
            else:
                lines = destinations_input  # Already a list
            
            for idx, line in enumerate(lines):
                line = line.strip()
                if line:
                    destinations.append({
                        "number": line,
                        "name": f"Destination {idx + 1}",
                        "priority": idx + 1,
                        "enabled": True,
                    })
            
            self._data[CONF_OUTBOUND_DESTINATIONS] = destinations
            return await self.async_step_audio()

        data_schema = vol.Schema({
            vol.Required(
                CONF_OUTBOUND_DESTINATIONS,
                default="5551234567"
            ): selector.TextSelector(
                selector.TextSelectorConfig(
                    multiple=True,
                    multiline=True,
                )
            ),
        })

        return self.async_show_form(
            step_id="outbound",
            data_schema=data_schema,
            description_placeholders={
                "name": "Outbound Destinations",
                "description": "Enter phone numbers, one per line. First number has highest priority.",
            },
        )

    async def async_step_audio(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle audio settings."""
        try:
            if user_input is not None:
                self._data.update(user_input)
                
                # Parse trusted caller IDs into list
                if CONF_TRUSTED_CALLER_IDS in self._data:
                    caller_ids = self._data[CONF_TRUSTED_CALLER_IDS]
                    if isinstance(caller_ids, str):
                        self._data[CONF_TRUSTED_CALLER_IDS] = [
                            cid.strip() for cid in caller_ids.split(",") if cid.strip()
                        ]
                
                # Create entry
                return self.async_create_entry(
                    title=f"VoIP Bridge ({self._data[CONF_SIP_EXTENSION]})",
                    data=self._data,
                )

            data_schema = vol.Schema({
                vol.Optional(CONF_ASSIST_PIPELINE, default=""): str,
                vol.Required(CONF_CODEC, default=DEFAULT_CODEC): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=["PCMU", "PCMA", "opus"],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_SAMPLE_RATE, default="8000"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=["8000", "16000"],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_VAD_AGGRESSIVENESS, default=2): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=3,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Required(CONF_SILENCE_TIMEOUT, default=1.5): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.5,
                        max=5.0,
                        step=0.5,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
            })

            return self.async_show_form(
                step_id="audio",
                data_schema=data_schema,
                description_placeholders={
                    "name": "Audio Settings",
                },
            )
        except Exception as e:
            _LOGGER.error(f"Error in async_step_audio: {e}", exc_info=True)
            raise

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return VoipBridgeOptionsFlowHandler()

class VoipBridgeOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for VoIP Bridge."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Show current settings for modification
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Optional(
                    CONF_REQUIRE_PIN_INTERNAL,
                    default=self.config_entry.data.get(CONF_REQUIRE_PIN_INTERNAL, False),
                ): bool,
                vol.Optional(
                    CONF_REQUIRE_PIN_EXTERNAL,
                    default=self.config_entry.data.get(CONF_REQUIRE_PIN_EXTERNAL, True),
                ): bool,
                vol.Optional(
                    CONF_PIN,
                    default=self.config_entry.data.get(CONF_PIN, "1234"),
                ): str,
            }),
        )