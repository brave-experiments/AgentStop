#
# This file is part of Glances.
#
# SPDX-FileCopyrightText: 2022 Nicolas Hennion <nicolas@nicolargo.com>
#
# SPDX-License-Identifier: LGPL-3.0-only
#

"""Battery plugin."""

import platform
import plistlib
import psutil
import subprocess

from glances.globals import LINUX
from glances.logger import logger
from glances.plugins.plugin.model import GlancesPluginModel

# Batinfo library (optional; Linux-only)
if LINUX:
    batinfo_tag = True
    try:
        import batinfo
    except ImportError:
        logger.debug("batinfo library not found. Fallback to psutil.")
        batinfo_tag = False
else:
    batinfo_tag = False

# PsUtil Sensors_battery available on Linux, Windows, FreeBSD, macOS
psutil_tag = True
try:
    psutil.sensors_battery()
except Exception as e:
    logger.error(f"Cannot grab battery status {e}.")
    psutil_tag = False


class PluginModel(GlancesPluginModel):
    """Glances battery capacity plugin.

    stats is a list
    """

    def __init__(self, args=None, config=None):
        """Init the plugin."""
        super().__init__(args=args, config=config, stats_init_value=[])

        # Init the sensor class
        try:
            self.glances_grab_bat = GlancesGrabBat()
        except Exception as e:
            logger.error(f"Can not init battery class ({e})")
            global batinfo_tag
            global psutil_tag
            batinfo_tag = False
            psutil_tag = False

        # We do not want to display the stat in a dedicated area
        # The HDD temp is displayed within the sensors plugin
        self.display_curse = False

    # @GlancesPluginModel._check_decorator
    @GlancesPluginModel._log_result_decorator
    def update(self):
        """Update battery capacity stats using the input method."""
        # Init new stats
        stats = self.get_init_value()

        if self.input_method == 'local':
            # Update stats
            self.glances_grab_bat.update()
            stats = self.glances_grab_bat.get()

        elif self.input_method == 'snmp':
            # Update stats using SNMP
            # Not available
            pass

        # Update the stats
        self.stats = stats

        return self.stats


class GlancesGrabBat:
    """Get batteries stats using the batinfo library."""

    def __init__(self):
        """Init batteries stats."""
        self.bat_list = []

        if batinfo_tag:
            self.bat = batinfo.batteries()
        elif psutil_tag:
            self.bat = psutil
        else:
            self.bat = None

    def update_bat_list(self, data, field, label, unit):
        value = data.get(field, None)
        if value is not None:
            self.bat_list.append({
                "label": label,
                "value": value,
                "unit": unit,
            })

    def update(self):
        """Update the stats."""
        self.bat_list = []
        if batinfo_tag:
            # Use the batinfo lib to grab the stats
            # Compatible with multiple batteries
            self.bat.update()
            # Batinfo support multiple batteries
            # ... so take it into account (see #1920)
            # self.bat_list = [{
            #     'label': 'Battery',
            #     'value': self.battery_percent,
            #     'unit': '%'}]
            for b in self.bat.stat:
                self.bat_list.append(
                    {
                        'label': 'BAT {}'.format(b.path.split('/')[-1]),
                        'value': b.capacity,
                        'unit': '%',
                        'status': b.status,
                    }
                )
        elif psutil_tag and hasattr(self.bat.sensors_battery(), 'percent'):
            # Use psutil to grab the stats
            # Give directly the battery percent
            self.bat_list = [
                {
                    'label': 'Battery',
                    'value': int(self.bat.sensors_battery().percent),
                    'unit': '%',
                    'status': 'Charging' if self.bat.sensors_battery().power_plugged else 'Discharging',
                }
            ]

        if platform.system() == "Darwin":
            try:
                ioreg = subprocess.check_output(["ioreg", "-r", "-n", "AppleSmartBattery", "-a"])
                bat_info = plistlib.loads(ioreg)[0]
                self.update_bat_list(
                    data=bat_info,
                    field="InstantAmperage",
                    label="Battery Current",
                    unit="mA",
                )
                self.update_bat_list(
                    data=bat_info,
                    field="AppleRawCurrentCapacity",
                    label="Battery Raw Capacity",
                    unit="mAh",
                )
                self.update_bat_list(
                    data=bat_info,
                    field="Voltage",
                    label="Battery Voltage",
                    unit="mW",
                )
                self.update_bat_list(
                    data=bat_info,
                    field="Temperature",
                    label="Battery Temperature",
                    unit="dK",
                )
                self.update_bat_list(
                    data=bat_info,
                    field="VirtualTemperature",
                    label="Battery Virtual Temperature",
                    unit="dK",
                )
                power_data = bat_info.get("PowerTelemetryData", None)
                if power_data is not None:
                    self.update_bat_list(
                        data=power_data,
                        field="BatteryPower",
                        label="Battery Power",
                        unit="mW",
                    )
            except:
                pass 

    def get(self):
        """Get the stats."""
        return self.bat_list

    @property
    def battery_percent(self):
        """Get batteries capacity percent."""
        if not batinfo_tag or not self.bat.stat:
            return []

        # Init the b_sum (sum of percent)
        # and Loop over batteries (yes a computer could have more than 1 battery)
        b_sum = 0
        for b in self.bat.stat:
            try:
                b_sum += int(b.capacity)
            except ValueError:
                return []

        # Return the global percent
        return int(b_sum / len(self.bat.stat))
