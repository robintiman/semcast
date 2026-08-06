"""Dagster entrypoint. No schedule attached on purpose — the job is
manually triggered, so materializing a partition from the UI or the CLI is
the whole trigger story. Adding a `ScheduleDefinition` here later is the
only change needed to make it automatic.
"""

from dagster import Definitions, load_assets_from_modules

from . import assets

defs = Definitions(assets=load_assets_from_modules([assets]))
