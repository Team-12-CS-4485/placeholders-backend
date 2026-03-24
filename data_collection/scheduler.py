"""
scheduler.py - YouTube Ingestion Scheduler

Runs the consolidated YouTube ingestion pipeline once a week:
  Weekly — Sunday 03:00 UTC (full run across all channels)

Usage:
    python scheduler.py                        # start recurring schedule
    python scheduler.py --run-now              # fire all channels immediately
    python scheduler.py --run-now --partial    # fire without advancing behaviour
    python scheduler.py --run-now --dry-run    # full run, no writes (testing)
"""

import logging
import os
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Ensure data_collection imports resolve
sys.path.insert(0, os.path.dirname(__file__))

import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ingestion_job(partial: bool = False, dry_run: bool = False):
    logger.info("Scheduler: starting ingestion run")
    try:
        pipeline.run_pipeline(partial=partial, dry_run=dry_run)
        logger.info("Scheduler: ingestion run completed")
    except Exception as e:
        logger.error(f"Scheduler: ingestion run failed: {e}", exc_info=True)


def main():
    if "--run-now" in sys.argv:
        partial = "--partial" in sys.argv
        dry_run = "--dry-run" in sys.argv
        logger.info("--run-now flag detected: firing ingestion job immediately")
        ingestion_job(partial=partial, dry_run=dry_run)
        return

    scheduler = BlockingScheduler(timezone="UTC")

    # Single weekly trigger — Sunday 03:00 UTC
    scheduler.add_job(
        ingestion_job,
        CronTrigger(day_of_week="sun", hour=3, minute=0, timezone="UTC"),
        kwargs={"partial": False},
        id="youtube_weekly",
        replace_existing=True,
    )

    logger.info("Scheduler started:")
    logger.info("  Weekly run — Sunday 03:00 UTC")
    logger.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
