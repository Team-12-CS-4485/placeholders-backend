"""
scheduler.py - YouTube Ingestion Scheduler

Runs the YouTube ingestion pipeline in three nightly batches (4 channels each):
  Batch 1 — Sunday    03:00 UTC  (partial — week counter unchanged)
  Batch 2 — Monday    03:00 UTC  (partial — week counter unchanged)
  Batch 3 — Tuesday   03:00 UTC  (full   — advances the week counter)

Usage:
    python scheduler.py                        # start recurring schedule
    python scheduler.py --run-now              # fire all channels immediately (full run)
    python scheduler.py --run-now --partial    # fire without advancing week counter
    python scheduler.py --run-now --batch=1   # fire only batch 1 immediately
"""

import logging
import os
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Ensure data_collection imports resolve
sys.path.insert(0, os.path.dirname(__file__))

from youtube_ingestion import main as run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ingestion_job(batch: int = 0, partial: bool = False):
    label = f"batch {batch}" if batch else "all channels"
    logger.info(f"Scheduler: starting ingestion run ({label})")
    try:
        run_ingestion(partial=partial, batch=batch)
        logger.info(f"Scheduler: ingestion run completed ({label})")
    except Exception as e:
        logger.error(f"Scheduler: ingestion run failed ({label}): {e}", exc_info=True)


def _parse_batch_arg() -> int:
    """Return the value of --batch=N from sys.argv, or 0 if not present."""
    for arg in sys.argv:
        if arg.startswith("--batch="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                pass
    return 0


def main():
    if "--run-now" in sys.argv:
        partial = "--partial" in sys.argv
        batch = _parse_batch_arg()
        logger.info("--run-now flag detected: firing ingestion job immediately")
        ingestion_job(batch=batch, partial=partial)
        return

    scheduler = BlockingScheduler(timezone="UTC")

    # Batch 1 — Sunday 03:00 UTC — partial (does not advance week counter)
    scheduler.add_job(
        ingestion_job,
        CronTrigger(day_of_week="sun", hour=3, minute=0, timezone="UTC"),
        kwargs={"batch": 1, "partial": True},
        id="youtube_batch_1",
        replace_existing=True,
    )

    # Batch 2 — Monday 03:00 UTC — partial
    scheduler.add_job(
        ingestion_job,
        CronTrigger(day_of_week="mon", hour=3, minute=0, timezone="UTC"),
        kwargs={"batch": 2, "partial": True},
        id="youtube_batch_2",
        replace_existing=True,
    )

    # Batch 3 — Tuesday 03:00 UTC — full run (advances week counter)
    scheduler.add_job(
        ingestion_job,
        CronTrigger(day_of_week="tue", hour=3, minute=0, timezone="UTC"),
        kwargs={"batch": 3, "partial": False},
        id="youtube_batch_3",
        replace_existing=True,
    )

    logger.info("Scheduler started:")
    logger.info("  Batch 1 — Sunday    03:00 UTC (partial)")
    logger.info("  Batch 2 — Monday    03:00 UTC (partial)")
    logger.info("  Batch 3 — Tuesday   03:00 UTC (full — advances week counter)")
    logger.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
