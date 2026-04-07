"""
check_data_consistency.py

Verifies referential integrity between youtube-videos and narrative-clusters:
  1. Every cluster_id on a video exists in narrative-clusters
  2. Every active cluster has at least one video pointing to it
  3. cluster video_count matches actual video count
  4. Videos with cluster_id but no cluster_label (partial write)
  5. Videos with a cluster_label but no cluster_id (orphaned label)
"""

import boto3
from collections import Counter, defaultdict
from decimal import Decimal
from app.core.config import settings

dynamo = boto3.resource("dynamodb", region_name=settings.aws_region)
videos_table = dynamo.Table(settings.dynamodb_table)
clusters_table = dynamo.Table("narrative-clusters")


def scan_all(table, projection=None, extra_kwargs=None):
    kwargs = {}
    if projection:
        kwargs["ProjectionExpression"] = projection
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    items = []
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def to_int(val):
    if val is None:
        return None
    if isinstance(val, Decimal):
        return int(val)
    return int(val)


print("Scanning videos...")
raw_videos = scan_all(
    videos_table,
    projection="SortKey, cluster_id, cluster_label, #wk",
    extra_kwargs={"ExpressionAttributeNames": {"#wk": "week"}},
)
print(f"  {len(raw_videos)} videos found")

print("Scanning clusters...")
raw_clusters = scan_all(clusters_table)
print(f"  {len(raw_clusters)} clusters found\n")

# Build cluster lookup {cluster_id (int): item}
cluster_map = {}
for c in raw_clusters:
    cid = to_int(c.get("cluster_id"))
    if cid is not None:
        cluster_map[cid] = c

# Build per-cluster video counts and per-week counts from videos
videos_per_cluster = Counter()
weeks_per_cluster = defaultdict(set)
orphaned_label = []  # cluster_label but no cluster_id
partial_write = []  # cluster_id but no cluster_label
dangling_cluster_ids = []  # cluster_id not in clusters table
unclustered = []  # no cluster_id

for v in raw_videos:
    vid = v.get("SortKey", "?")
    cid = to_int(v.get("cluster_id"))
    label = v.get("cluster_label")
    week = v.get("week", "unknown")

    if cid is None and label:
        orphaned_label.append(vid)
    elif cid is not None and not label:
        partial_write.append((vid, cid))
    elif cid is None:
        unclustered.append(vid)

    if cid is not None:
        videos_per_cluster[cid] += 1
        weeks_per_cluster[cid].add(week)
        if cid not in cluster_map:
            dangling_cluster_ids.append((vid, cid))

# Check stored video_count vs actual
count_mismatches = []
for cid, cluster in cluster_map.items():
    stored = to_int(cluster.get("video_count"))
    actual = videos_per_cluster.get(cid, 0)
    if stored != actual:
        count_mismatches.append(
            {
                "cluster_id": cid,
                "label": cluster.get("cluster_label", "?"),
                "stored_count": stored,
                "actual_count": actual,
            }
        )

# Clusters with no videos at all
no_video_clusters = [
    {
        "cluster_id": cid,
        "label": c.get("cluster_label", "?"),
        "status": c.get("status", "?"),
    }
    for cid, c in cluster_map.items()
    if videos_per_cluster.get(cid, 0) == 0
]

# ── Report ────────────────────────────────────────────────────────────────────

print("=" * 60)
print("DATA CONSISTENCY REPORT")
print("=" * 60)

print(f"\n[1] Videos with no cluster_id:          {len(unclustered)}")
print(f"[2] Videos with cluster_id, no label:   {len(partial_write)}")
print(f"[3] Videos with label, no cluster_id:   {len(orphaned_label)}")
print(f"[4] Videos pointing to missing cluster: {len(dangling_cluster_ids)}")
print(f"[5] video_count mismatches:             {len(count_mismatches)}")
print(f"[6] Clusters with 0 videos:             {len(no_video_clusters)}")

if partial_write:
    print(f"\n--- Partial writes (cluster_id, no label) ---")
    for vid, cid in partial_write[:20]:
        print(f"  video={vid}  cluster_id={cid}")

if dangling_cluster_ids:
    print(f"\n--- Videos pointing to non-existent clusters ---")
    for vid, cid in dangling_cluster_ids[:20]:
        print(f"  video={vid}  cluster_id={cid}")

if count_mismatches:
    print(f"\n--- video_count mismatches ---")
    for m in count_mismatches:
        print(
            f"  cluster={m['cluster_id']} ({m['label']}): stored={m['stored_count']} actual={m['actual_count']}"
        )

if no_video_clusters:
    print(f"\n--- Clusters with 0 videos ---")
    for c in no_video_clusters:
        print(f"  cluster={c['cluster_id']} ({c['label']}) status={c['status']}")

if orphaned_label:
    print(f"\n--- Orphaned labels (label, no cluster_id) ---")
    for vid in orphaned_label[:20]:
        print(f"  video={vid}")

print(f"\n--- Active cluster breakdown ---")
active = sorted(
    [(cid, c) for cid, c in cluster_map.items() if c.get("status") == "active"],
    key=lambda x: x[0],
)
for cid, c in active:
    actual = videos_per_cluster.get(cid, 0)
    stored = to_int(c.get("video_count"))
    weeks = sorted(weeks_per_cluster.get(cid, set()))
    match = "OK" if stored == actual else f"MISMATCH(stored={stored})"
    print(
        f"  {cid:3}: {c.get('cluster_label', '?'):<45} videos={actual} {match}  weeks={weeks}"
    )

declining = [
    (cid, c) for cid, c in cluster_map.items() if c.get("status") == "declining"
]
if declining:
    print(f"\n--- Declining clusters ---")
    for cid, c in sorted(declining):
        print(
            f"  {cid}: {c.get('cluster_label', '?')} (videos={videos_per_cluster.get(cid, 0)})"
        )
