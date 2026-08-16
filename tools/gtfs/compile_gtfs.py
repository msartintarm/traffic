#!/usr/bin/env python3
"""Compile real GTFS feeds into the transit artifact the engine's
`sim::rail::transit_from_json` consumes.

The output schema is the contract between this tool and the Rust engine:

    {
      "meta":       {"attribution": [str], "license_urls": [str], "feeds": [str]},
      "rail_trips": [{"class", "weekend", "carriages",
                      "stops": [{"x", "y", "arr", "dep", "timepoint"}]}],
      "bus_trips":  [{"line", "weekend", "stops": [...]}]
    }

`x`/`y` are metres in the map artifact's local equirectangular projection
(read from its `meta.origin`); `arr`/`dep` are day-seconds, kept raw past
midnight (a 25:10 departure stays 90600). `class` is one of
emu|metro|light_rail|diesel; `weekend` false is a representative weekday
(a concrete Wednesday inside the feed's validity window), true a Saturday.

Trips are clipped to the map bbox: the longest contiguous run of in-box stops
is kept and, where the trip continues beyond the box, a pseudo-stop
(arr == dep, timepoint false) is interpolated where the path between the
boundary pair of stops crosses the bbox edge — so a train that merely passes
through (or serves one in-box station) still runs across the whole box.

Feeds come from `--feed` (GTFS zip URLs or local paths) or, when none are
given, are discovered from the Mobility Database catalog: every active GTFS
feed whose bounding box intersects the map's, most local first. With
TRANSIT_511_API_KEY set, the 511.org Bay Area regional feed is added.

Usage:
    python3 compile_gtfs.py --map ../../web/public/map.json.gz --out map.transit.json.gz
"""

import argparse
import csv
import datetime
import gzip
import io
import json
import math
import os
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict

CATALOG_URL = "https://files.mobilitydatabase.org/feeds_v2.csv"
URL_511 = "http://api.511.org/transit/datafeeds?operator_id=RG"

CARRIAGES = {"emu": 7, "metro": 10, "light_rail": 2, "diesel": 6}
# Heavy rail defaults to EMU; these operators haul diesel consists.
DIESEL_AGENCIES = ("amtrak", "altamont", "capitol corridor", "san joaquin")
DOW_COLS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def project(lat, lon, lat0, lon0):
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * 6371000.0
    y = math.radians(lat - lat0) * 6371000.0
    return round(x, 2), round(y, 2)


def load_map(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        meta = json.load(f)["meta"]
    return meta["bbox"], meta["origin"]


def http_get(url, attempts=3, timeout=120):
    headers = {"User-Agent": "traffic-sim-gtfs-compiler/0.1 (github traffic sim)"}
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except OSError as err:
            last = err
            time.sleep(2**attempt)
    print(f"warning: download failed after {attempts} attempts ({last}): {url}")
    return None


def discover_feeds(bbox, max_feeds):
    """Active GTFS feeds from the Mobility Database catalog whose bounding box
    intersects the map's, ranked by how much of the *map* box they cover
    (descending) — a corridor-wide operator that spans the whole box beats a
    campus shuttle that nicks a corner — with smaller feed boxes as the
    tiebreak so a regional feed still outranks a national one."""
    blob = http_get(CATALOG_URL)
    if blob is None:
        raise SystemExit("mobility database catalog unreachable and no --feed given")
    s, w, n, e = bbox
    feeds = []
    for row in csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))):
        if row.get("data_type") != "gtfs" or row.get("status") in ("deprecated", "inactive"):
            continue
        try:
            fs = float(row["location.bounding_box.minimum_latitude"])
            fn = float(row["location.bounding_box.maximum_latitude"])
            fw = float(row["location.bounding_box.minimum_longitude"])
            fe = float(row["location.bounding_box.maximum_longitude"])
        except (KeyError, ValueError):
            continue
        if fs > n or fn < s or fw > e or fe < w:
            continue
        hosted = row.get("urls.latest", "")
        url = hosted or row.get("urls.direct_download", "")
        if not url or (not hosted and row.get("urls.authentication_type") not in ("", "0")):
            continue
        inter = (min(fn, n) - max(fs, s)) * (min(fe, e) - max(fw, w))
        cover = inter / max((n - s) * (e - w), 1e-12)
        feeds.append((-cover, (fn - fs) * (fe - fw), row.get("provider", ""), url, row.get("urls.license", "")))
    feeds.sort(key=lambda f: (f[0], f[1]))
    return [(name, url, lic) for _, _, name, url, lic in feeds[:max_feeds]]


# --- GTFS tables ------------------------------------------------------------


def open_table(z, name):
    member = next((m for m in z.namelist() if m.split("/")[-1] == name), None)
    return csv.DictReader(io.TextIOWrapper(z.open(member), encoding="utf-8-sig")) if member else iter(())


def parse_gtfs_date(s):
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def parse_time(s):
    parts = s.split(":") if s else []
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None


def first_weekday(dow, start, end):
    d = start + datetime.timedelta((dow - start.weekday()) % 7)
    return d if d <= end else None


def pick_dates(cal, cal_dates):
    """A concrete Wednesday and Saturday inside the feed's validity window
    (from today on when possible); calendar-less feeds use their busiest
    exception-added dates instead."""
    if cal:
        start = min(parse_gtfs_date(r["start_date"]) for r in cal)
        end = max(parse_gtfs_date(r["end_date"]) for r in cal)
        base = max(start, min(datetime.date.today(), end))
        return (
            first_weekday(2, base, end) or first_weekday(2, start, end),
            first_weekday(5, base, end) or first_weekday(5, start, end),
        )
    added = Counter(r["date"] for r in cal_dates if r.get("exception_type") == "1")

    def busiest(dow):
        days = [d for d in added if parse_gtfs_date(d).weekday() == dow]
        return parse_gtfs_date(max(days, key=lambda d: added[d])) if days else None

    return busiest(2), busiest(5)


def active_services(day, cal, cal_dates):
    svc = set()
    col = DOW_COLS[day.weekday()]
    for r in cal:
        try:
            if r.get(col) == "1" and parse_gtfs_date(r["start_date"]) <= day <= parse_gtfs_date(r["end_date"]):
                svc.add(r["service_id"])
        except (KeyError, ValueError):
            continue
    key = day.strftime("%Y%m%d")
    for r in cal_dates:
        if r.get("date") == key:
            (svc.add if r.get("exception_type") == "1" else svc.discard)(r["service_id"])
    return svc


def route_class(route, agency):
    try:
        rt = int(route.get("route_type") or -1)
    except ValueError:
        return None
    if rt in (3, 11) or 200 <= rt < 300:
        return "bus"
    if rt in (0, 5) or 900 <= rt < 1000:
        return "light_rail"
    if rt in (1, 12) or 400 <= rt < 500:
        return "metro"
    if rt == 2 or 100 <= rt < 200:
        name = (agency.get(route.get("agency_id") or "") or next(iter(agency.values()), "")).lower()
        diesel = any(k in name for k in DIESEL_AGENCIES) or "ace" in name.split()
        return "diesel" if diesel else "emu"
    return None


# --- geometry ---------------------------------------------------------------


def bbox_rect(bbox, origin):
    s, w, n, e = bbox
    x0, y0 = project(s, w, *origin)
    x1, y1 = project(n, e, *origin)
    return x0, y0, x1, y1


def in_rect(p, rect):
    return rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]


def clip_seg(p, q, rect):
    """Liang–Barsky: the [t0, t1] parameter slice of segment p→q inside rect."""
    t0, t1 = 0.0, 1.0
    for s, d, lo, hi in ((p[0], q[0] - p[0], rect[0], rect[2]), (p[1], q[1] - p[1], rect[1], rect[3])):
        if d == 0.0:
            if not lo <= s <= hi:
                return None
            continue
        ta, tb = (lo - s) / d, (hi - s) / d
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return None
    return t0, t1


def crossings(path, rect):
    """Bbox-boundary events along a polyline: (fraction of path length, point,
    entering) per crossing, in path order."""
    lens = [math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(path, path[1:])]
    total = sum(lens)
    if total == 0.0:
        return []
    evs, acc, inside = [], 0.0, in_rect(path[0], rect)
    for (p, q), ln in zip(zip(path, path[1:]), lens):
        if ln == 0.0:
            continue
        cut = clip_seg(p, q, rect)
        if cut is None:
            inside = False
        else:
            t0, t1 = cut
            if not inside and t0 < 1.0:
                evs.append(((acc + t0 * ln) / total, (p[0] + t0 * (q[0] - p[0]), p[1] + t0 * (q[1] - p[1])), True))
            if t1 < 1.0:
                evs.append(((acc + t1 * ln) / total, (p[0] + t1 * (q[0] - p[0]), p[1] + t1 * (q[1] - p[1])), False))
            inside = t1 >= 1.0
        acc += ln
    return evs


def chainage(pts, cum, p):
    """Distance along the polyline of p's nearest point on it."""
    best, best_s = math.inf, 0.0
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        dx, dy = pts[i + 1][0] - ax, pts[i + 1][1] - ay
        dd = dx * dx + dy * dy
        t = 0.0 if dd == 0.0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / dd))
        d = (p[0] - ax - t * dx) ** 2 + (p[1] - ay - t * dy) ** 2
        if d < best:
            best, best_s = d, cum[i] + t * math.sqrt(dd)
    return best_s


def sub_path(shape, sa, sb, a, b):
    """Polyline from stop a to stop b: the shape slice between their chainages
    when it runs forward, else the straight line."""
    if shape is None or sb <= sa:
        return [a, b]
    pts, cum, _ = shape
    return [a] + [pts[i] for i in range(len(pts)) if sa < cum[i] < sb] + [b]


# --- per-feed compile -------------------------------------------------------


def load_shapes(z, wanted, rect, origin):
    lat0, lon0 = origin
    raw = defaultdict(list)
    for r in open_table(z, "shapes.txt"):
        sid = r.get("shape_id")
        if sid not in wanted:
            continue
        try:
            raw[sid].append((int(r["shape_pt_sequence"]), *project(float(r["shape_pt_lat"]), float(r["shape_pt_lon"]), lat0, lon0)))
        except (KeyError, ValueError):
            continue
    shapes = {}
    for sid, seq in raw.items():
        pts = [(x, y) for _, x, y in sorted(seq)]
        if len(pts) < 2:
            continue
        cum, acc = [0.0], 0.0
        for p, q in zip(pts, pts[1:]):
            acc += math.hypot(q[0] - p[0], q[1] - p[1])
            cum.append(acc)
        hits = any(in_rect(p, rect) for p in pts) or any(clip_seg(p, q, rect) for p, q in zip(pts, pts[1:]))
        shapes[sid] = (pts, cum, hits)
    return shapes


def load_stop_times(z, wanted):
    rows = defaultdict(list)
    for r in open_table(z, "stop_times.txt"):
        tid = r.get("trip_id")
        if tid not in wanted:
            continue
        try:
            seq = int(r["stop_sequence"])
        except (KeyError, ValueError):
            continue
        sd = (r.get("shape_dist_traveled") or "").strip()
        rows[tid].append((seq, r.get("stop_id"), parse_time(r.get("arrival_time")), parse_time(r.get("departure_time")), float(sd) if sd else None, r.get("timepoint") or ""))
    for tid in rows:
        rows[tid].sort()
    return rows


def timed_rows(raw, stops):
    """stop_times rows resolved to projected stops with complete times: blanks
    interpolated between the surrounding timed rows, weighted by
    shape_dist_traveled when present, else by row index."""
    rows = []
    for _, stop_id, arr, dep, sd, tp in raw:
        pos = stops.get(stop_id)
        if pos is None:
            continue
        if arr is None:
            arr = dep
        if dep is None:
            dep = arr
        if dep is not None:
            dep = max(dep, arr)
        rows.append({"pos": pos, "arr": arr, "dep": dep, "sd": sd, "tp": tp, "interp": False})
    timed = [i for i, r in enumerate(rows) if r["arr"] is not None]
    if not timed:
        return []
    rows = rows[timed[0] : timed[-1] + 1]  # untimed ends are unrecoverable
    timed = [i for i, r in enumerate(rows) if r["arr"] is not None]
    for i, j in zip(timed, timed[1:]):
        if j == i + 1:
            continue
        sds = [rows[k]["sd"] for k in range(i, j + 1)]
        by_sd = all(s is not None for s in sds) and sds[-1] > sds[0]
        t0, t1 = rows[i]["dep"], rows[j]["arr"]
        for k in range(i + 1, j):
            w = (rows[k]["sd"] - sds[0]) / (sds[-1] - sds[0]) if by_sd else (k - i) / (j - i)
            rows[k]["arr"] = rows[k]["dep"] = t0 + w * (t1 - t0)
            rows[k]["interp"] = True
    for r in rows:
        r["timepoint"] = r["tp"] == "1" if r["tp"] in ("0", "1") else not r["interp"]
    return rows


def stop_json(pos, arr, dep, timepoint):
    return {"x": round(pos[0], 2), "y": round(pos[1], 2), "arr": round(arr, 1), "dep": round(dep, 1), "timepoint": timepoint}


def clip_trip(rows, rect, shape):
    """Clip a trip to the map box: keep the longest contiguous in-box stop run,
    with pseudo-stops (arr == dep) interpolated where the path between the
    boundary pair of stops crosses the box edge. A trip with no in-box stop
    whose path still crosses the box becomes entry + exit pseudo-stops."""
    flags = [in_rect(r["pos"], rect) for r in rows]
    if not any(flags) and shape is not None and not shape[2]:
        return None, "outside-box"
    chain = {}

    def ch(i):
        if i not in chain:
            chain[i] = chainage(shape[0], shape[1], rows[i]["pos"])
        return chain[i]

    def cross(i, entering):
        a, b = rows[i], rows[i + 1]
        path = sub_path(shape, ch(i), ch(i + 1), a["pos"], b["pos"]) if shape else [a["pos"], b["pos"]]
        evs = [(f, pt) for f, pt, ent in crossings(path, rect) if ent == entering]
        if not evs:
            return None
        f, pt = evs[-1] if entering else evs[0]
        t = a["dep"] + f * (b["arr"] - a["dep"])
        return stop_json(pt, t, t, False)

    if any(flags):
        best, i = (0, 0, -1), 0
        while i < len(flags):
            j = i
            while flags[i] and j + 1 < len(flags) and flags[j + 1]:
                j += 1
            if flags[i] and j - i + 1 > best[0]:
                best = (j - i + 1, i, j)
            i = j + 1
        _, i0, i1 = best
        stops = [stop_json(r["pos"], r["arr"], r["dep"], r["timepoint"]) for r in rows[i0 : i1 + 1]]
        if i0 > 0:
            ent = cross(i0 - 1, True)
            if ent:
                stops.insert(0, ent)
        if i1 < len(rows) - 1:
            ext = cross(i1, False)
            if ext:
                stops.append(ext)
    else:
        ent = ext = None
        for i in range(len(rows) - 1):
            ent = cross(i, True)
            if ent:
                break
        for i in reversed(range(len(rows) - 1)):
            ext = cross(i, False)
            if ext:
                break
        if not ent or not ext or ext["arr"] < ent["arr"]:
            return None, "outside-box"
        stops = [ent, ext]
    if len(stops) < 2:
        return None, "too-few-stops"
    seq = [t for s in stops for t in (s["arr"], s["dep"])]
    if any(b < a for a, b in zip(seq, seq[1:])):
        return None, "non-monotone-times"
    return stops, None


def compile_feed(label, blob, rect, origin):
    """One GTFS zip → (rail trips, bus trips, agency names)."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    lat0, lon0 = origin
    agency = {r.get("agency_id") or "": r.get("agency_name") or "" for r in open_table(z, "agency.txt")}
    routes = {r["route_id"]: r for r in open_table(z, "routes.txt") if r.get("route_id")}
    cal = [r for r in open_table(z, "calendar.txt") if r.get("service_id")]
    cal_dates = [r for r in open_table(z, "calendar_dates.txt") if r.get("service_id")]
    wed, sat = pick_dates(cal, cal_dates)
    svc = {wknd: active_services(d, cal, cal_dates) for d, wknd in ((wed, False), (sat, True)) if d}
    if not svc:
        print(f"  {label}: no usable service dates")
        return [], [], list(agency.values())

    drops = Counter()
    wanted = {}
    for r in open_table(z, "trips.txt"):
        tid, route = r.get("trip_id"), routes.get(r.get("route_id"))
        if not tid or route is None:
            continue
        days = [wknd for wknd, ids in svc.items() if r.get("service_id") in ids]
        if not days:
            continue
        cls = route_class(route, agency)
        if cls is None:
            drops["route-type-skipped"] += 1
            continue
        wanted[tid] = (route, cls, r.get("shape_id") or None, days)

    stops = {}
    for r in open_table(z, "stops.txt"):
        try:
            stops[r["stop_id"]] = project(float(r["stop_lat"]), float(r["stop_lon"]), lat0, lon0)
        except (KeyError, ValueError):
            continue
    shapes = load_shapes(z, {sid for _, _, sid, _ in wanted.values() if sid}, rect, origin)
    stop_times = load_stop_times(z, wanted)
    freqs = defaultdict(list)
    for r in open_table(z, "frequencies.txt"):
        s, e = parse_time(r.get("start_time")), parse_time(r.get("end_time"))
        try:
            h = int(r.get("headway_secs") or 0)
        except ValueError:
            h = 0
        if r.get("trip_id") and s is not None and e is not None and h > 0:
            freqs[r["trip_id"]].append((s, e, h))

    rail, bus = [], []
    for tid, (route, cls, sid, days) in wanted.items():
        raw = stop_times.get(tid)
        if not raw:
            drops["no-stop-times"] += 1
            continue
        rows = timed_rows(raw, stops)
        if len(rows) < 2:
            drops["too-few-stops"] += 1
            continue
        clipped, reason = clip_trip(rows, rect, shapes.get(sid))
        if clipped is None:
            drops[reason] += 1
            continue
        offsets = [0.0]
        if tid in freqs:
            base = rows[0]["arr"]
            offsets = [t - base for s, e, h in freqs[tid] for t in range(int(s), int(e), h)]
        for wknd in days:
            for off in offsets:
                st = clipped if off == 0.0 else [dict(s, arr=round(s["arr"] + off, 1), dep=round(s["dep"] + off, 1)) for s in clipped]
                if cls == "bus":
                    line = route.get("route_short_name") or route.get("route_long_name") or route["route_id"]
                    bus.append({"line": line, "weekend": wknd, "stops": st})
                else:
                    rail.append({"class": cls, "weekend": wknd, "carriages": CARRIAGES[cls], "stops": st})
    detail = ", ".join(f"{k}: {v}" for k, v in sorted(drops.items()))
    print(f"  {label} (weekday {wed}, saturday {sat}): {len(rail)} rail + {len(bus)} bus trips kept" + (f"; dropped {detail}" if detail else ""))
    return rail, bus, [a for a in agency.values() if a]


def cap_bus(bus, cap, center):
    """Keep whole bus lines nearest the bbox centre until the cap is reached."""
    if len(bus) <= cap:
        return bus
    by_line = defaultdict(list)
    for t in bus:
        by_line[t["line"]].append(t)

    def dist(line):
        return min(math.hypot(s["x"] - center[0], s["y"] - center[1]) for t in by_line[line] for s in t["stops"])

    kept = []
    for line in sorted(by_line, key=dist):
        if len(kept) + len(by_line[line]) <= cap:
            kept.extend(by_line[line])
        else:
            print(f"bus cap: dropped line {line} ({len(by_line[line])} trips, {dist(line):.0f} m from centre)")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="map artifact (.json or .json.gz); supplies bbox + origin")
    ap.add_argument("--out", default="map.transit.json.gz")
    ap.add_argument("--feed", action="append", default=[], help="GTFS zip URL or local path (repeatable); default: discover from the Mobility Database")
    ap.add_argument("--max-feeds", type=int, default=8)
    ap.add_argument("--bus-cap", type=int, default=400)
    args = ap.parse_args()

    bbox, origin = load_map(args.map)
    rect = bbox_rect(bbox, origin)

    sources = []  # (label, fetch url/path, meta url, license url, attribution)
    if args.feed:
        sources = [(f.rstrip("/").split("/")[-1] or f, f, f, "", None) for f in args.feed]
    else:
        for name, url, lic in discover_feeds(tuple(bbox), args.max_feeds):
            sources.append((name, url, url, lic, name))
    key = os.environ.get("TRANSIT_511_API_KEY")
    if key:
        sources.append(("511.org regional", f"{URL_511}&api_key={key}", URL_511, "", "Data provided by 511.org"))

    rail, bus = [], []
    meta = {"attribution": [], "license_urls": [], "feeds": []}
    for label, src, url, lic, attrib in sources:
        if os.path.exists(src):
            with open(src, "rb") as f:
                blob = f.read()
        else:
            blob = http_get(src)
        if blob is None:
            continue
        try:
            frail, fbus, agencies = compile_feed(label, blob, rect, origin)
        except (zipfile.BadZipFile, KeyError, ValueError, OSError) as err:
            print(f"warning: {label}: {err!r}; skipped")
            continue
        if not frail and not fbus:
            continue
        rail += frail
        bus += fbus
        meta["attribution"].append(attrib or ", ".join(agencies) or label)
        if lic:
            meta["license_urls"].append(lic)
        meta["feeds"].append(url)
    meta["license_urls"] = list(dict.fromkeys(meta["license_urls"]))

    center = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
    bus = cap_bus(bus, args.bus_cap, center)

    doc = {"meta": meta, "rail_trips": rail, "bus_trips": bus}
    payload = json.dumps(doc, separators=(",", ":"))
    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wt") as f:
        f.write(payload)
    print(f"wrote {args.out}: {len(rail)} rail trips, {len(bus)} bus trips from {len(meta['feeds'])} feeds")


if __name__ == "__main__":
    main()
