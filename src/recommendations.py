"""Evidence-based corrective action engine.

Every recommendation cites the empirically observed issue rate for the
order's condition (from training-data evidence tables) versus the overall
base rate. Actions are operational suggestions, not guarantees.
"""
from __future__ import annotations

import numpy as np

PRIORITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _numeric_bin(ev, feature, value):
    t = ev["numeric"].get(feature)
    if t is None:
        return None, None
    edges = np.asarray(t["edges"])
    idx = int(np.clip(np.searchsorted(edges, value, side="right") - 1,
                      0, len(t["issue_rate"]) - 1))
    return t["labels"][idx], t["issue_rate"][idx]


def _cat_rate(ev, feature, level):
    t = ev["categorical"].get(feature)
    if t is None:
        return None
    m = dict(zip(t["level"], t["issue_rate"]))
    return m.get(level)


def _ev_text(label, rate, base):
    if rate is None:
        return ""
    return f"observed issue rate for {label}: {rate:.0%} vs overall {base:.0%}"


def generate_recommendations(order: dict, contributions: list, bundle: dict) -> list:
    ev = bundle["evidence"]
    base = ev["base_rate"]
    recs = []

    def add(feature, title, action, priority, label, rate):
        recs.append({"feature": feature, "title": title, "action": action,
                     "priority": priority, "evidence": _ev_text(label, rate, base)})

    top_features = [c["feature"] for c in (contributions or [])[:3]
                    if c.get("contribution", 0) > 0]

    # --- rainfall
    v = order.get("rainfall_mm")
    if v is not None and v >= 5:
        label, r = _numeric_bin(ev, "rainfall_mm", v)
        add("rainfall_mm", "Wet-weather conditions",
            "Activate rain protocol: waterproof packaging, extended SLA buffer, "
            "and rider rain gear before dispatch.",
            "High" if v >= 10 else "Medium", f"rainfall of {v:g} mm", r)

    # --- traffic
    v = order.get("traffic_index")
    if v is not None and v >= 7:
        label, r = _numeric_bin(ev, "traffic_index", v)
        add("traffic_index", "Severe traffic conditions",
            "Reroute via low-congestion corridors and add a dispatch buffer "
            "to the promised delivery time.",
            "High" if v >= 9 else "Medium", f"traffic index {v:g}", r)

    # --- distance
    v = order.get("distance_km")
    if v is not None and v >= 7.5:
        label, r = _numeric_bin(ev, "distance_km", v)
        add("distance_km", "Long delivery distance",
            "Assign a zone-optimised rider and avoid batching additional orders "
            "onto this run.",
            "Medium", f"distance of {v:g} km", r)
        vt = order.get("vehicle_type")
        if vt == "Bicycle":
            add("vehicle_type", "Bicycle assigned to a long route",
                "Upgrade the assignment to a scooter or motorcycle for this distance.",
                "High", f"{vt} on {v:g} km", _cat_rate(ev, "vehicle_type", vt))

    # --- restaurant load
    v = order.get("restaurant_prep_est_min")
    if v is not None and v >= 25:
        label, r = _numeric_bin(ev, "restaurant_prep_est_min", v)
        add("restaurant_prep_est_min", "High kitchen prep time",
            "Notify the restaurant to prioritise this ticket and alert the customer "
            "with a realistic ETA.",
            "Medium", f"prep estimate {v:g} min", r)

    v = order.get("orders_ahead")
    if v is not None and v >= 5:
        label, r = _numeric_bin(ev, "orders_ahead", v)
        add("orders_ahead", "Rider queue congestion",
            "Rebalance rider load or dispatch surge riders for this zone.",
            "Medium", f"{v:g} orders already in queue", r)

    # --- peak hour combined with load
    if order.get("is_peak_hour") == 1 and (order.get("orders_ahead") or 0) >= 4:
        add("is_peak_hour", "Peak-hour overload",
            "Trigger the peak protocol: dynamic dispatch buffers and proactive "
            "customer updates on delay risk.",
            "Medium", "peak hour with a busy rider queue",
            _cat_rate(ev, "is_peak_hour", 1))

    # --- rider experience on a demanding route
    rx = order.get("rider_experience_months")
    demanding = (order.get("distance_km") or 0) >= 7.5 or (order.get("traffic_index") or 0) >= 7
    if rx is not None and rx <= 6 and demanding:
        label, r = _numeric_bin(ev, "rider_experience_months", rx)
        add("rider_experience_months", "Low-experience rider on a demanding route",
            "Pair with an experienced rider or enable assisted navigation for this run.",
            "Medium", f"rider with {rx:g} months experience", r)

    # --- high value order
    ov = order.get("order_value_inr")
    if ov is not None and ov >= 1000 and top_features:
        add("order_value_inr", "High-value order at elevated risk",
            "Flag for priority handling and proactive support follow-up.",
            "High", f"order value Rs {ov:g}", _numeric_bin(ev, "order_value_inr", ov)[1])

    # --- city concentration
    ct = order.get("city")
    if ct:
        r = _cat_rate(ev, "city", ct)
        if r is not None and r > base * 1.3:
            add("city", "City-level risk concentration",
                f"Review {ct} operations: rider density, SLA compliance and "
                "restaurant partner performance.",
                "Medium", f"{ct}", r)

    # boost priority when the model also blames this factor
    for rec in recs:
        if rec["feature"] in top_features and PRIORITY_RANK[rec["priority"]] > 1:
            rec["priority"] = "High"
        rec["rank"] = PRIORITY_RANK[rec["priority"]]

    recs.sort(key=lambda d: d["rank"])
    if not recs:
        recs.append({"feature": None, "title": "No elevated-risk conditions detected",
                     "action": "Standard monitoring applies; no intervention recommended.",
                     "priority": "Low", "evidence": "", "rank": 3})
    return recs[:6]
