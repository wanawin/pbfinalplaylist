import streamlit as st
from collections import Counter
from itertools import permutations
import csv, os, re
import pandas as pd
import numpy as np
# --- UI Patch: Eliminate combos whose SUM matches any of the last N seed sums ---
# Drop-in snippet for your Streamlit app (place in your Filters section)

import pandas as pd
import streamlit as st

def get_last_n_seed_sums(draws_df: pd.DataFrame, n: int) -> set:
    """
    draws_df: DataFrame of historical draws with numeric columns P1..P5 (ordered chronologically).
    Returns a set of the last N *seed* sums (sum of previous draw's five numbers).
    """
    # Seed is previous draw; its sum is the sum of that row
    seed_sums = draws_df[["P1","P2","P3","P4","P5"]].sum(axis=1)
    # Take the last N seed sums (most recent N rows excluding the very last if you wish to avoid self-reference)
    last_n = seed_sums.tail(n)
    return set(last_n.tolist())

def sum_from_combo_row(row: pd.Series) -> int:
    """Supports pools with P1..P5 columns OR a 'combo'/'Result' string like '03-18-22-27-33'."""
    if {"P1","P2","P3","P4","P5"}.issubset(row.index):
        return int(row["P1"]) + int(row["P2"]) + int(row["P3"]) + int(row["P4"]) + int(row["P5"])  
    for key in ("combo", "Result", "numbers"):
        if key in row and isinstance(row[key], str):
            parts = [int(x) for x in row[key].replace(","," ").replace("-"," ").split()[:5]]
            return sum(parts)
    raise ValueError("Pool row must have P1..P5 or a combo/Result string with five numbers.")

@st.cache_data(show_spinner=False)
def apply_sum_in_last_n_seeds_filter(pool_df: pd.DataFrame, draws_df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """
    Returns (kept_df, eliminated_df, blocked_sums)
    """
    blocked_sums = get_last_n_seed_sums(draws_df, n)
    pool_with_sum = pool_df.copy()
    pool_with_sum["_sum"] = pool_with_sum.apply(sum_from_combo_row, axis=1)
    mask_keep = ~pool_with_sum["_sum"].isin(blocked_sums)
    kept = pool_with_sum[mask_keep].drop(columns=["_sum"], errors="ignore")
    eliminated = pool_with_sum[~mask_keep].drop(columns=["_sum"], errors="ignore")
    return kept, eliminated, blocked_sums

# ----------------- UI Controls -----------------
st.subheader("Sum Filter · Block sums seen in recent seeds")
colA, colB, colC = st.columns([1.2,1,1.2])
with colA:
    enable_sum_block = st.checkbox("Enable 'Sum in Last N Seeds' filter", value=False)
with colB:
    lookback_n = st.number_input("Lookback N (seeds)", min_value=5, max_value=60, value=20, step=1)
with colC:
    st.caption("Historically, N=20 blocks ~10% of winners (rate varies by dataset).")

if enable_sum_block:
    try:
        # EXPECTED: you already have these DataFrames in your app
        # pool_df: your current candidate combinations
        # draws_df: historical draws with P1..P5 numeric columns
        kept_df, eliminated_df, blocked = apply_sum_in_last_n_seeds_filter(pool_df, draws_df, lookback_n)
        st.success(f"Applied: blocked {len(blocked)} unique sums · Eliminated {len(eliminated_df):,} · Kept {len(kept_df):,}")
        with st.expander("View eliminated combos"):
            st.dataframe(eliminated_df, use_container_width=True, height=300)
        # Replace pool_df in-session if you chain filters
        pool_df = kept_df
    except Exception as e:
        st.error(f"Sum-in-last-N filter error: {e}")

F208,Seed-sum last-digit rule: if seed sum ends in 3, eliminate combos with sum ending in 1 (153/155 kept),True,(sum(seed) % 10) == 3,"(sum(combo) % 10) == 1",,,,,,,,,,


# ----------------------
# Helpers: parsing & normalization
# ----------------------
def _split_tokens(text: str):
    toks = []
    for line in (text or "").splitlines():
        for token in re.split(r"[,\s]+", line):
            token = token.strip()
            if token:
                toks.append(token)
    return toks


def normalize_tens(text: str):
    """Accept tens survivors as 5-digit strings (digits 0-6), one per line/comma.
       Return (normalized_sorted_multiset_strings, bad_tokens)."""
    toks = _split_tokens(text)
    out, bad = [], []
    for tok in toks:
        digs = [c for c in tok if c.isdigit()]
        if len(digs) != 5 or any(c not in "0123456" for c in digs):
            bad.append(tok)
            continue
        out.append("".join(sorted(digs)))
    # de-dup, preserve order
    seen = set()
    out2 = []
    for o in out:
        if o not in seen:
            out2.append(o)
            seen.add(o)
    return out2, bad


def normalize_ones(text: str):
    """Accept ones survivors as 5-digit strings (digits 0-9)."""
    toks = _split_tokens(text)
    out, bad = [], []
    for tok in toks:
        digs = [c for c in tok if c.isdigit()]
        if len(digs) != 5 or any(c not in "0123456789" for c in digs):
            bad.append(tok)
            continue
        out.append("".join(sorted(digs)))
    seen = set()
    out2 = []
    for o in out:
        if o not in seen:
            out2.append(o)
            seen.add(o)
    return out2, bad


def normalize_final_sets(text: str):
    """Accept full 5-number sets like 01-16-21-47-60; return (list_of_tuples, bad_tokens)."""
    toks = _split_tokens(text)
    out, bad = [], []
    for tok in toks:
        nums = [int(x) for x in re.findall(r"\d+", tok)]
        if len(nums) != 5 or any(n < 1 or n > 69 for n in nums):
            bad.append(tok)
            continue
        out.append(tuple(sorted(nums)))
    seen = set()
    out2 = []
    for o in out:
        if o not in seen:
            out2.append(o)
            seen.add(o)
    return out2, bad

# ----------------------
# Pair tens+ones -> 5 numbers
# ----------------------
def all_unique_perms(seq):
    """Return unique permutations of given sequence (len <= 5)."""
    # Using set(permutations(...)) is fine for len=5 (120 perms).
    return set(permutations(seq, len(seq)))


def pair_tens_ones(tens_str, ones_str):
    """Return a set of sorted 5-number tuples (valid 1..69, all unique)."""
    t = [int(c) for c in tens_str]
    o = [int(c) for c in ones_str]
    nums_set = set()
    for p in all_unique_perms(o):
        nums = [10 * t[i] + p[i] for i in range(5)]
        if any(n < 1 or n > 69 for n in nums):
            continue
        if len(set(nums)) != 5:
            continue
        nums_set.add(tuple(sorted(nums)))
    return nums_set

# ----------------------
# Context + quick metrics
# ----------------------
def multiset_shared(a, b):
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())


def final_sum(a):
    return sum(a)


def final_range(a):
    return max(a) - min(a)


def ones_sum(a):
    return sum(n % 10 for n in a)


def tens_sum(a):
    return sum(n // 10 for n in a)


def build_ctx(combo_nums, seed_nums, prev_nums):
    fs = final_sum(combo_nums)
    fr = final_range(combo_nums)
    combo_tens = [n // 10 for n in combo_nums]
    combo_ones = [n % 10 for n in combo_nums]
    ctx = {
        "combo_numbers": combo_nums,
        "seed_numbers": seed_nums,
        "prev_seed_numbers": prev_nums,
        "final_sum": fs,
        "final_range": fr,
        "final_min": min(combo_nums),
        "final_max": max(combo_nums),
        "combo_tens": combo_tens,
        "combo_ones": combo_ones,
        "ones_sum": sum(combo_ones),
        "tens_sum": sum(combo_tens),
        # helpers
        "shared_numbers": multiset_shared,
        "shared_ones": multiset_shared,
        "shared_tens": multiset_shared,
        "sum": sum,
        "min": min,
        "max": max,
        "len": len,
        "abs": abs,
        "set": set,
        "any": any,
        "all": all,
        "range": range,
    }
    return ctx

# ----------------------
# History & percentile bands
# ----------------------
def load_history_rows(history_path="pwrbll.txt"):
    rows = []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                m = re.search(r"^(.*?),\s*Powerball:\s*(\d+)", ln)
                if not m:
                    continue
                date_and_set, _ = m.groups()
                parts = re.split(r"\t", date_and_set)
                if not parts:
                    continue
                five = parts[-1]
                nums = [int(x) for x in re.findall(r"\d{2}", five)]
                if len(nums) == 5:
                    rows.append(tuple(sorted(nums)))
        rows = rows[::-1]  # chronological
    except Exception:
        rows = []
    return rows


def compute_percentile_bands(history_rows, lo_q=0.10, hi_q=0.90):
    if not history_rows:
        # sensible defaults
        return {
            "sum": (122, 222),
            "range": (30, 60),
            "ones": (15, 29),
            "tens": (10, 20),
        }
    sums = pd.Series([final_sum(a) for a in history_rows])
    rngs = pd.Series([final_range(a) for a in history_rows])
    ones = pd.Series([ones_sum(a) for a in history_rows])
    tens = pd.Series([tens_sum(a) for a in history_rows])

    def band(series):
        lo = int(np.floor(series.quantile(lo_q)))
        hi = int(np.ceil(series.quantile(hi_q)))
        return lo, hi

    return {"sum": band(sums), "range": band(rngs), "ones": band(ones), "tens": band(tens)}


def in_band(val, lo, hi):
    return (lo is None or val >= lo) and (hi is None or val <= hi)

# ----------------------
# App
# ----------------------
def main():
    st.sidebar.header("🔗 Final Pairing — Tens × Ones → 5 numbers")

    # Tens & Ones inputs
    t_text = st.sidebar.text_area("Paste tens survivors (one per line, e.g., 11566)", height=150)
    o_text = st.sidebar.text_area("Paste ones survivors (one per line, e.g., 57999)", height=150)

    # Seed winner inputs
    seed_text = st.sidebar.text_input(
        "Seed winner (5 nums 1–69, e.g., 01-16-21-47-60)", value=""
    ).strip()
    prev_text = st.sidebar.text_input("Prev winner (optional)", value="").strip()

    def parse_numbers(s: str):
        nums = [int(x) for x in re.findall(r"\d+", s or "")]
        nums = [n for n in nums if 1 <= n <= 69]
        return sorted(nums) if len(nums) == 5 else []

    seed_numbers = parse_numbers(seed_text)
    prev_numbers = parse_numbers(prev_text)

    # Normalize tens/ones
    tens_list, bad_t = normalize_tens(t_text)
    ones_list, bad_o = normalize_ones(o_text)
    if bad_t:
        st.sidebar.warning(
            "Ignored invalid tens entries: "
            + ", ".join(bad_t[:5])
            + (" ..." if len(bad_t) > 5 else "")
        )
    if bad_o:
        st.sidebar.warning(
            "Ignored invalid ones entries: "
            + ", ".join(bad_o[:5])
            + (" ..." if len(bad_o) > 5 else "")
        )

    st.sidebar.markdown(f"**Tens combos:** {len(tens_list)}")
    st.sidebar.markdown(f"**Ones combos:** {len(ones_list)}")

    # Track/Test combos
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area(
        "Track/Test 5-number sets (e.g., 01-16-21-47-60)", height=120
    )
    preserve_tracked = st.sidebar.checkbox(
        "Preserve tracked combos during filtering", value=True
    )
    inject_tracked = st.sidebar.checkbox(
        "Inject tracked combos even if not generated", value=False
    )
    tracked, _ = normalize_final_sets(track_text)
    Tracked = set(tracked)

    # Percentile auto-screen settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Auto percentile screen")
    auto_on = st.sidebar.checkbox(
        "Apply after generation (before deduplication)", value=True
    )
    pct_window = st.sidebar.selectbox("Band", ["P10–P90", "P20–P80"], index=0)
    use_sum = st.sidebar.checkbox("Use FINAL SUM band", value=True)
    use_range = st.sidebar.checkbox("Use FINAL RANGE band", value=True)
    use_ones = st.sidebar.checkbox("Use ONES-SUM band", value=False)
    use_tens = st.sidebar.checkbox("Use TENS-SUM band", value=False)

    # Compute bands from history
    history_rows = load_history_rows("pwrbll.txt")
    lo_q, hi_q = (0.10, 0.90) if pct_window == "P10–P90" else (0.20, 0.80)
    bands = compute_percentile_bands(history_rows, lo_q, hi_q)

    # Combine tens × ones (RAW list so we can filter before dedup)
    st.header("🧮 Combine Tens × Ones → Candidates")
    raw = []
    for t in tens_list:
        for o in ones_list:
            for combo in pair_tens_ones(t, o):
                raw.append(combo)
    st.write(f"Generated candidates (pre-filter RAW): {len(raw):,}")

    # Inject tracked combos before auto filter if requested
    if inject_tracked:
        for c in tracked:
            raw.append(c)

    # Auto percentile screening
    def pass_pct(c):
        if not auto_on:
            return True
        fs = final_sum(c)
        fr = final_range(c)
        os = ones_sum(c)
        ts = tens_sum(c)
        if use_sum and not in_band(fs, *bands["sum"]):
            return False
        if use_range and not in_band(fr, *bands["range"]):
            return False
        if use_ones and not in_band(os, *bands["ones"]):
            return False
        if use_tens and not in_band(ts, *bands["tens"]):
            return False
        return True

    screened = [c for c in raw if (c in Tracked and preserve_tracked) or pass_pct(c)]
    st.write(f"After auto percentile screen (pre-dedup): {len(screened):,}")

    # Now deduplicate
    candidates = sorted(set(screened))
    st.write(f"After deduplication: {len(candidates):,}")

    # Show bands for transparency
    with st.expander("Percentile bands in use"):
        st.write(
            {
                "SUM": bands["sum"],
                "RANGE": bands["range"],
                "ONES-SUM": bands["ones"],
                "TENS-SUM": bands["tens"],
            }
        )

    # ======================= CSV-driven Manual Filters (final-stage) =======================
    from pathlib import Path

    st.header("🛠️ Manual Filters (final-stage)")
    st.write(
        "Percentile screens already applied above. You can add/stack CSV-based filters here."
    )

    _cols = ["id", "name", "enabled", "applicable_if", "expression"]

    def _load_filters_csv(src):
        try:
            df = pd.read_csv(src, dtype=str).fillna("")
        except Exception as e:
            st.warning(f"Could not read filter CSV: {e}")
            return pd.DataFrame(columns=_cols)
        for c in _cols:
            if c not in df.columns:
                df[c] = ""
        return df[_cols]

    use_default = st.checkbox(
        "Use default final filters (pb_final_filters_all.csv)", value=True
    )
    uploaded = st.file_uploader("Upload additional filter CSV (optional)", type="csv")

    _filters_df = pd.DataFrame(columns=_cols)
    if use_default:
        _default_path = Path(__file__).with_name("pb_final_filters_all.csv")
        if _default_path.exists():
            _filters_df = pd.concat(
                [_filters_df, _load_filters_csv(_default_path)], ignore_index=True
            )
        else:
            st.warning("Default pack pb_final_filters_all.csv not found alongside this app.")

    if uploaded is not None:
        _filters_df = pd.concat(
            [_filters_df, _load_filters_csv(uploaded)], ignore_index=True
        )

    if not _filters_df.empty:
        # Compile rules (safe)
        _compiled = []
        _invalid_rows = []
        for _, r in _filters_df.iterrows():
            # keep only rows explicitly enabled (blank/true/1/yes)
            if str(r["enabled"]).strip().lower() not in ("", "true", "1", "yes"):
                continue
            fid = (str(r["id"]).strip() or "UNKNOWN")
            name = (str(r["name"]).strip() or fid)
            app = (str(r["applicable_if"]).strip() or "True")
            expr = str(r.get("expression", "")).strip()

            # mark invalid expression (empty or literal true)
            expr_invalid = (not expr) or (expr.lower() in {"true", "1"})
            try:
                app_c = compile(app, f"<appif:{fid}>", "eval")
            except Exception:
                # if applicable_if is broken, treat as never applicable
                app_c = compile("False", f"<appif:{fid}>", "eval")

            expr_c = None
            if not expr_invalid:
                try:
                    expr_c = compile(expr, f"<expr:{fid}>", "eval")
                except Exception:
                    expr_c = None
                    expr_invalid = True

            if expr_invalid:
                _invalid_rows.append((fid, name, app, expr))
            _compiled.append((fid, name, app_c, expr_c))

        with st.expander("Inspect first 20 rules"):
            try:
                st.dataframe(_filters_df[["id","applicable_if","expression"]].head(20))
            except Exception:
                st.write("(no preview)")
            if _invalid_rows:
                st.warning(f"{len(_invalid_rows)} rules have invalid or literal-True expressions and will be ignored unless you edit the CSV.")

        # ---------- Performance controls ----------
        fast_mode = st.checkbox(
            "⚡ Fast mode (skip init-cut counting)", value=True,
            help="Keeps UI snappy. If off, app estimates per-filter cuts (slow for big pools)."
        )
        sample_for_counts = 0
        if not fast_mode:
            sample_for_counts = st.number_input(
                "Sample size for init-cut counts (0 = full pool)",
                min_value=0, max_value=len(candidates),
                value=min(5000, len(candidates)), step=1000,
            )

        _hide_zero = st.checkbox("Hide filters with 0 initial eliminations", value=True)

        # Buttons that truly toggle checkboxes
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Select all"):
                for fid, *_ in _compiled:
                    st.session_state[f"final_{fid}"] = True
        with c2:
            if st.button("Deselect all"):
                for fid, *_ in _compiled:
                    st.session_state[f"final_{fid}"] = False

        # ---------- Init-cut counting (optional) ----------
        _init_counts = {}
        if not fast_mode:
            import random
            pool_for_counts = candidates if sample_for_counts == 0 else random.sample(
                candidates, min(sample_for_counts, len(candidates))
            )
            for fid, name, app_c, expr_c in _compiled:
                # skip invalid expressions in counting
                if expr_c is None:
                    _init_counts[fid] = 0
                    continue
                cuts = 0
                for _c in pool_for_counts:
                    _ctx = build_ctx(_c, seed_numbers, prev_numbers)
                    try:
                        if eval(app_c, {}, _ctx) and eval(expr_c, {}, _ctx):
                            cuts += 1
                    except Exception:
                        pass
                # scale estimate if sampling
                if sample_for_counts and len(pool_for_counts) > 0:
                    cuts = int(round(cuts * (len(candidates) / len(pool_for_counts))))
                _init_counts[fid] = cuts

        # ---------- Render checkboxes ----------
        for fid, name, app_c, expr_c in _compiled:
            cuts = _init_counts.get(fid, None)  # None in fast mode
            if _hide_zero and (cuts is not None) and cuts == 0:
                continue
            key = f"final_{fid}"
            if key not in st.session_state:
                # default unchecked so user can choose
                st.session_state[key] = False
            label = f"{fid}: {name}"
            if expr_c is None:
                label += "  ⚠️ invalid expr"
            if cuts is not None:
                label += f" — init cuts {cuts}"
                if len(candidates) > 0 and cuts / len(candidates) >= 0.95:
                    label += "  ⚠️ (kills pool)"
            st.checkbox(label, key=key, disabled=(expr_c is None))

        # Active rule map from session state
        _active = {fid: st.session_state.get(f"final_{fid}", False) for fid, *_ in _compiled}

        # ---------- Apply selected rules ----------
        _survivors = []
        for _c in candidates:
            _ctx = build_ctx(_c, seed_numbers, prev_numbers)
            eliminated = False
            for fid, name, app_c, expr_c in _compiled:
                if not _active.get(fid, False) or expr_c is None:
                    continue
                try:
                    if eval(app_c, {}, _ctx) and eval(expr_c, {}, _ctx):
                        eliminated = True
                        break
                except Exception:
                    pass
            if not eliminated:
                _survivors.append(_c)

        candidates = _survivors  # update pool

    # Survivors block
    st.subheader(f"Remaining after manual filters: {len(candidates)}")

    st.markdown("### ✅ Final Survivors")
    with st.expander("Show remaining 5-number sets"):
        tracked_survivors = [c for c in candidates if c in Tracked]
        if tracked_survivors:
            st.write("**Tracked survivors:**")
            for c in tracked_survivors:
                st.write("-".join(f"{x:02d}" for x in c))
            st.write("---")
        for c in candidates:
            if c not in Tracked:
                st.write("-".join(f"{x:02d}" for x in c))

    # Downloads
    def fmt_combo(c):
        return "-".join(f"{int(x):02d}" for x in sorted(c))

    df_out = pd.DataFrame({"numbers": [fmt_combo(c) for c in candidates]})
    st.download_button(
        "Download survivors (CSV)",
        df_out.to_csv(index=False),
        file_name="pb_final_survivors.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download survivors (TXT)",
        "\n".join(df_out["numbers"]),
        file_name="pb_final_survivors.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
