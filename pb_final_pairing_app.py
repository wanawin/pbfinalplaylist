
import streamlit as st
from collections import Counter
from itertools import permutations
import csv, os, re
import pandas as pd
import numpy as np

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
            out2.append(o); seen.add(o)
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
            out2.append(o); seen.add(o)
    return out2, bad

def normalize_final_sets(text: str):
    """Accept full 5-number sets like 01-16-21-47-60; return (list_of_tuples, bad_tokens)."""
    toks = _split_tokens(text)
    out, bad = [], []
    for tok in toks:
        nums = [int(x) for x in re.findall(r"\d+", tok)]
        if len(nums) != 5 or any(n < 1 or n > 69 for n in nums):
            bad.append(tok); continue
        out.append(tuple(sorted(nums)))
    seen = set()
    out2 = []
    for o in out:
        if o not in seen:
            out2.append(o); seen.add(o)
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

def final_sum(a): return sum(a)
def final_range(a): return max(a)-min(a)
def ones_sum(a): return sum(n%10 for n in a)
def tens_sum(a): return sum(n//10 for n in a)

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
        "sum": sum, "min": min, "max": max, "len": len, "abs": abs, "set": set, "any": any, "all": all, "range": range
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
                if not ln: continue
                m = re.search(r"^(.*?),\s*Powerball:\s*(\d+)", ln)
                if not m: continue
                date_and_set, _ = m.groups()
                parts = re.split(r"\t", date_and_set)
                if not parts: continue
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
    return {
        "sum": band(sums),
        "range": band(rngs),
        "ones": band(ones),
        "tens": band(tens),
    }

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
    seed_text = st.sidebar.text_input("Seed winner (5 nums 1–69, e.g., 01-16-21-47-60)", value="").strip()
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
        st.sidebar.warning("Ignored invalid tens entries: " + ", ".join(bad_t[:5]) + (" ..." if len(bad_t) > 5 else ""))
    if bad_o:
        st.sidebar.warning("Ignored invalid ones entries: " + ", ".join(bad_o[:5]) + (" ..." if len(bad_o) > 5 else ""))

    st.sidebar.markdown(f"**Tens combos:** {len(tens_list)}")
    st.sidebar.markdown(f"**Ones combos:** {len(ones_list)}")

    # Track/Test combos
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area("Track/Test 5-number sets (e.g., 01-16-21-47-60)", height=120)
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)
    tracked, _ = normalize_final_sets(track_text)
    Tracked = set(tracked)

    # Percentile auto-screen settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Auto percentile screen")
    auto_on = st.sidebar.checkbox("Apply after generation (before deduplication)", value=True)
    pct_window = st.sidebar.selectbox("Band", ["P10–P90","P20–P80"], index=0)
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
        fs = final_sum(c); fr = final_range(c); os = ones_sum(c); ts = tens_sum(c)
        if use_sum and not in_band(fs, *bands["sum"]): return False
        if use_range and not in_band(fr, *bands["range"]): return False
        if use_ones and not in_band(os, *bands["ones"]): return False
        if use_tens and not in_band(ts, *bands["tens"]): return False
        return True

    screened = [c for c in raw if (c in Tracked and preserve_tracked) or pass_pct(c)]
    st.write(f"After auto percentile screen (pre-dedup): {len(screened):,}")

    # Now deduplicate
    candidates = sorted(set(screened))
    st.write(f"After deduplication: {len(candidates):,}")

    # Show bands for transparency
    with st.expander("Percentile bands in use"):
        # === CSV-driven manual filters (final-stage) ===
candidates = manual_filters_ui(
    candidates=candidates,
    seed_numbers=seed_numbers,
    prev_seed_numbers=prev_seed_numbers,
)
st.caption(f"Manual filters block ran — pool now: {len(candidates)}")

        st.write({
            "SUM": bands["sum"],
            "RANGE": bands["range"],
            "ONES-SUM": bands["ones"],
            "TENS-SUM": bands["tens"],
        })

def manual_filters_ui(candidates, seed_numbers, prev_seed_numbers):
    """CSV-driven final-stage filters. Returns the filtered candidate list."""
    import pandas as pd
    from pathlib import Path
    import streamlit as st

    st.header("🛠️ Manual Filters (final-stage)")
    st.write("Percentile screens already applied above. You can add/stack CSV-based filters here.")

    # ---------- Load filter CSVs ----------
    cols = ["id", "name", "enabled", "applicable_if", "expression"]

    def load_filters_csv(src):
        try:
            df = pd.read_csv(src, dtype=str).fillna("")
        except Exception as e:
            st.warning(f"Could not read filter CSV: {e}")
            return pd.DataFrame(columns=cols)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]

    use_default = st.checkbox("Use default final filters (pb_final_filters_all.csv)", value=True)
    uploaded = st.file_uploader("Upload additional filter CSV (optional)", type="csv")

    filters_df = pd.DataFrame(columns=cols)
    if use_default:
        default_path = Path(__file__).with_name("pb_final_filters_all.csv")
        if default_path.exists():
            filters_df = pd.concat([filters_df, load_filters_csv(default_path)], ignore_index=True)
        else:
            st.warning("Default pack pb_final_filters_all.csv not found alongside this app.")

    if uploaded is not None:
        filters_df = pd.concat([filters_df, load_filters_csv(uploaded)], ignore_index=True)

    if filters_df.empty:
        st.info("No manual filters loaded; skipping this stage.")
        st.subheader(f"Remaining after manual filters: {len(candidates)}")
        return candidates

    # ---------- Compile rules ----------
    compiled = []
    for _, r in filters_df.iterrows():
        enabled_flag = str(r["enabled"]).strip().lower()
        if enabled_flag not in ("", "true", "1", "yes"):
            continue
        fid  = (str(r["id"]).strip() or "UNKNOWN")
        name = (str(r["name"]).strip() or fid)
        app  = (str(r["applicable_if"]).strip() or "True")
        expr = (str(r["expression"]).strip())
        try:
            app_c  = compile(app,  f"<appif:{fid}>", "eval")
            expr_c = compile(expr, f"<expr:{fid}>",  "eval")
            compiled.append((fid, name, app_c, expr_c))
        except Exception as e:
            st.warning(f"Skipping {fid} (compile error): {e}")

    if not compiled:
        st.info("No enabled/valid rules after compilation.")
        st.subheader(f"Remaining after manual filters: {len(candidates)}")
        return candidates

    # ---------- Preview initial elimination counts ----------
    hide_zero = st.checkbox("Hide filters with 0 initial eliminations", value=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        sel_all = st.button("Select all")
    with col2:
        desel_all = st.button("Deselect all")

    init_counts = {}
    for fid, name, app_c, expr_c in compiled:
        cuts = 0
        for c in candidates:
            ctx = build_ctx(c, seed_numbers, prev_seed_numbers)
            try:
                if eval(app_c, {}, ctx) and eval(expr_c, {}, ctx):
                    cuts += 1
            except Exception:
                # Bad rule or context error; treat as no-cut
                pass
        init_counts[fid] = cuts

    # ---------- Let user pick active rules ----------
    active = {}
    for fid, name, app_c, expr_c in compiled:
        cuts = init_counts.get(fid, 0)
        if hide_zero and cuts == 0:
            continue
        label = f"{fid}: {name} — init cuts {cuts}"
        default_checked = True if (sel_all or (cuts > 0 and not desel_all)) else False
        active[fid] = st.checkbox(label, value=default_checked, key=f"final_{fid}")

    # ---------- Apply selected rules ----------
    survivors = []
    for c in candidates:
        ctx = build_ctx(c, seed_numbers, prev_seed_numbers)
        eliminated = False
        for fid, name, app_c, expr_c in compiled:
            if not active.get(fid, False):
                continue
            try:
                if eval(app_c, {}, ctx) and eval(expr_c, {}, ctx):
                    eliminated = True
                    break
            except Exception:
                pass
        if not eliminated:
            survivors.append(c)

    st.subheader(f"Remaining after manual filters: {len(survivors)}")
    return survivors


    st.download_button(
        "Download survivors (TXT)",
        "\n".join(df_out["numbers"]),
        file_name="pb_final_survivors.txt",
        mime="text/plain",
    )

if __name__ == "__main__":
    main()
