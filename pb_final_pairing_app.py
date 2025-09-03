
import streamlit as st
from collections import Counter
from itertools import permutations
import csv, os, re
import pandas as pd

# ----------------------
# Helper: parsing & context
# ----------------------
def normalize_tens(text):
    toks = []
    for line in text.splitlines():
        for token in line.replace(',',' ').split():
            toks.append(token.strip())
    out, bad = [], []
    for tok in toks:
        digs = [c for c in tok if c.isdigit()]
        if len(digs) != 5 or any(c not in '0123456' for c in digs):
            bad.append(tok); continue
    out.append(''.join(sorted(digs)))
    # dedupe preserve order
    seen=set(); out2=[]
    for o in out:
        if o not in seen: out2.append(o); seen.add(o)
    return out2, bad

def normalize_ones(text):
    toks = []
    for line in text.splitlines():
        for token in line.replace(',',' ').split():
            toks.append(token.strip())
    out, bad = [], []
    for tok in toks:
        digs = [c for c in tok if c.isdigit()]
        if len(digs) != 5 or any(c not in '0123456789' for c in digs):
            bad.append(tok); continue
        out.append(''.join(sorted(digs)))
    seen=set(); out2=[]
    for o in out:
        if o not in seen: out2.append(o); seen.add(o)
    return out2, bad

def all_unique_perms(seq):
    # returns set of unique permutations for up to length 5
    return {tuple(p) for p in set(permutations(seq, len(seq)))}

def pair_tens_ones(tens_str, ones_str):
    # returns a set of sorted 5-number tuples (valid 1..69, unique)
    t = [int(c) for c in tens_str]
    o = [int(c) for c in ones_str]
    nums_set = set()
    for p in all_unique_perms(o):
        nums = [10*t[i] + p[i] for i in range(5)]
        if any(n < 1 or n > 69 for n in nums): continue
        if len(set(nums)) != 5: continue
        nums_set.add(tuple(sorted(nums)))
    return nums_set

def multiset_shared(a,b):
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())

def build_ctx(combo_nums, seed_nums, prev_nums):
    final_sum = sum(combo_nums)
    final_even = sum(1 for n in combo_nums if n%2==0)
    final_odd = 5 - final_even
    final_min, final_max = min(combo_nums), max(combo_nums)
    final_range = final_max - final_min
    final_low = sum(1 for n in combo_nums if 1 <= n <= 34)
    final_high = 5 - final_low

    # Derive tens/ones
    combo_tens = [n // 10 for n in combo_nums]
    combo_ones = [n % 10 for n in combo_nums]
    seed_tens = [n // 10 for n in (seed_nums or [])]
    seed_ones = [n % 10 for n in (seed_nums or [])]
    prev_seed_tens = [n // 10 for n in (prev_nums or [])]
    prev_seed_ones = [n % 10 for n in (prev_nums or [])]

    # Ones metrics
    ones_sum = sum(combo_ones)
    ones_even = sum(1 for d in combo_ones if d % 2 == 0)
    ones_odd = 5 - ones_even
    ones_unique = len(set(combo_ones))
    ones_range = max(combo_ones) - min(combo_ones) if combo_ones else 0
    ones_low = sum(1 for d in combo_ones if 0 <= d <= 4)
    ones_high = 5 - ones_low

def multiset_shared(a,b):
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())

    ctx = {
        # Numbers-level
        'combo_numbers': combo_nums,
        'seed_numbers': seed_nums,
        'prev_seed_numbers': prev_nums,
        'final_sum': final_sum,
        'final_even_count': final_even,
        'final_odd_count': final_odd,
        'final_range': final_range,
        'final_min': final_min,
        'final_max': final_max,
        'final_low_count': final_low,
        'final_high_count': final_high,
        # Digits-level
        'combo_tens': combo_tens,
        'combo_ones': combo_ones,
        'seed_tens': seed_tens,
        'seed_ones': seed_ones,
        'prev_seed_tens': prev_seed_tens,
        'prev_seed_ones': prev_seed_ones,
        # Ones metrics
        'ones_sum': ones_sum,
        'ones_even_count': ones_even,
        'ones_odd_count': ones_odd,
        'ones_unique_count': ones_unique,
        'ones_range': ones_range,
        'ones_low_count': ones_low,
        'ones_high_count': ones_high,
        # helpers
        'shared_numbers': multiset_shared,
        'shared_ones': multiset_shared,
        'shared_tens': multiset_shared,
    }
    return ctx

def load_filters(paths):
    filters = []
    if not isinstance(paths, (list, tuple)): paths = [paths]
    for path in paths:
        if not path or not os.path.exists(path): continue
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row = {k.lower(): v for k, v in raw.items()}
                row['id'] = row.get('id', row.get('fid', '')).strip()
                for key in ('name','applicable_if','expression'):
                    if key in row and isinstance(row[key], str):
                        row[key] = row[key].strip().strip('"').strip("'")
                row['expression'] = (row.get('expression') or 'False').replace('!==','!=')
                row['expr_str'] = row['expression']
                applicable = row.get('applicable_if') or 'True'
                expr = row.get('expression') or 'False'
                try:
                    row['applicable_code'] = compile(applicable,'<applicable>','eval')
                    row['expr_code'] = compile(expr,'<expr>','eval')
                except SyntaxError as e:
                    st.sidebar.warning(f"Syntax error in filter {row.get('id','?')}: {e}")
                    continue
                row['enabled_default'] = (row.get('enabled','').lower() == 'true')
                filters.append(row)
    return filters

def main():
    st.sidebar.header("🔗 Final Pairing — Tens × Ones → 5 numbers")

    # Default filters
    default_filters_path = "pb_final_foundational_filters.csv"
    default_extra_path = "pb_final_percentile_filters.csv"
    default_user_adapted = "pb_final_filters_adapted.csv"
    use_default = st.sidebar.checkbox("Use default final filters", value=True)
    uploaded_filters = st.sidebar.file_uploader("Upload additional filter CSV (optional)", type=["csv"])

    filter_paths = []
    if use_default and os.path.exists(default_filters_path): filter_paths.append(default_filters_path)
    if os.path.exists(default_extra_path): filter_paths.append(default_extra_path)
    if os.path.exists(default_user_adapted): filter_paths.append(default_user_adapted)
    if uploaded_filters is not None:
        upath = "user_final_filters.csv"
        with open(upath, "wb") as f: f.write(uploaded_filters.getbuffer())
        filter_paths.append(upath)

    filters = load_filters(filter_paths)

    # Tens & Ones inputs
    t_text = st.sidebar.text_area("Paste tens survivors (one per line, e.g., 11566)", height=150)
    o_text = st.sidebar.text_area("Paste ones survivors (one per line, e.g., 57999)", height=150)

    # Seed winner inputs (for carryover)
    seed_text = st.sidebar.text_input("Seed winner (last draw numbers, 5 nums 1–69, e.g., 01-16-21-47-60)", value="").strip()
    prev_text = st.sidebar.text_input("Prev winner (optional)", value="").strip()

    # Normalize seed numbers
def parse_numbers(s):
        nums = [int(x) for x in re.findall(r"\d+", s)]
        nums = [n for n in nums if 1 <= n <= 69]
        if len(nums) == 5:
            return sorted(nums)
            return []
seed_numbers = parse_numbers(seed_text)
prev_numbers = parse_numbers(prev_text)

    # --- History seeds (auto from pwrbll.txt if available)
history_seeds = []
history_labels = []
try:
        with open('pwrbll.txt','r',encoding='utf-8') as _f:
            _lines = _f.readlines()
        _rows = []
        import re
        _pat = re.compile(r"^(.*?),\s+Powerball:\s+(\d+)")
        for _line in _lines:
            _m = _pat.search(_line)
            if not _m: continue
            _date_numbers, _pb = _m.groups()
            _parts = _date_numbers.split('\\t')
            _date = _parts[0].strip()
            _nums = [int(n) for n in re.findall(r"\\d{2}", _parts[-1])]
            if len(_nums)==5:
                _rows.append((_date, sorted(_nums)))
        _rows = _rows[::-1]  # chronological
        history_seeds = [r[1] for r in _rows]
        history_labels = [f"{i+1:02d} | {r[0]} — " + "-".join(f"{x:02d}" for x in r[1]) for i,r in enumerate(_rows)]
except Exception as _e:
        pass
seed_source = st.sidebar.radio('Seed source', ['History', 'Manual'], index=0 if history_seeds else 1)
use_hist = (seed_source == 'History')
seed_index = 0
if use_hist and history_seeds:
        seed_index = st.sidebar.slider("Select seed (0 = latest)", min_value=0, max_value=len(history_seeds)-1, value=0, step=1)
        seed_numbers = history_seeds[-(seed_index+1)]  # 0=latest
        st.sidebar.caption(f"Seed set to: " + "-".join(f"{x:02d}" for x in seed_numbers))
        # optionally prev
        if seed_index+1 < len(history_seeds):
            prev_numbers = history_seeds[-(seed_index+2)]
        else:
            # fall back to manual parse below
            pass
        if seed_text and not seed_numbers:
           st.sidebar.error("Seed winner must have exactly 5 integers in 1..69.")
        if prev_text and not prev_numbers:
           st.sidebar.error("Prev winner must have exactly 5 integers in 1..69.")

    tens_list, bad_t = normalize_tens(t_text)
    ones_list, bad_o = normalize_ones(o_text)
    if bad_t: st.sidebar.warning(f"Ignored invalid tens entries: {', '.join(bad_t[:5])}" + (" ..." if len(bad_t)>5 else ""))
    if bad_o: st.sidebar.warning(f"Ignored invalid ones entries: {', '.join(bad_o[:5])}" + (" ..." if len(bad_o)>5 else ""))

    st.sidebar.markdown(f"**Tens combos:** {len(tens_list)}")
    st.sidebar.markdown(f"**Ones combos:** {len(ones_list)}")

    # Track/Test combos
    st.sidebar.markdown("---")
    track_text = st.sidebar.text_area("Track/Test 5-number sets (e.g., 01-16-21-47-60)", height=120)
    preserve_tracked = st.sidebar.checkbox("Preserve tracked combos during filtering", value=True)
    inject_tracked = st.sidebar.checkbox("Inject tracked combos even if not generated", value=False)

    def normalize_final(text):
        toks=[]
        for line in text.splitlines():
            toks += [t.strip() for t in re.split(r"[,\s]+", line) if t.strip()]
        out,bad=[],[]
        for tok in toks:
            nums = sorted(int(x) for x in re.findall(r"\d+", tok))
            if len(nums)!=5 or any(n<1 or n>69 for n in nums):
                bad.append(tok); continue
            out.append(tuple(nums))
        seen=set(); out2=[]
        for o in out:
            if o not in seen: out2.append(o); seen.add(o)
        return out2,bad

    tracked, bad_finals = normalize_final(track_text)
    if bad_finals:
        st.sidebar.warning(f"Ignored invalid tracked sets: {', '.join(bad_finals[:3])}" + (" ..." if len(bad_finals)>3 else ""))
    tracked_set = set(tracked)

    # Combine
    st.header("🧮 Combine Tens × Ones → Candidates")
    candidates = set()
    for t in tens_list:
        for o in ones_list:
            for combo in pair_tens_ones(t, o):
                candidates.add(combo)
    st.write(f"Generated candidates (pre-filter): {len(candidates)}")

    # Inject tracked if requested
    if inject_tracked:
        for c in tracked:
            if c not in candidates: candidates.add(c)

    # Build audit map
    audit = { c: {"combo": c, "generated": (c in candidates), "preserved": bool(preserve_tracked),
                  "injected": (c in tracked and c not in candidates),
                  "eliminated": False, "eliminated_by": None, "eliminated_name": None, "eliminated_order": None,
                  "would_eliminate_by": None, "would_eliminate_name": None, "would_eliminate_order": None}
              for c in tracked }

    # Initial elimination counts
    init_counts = {flt['id']: 0 for flt in filters}
    for flt in filters:
        ic = 0
        for combo in candidates:
            ctx = build_ctx(list(combo), seed_numbers, prev_numbers)
            try:
                if eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx): ic += 1
            except Exception: pass
        init_counts[flt['id']] = ic

    select_all = st.sidebar.checkbox("Select/Deselect All Filters", value=False)
    hide_zero = st.sidebar.checkbox("Hide filters with 0 initial eliminations", value=True)

    sorted_filters = sorted(filters, key=lambda flt: (init_counts[flt['id']] == 0, -init_counts[flt['id']]))
    display_filters = [f for f in sorted_filters if init_counts[f['id']] > 0] if hide_zero else sorted_filters

    # Apply
    pool = list(sorted(candidates))
    st.header("🔧 Manual Filters (final-stage)")
    order_index = 0; dynamic_counts = {}
    for flt in display_filters:
        order_index += 1
        key = f"filter_{flt['id']}"
        default_checked = select_all and flt['enabled_default']
        checked = st.checkbox(f"{flt['id']}: {flt['name']} — init cuts {init_counts[flt['id']]}", key=key, value=default_checked)
        if checked:
            survivors = []
            dc = 0
            for combo in pool:
                ctx = build_ctx(list(combo), seed_numbers, prev_numbers)
                eliminate = False
                try:
                    eliminate = eval(flt['applicable_code'], ctx, ctx) and eval(flt['expr_code'], ctx, ctx)
                except Exception: eliminate = False
                is_tracked = combo in tracked_set
                if eliminate:
                    if is_tracked and preserve_tracked:
                        info = audit.get(combo, None)
                        if info and info.get("would_eliminate_by") is None:
                            info["would_eliminate_by"] = flt['id']
                            info["would_eliminate_name"] = flt.get('name','')
                            info["would_eliminate_order"] = order_index
                        survivors.append(combo); continue
                    dc += 1
                    if is_tracked and not audit[combo]["eliminated"]:
                        audit[combo]["eliminated"] = True
                        audit[combo]["eliminated_by"] = flt['id']
                        audit[combo]["eliminated_name"] = flt.get('name','')
                        audit[combo]["eliminated_order"] = order_index
                else:
                    survivors.append(combo)
            pool = survivors
            dynamic_counts[flt['id']] = dc

    st.subheader(f"Remaining after manual filters: {len(pool)}")
    survivors_set = set(pool)

    # Audit
    if tracked:
        st.markdown("### 🔎 Tracked/Preserved Sets — Audit")
        rows = []
        for c in tracked:
            info = audit.get(c, {})
            rows.append({
                "combo": "-".join(f"{x:02d}" for x in c),
                "generated": info.get("generated", False),
                "survived": (c in survivors_set),
                "eliminated": info.get("eliminated", False),
                "eliminated_by": info.get("eliminated_by"),
                "eliminated_order": info.get("eliminated_order"),
                "eliminated_name": info.get("eliminated_name"),
                "would_eliminate_by": info.get("would_eliminate_by"),
                "would_eliminate_order": info.get("would_eliminate_order"),
                "would_eliminate_name": info.get("would_eliminate_name"),
                "injected": info.get("injected", False),
                "preserved": info.get("preserved", False),
            })
        df_audit = pd.DataFrame(rows)
        st.dataframe(df_audit, use_container_width=True)
        st.download_button("Download audit (CSV)", df_audit.to_csv(index=False), file_name="pb_final_audit_tracked.csv", mime="text/csv")

    # Survivors display
    st.markdown("### ✅ Final Survivors")
    with st.expander("Show remaining 5-number sets"):
        tracked_survivors = [c for c in pool if c in tracked_set]
        if tracked_survivors:
            st.write("**Tracked survivors:**")
            for c in tracked_survivors:
                info = audit.get(c, {})
                label = "-".join(f"{x:02d}" for x in c)
                if info and info.get("would_eliminate_by"):
                    st.write(f"{label} — ⚠ would be eliminated by {info['would_eliminate_by']} at step {info.get('would_eliminate_order')} ({info.get('would_eliminate_name')}) — preserved")
                else:
                    st.write(label)
            st.write("---")
        for c in pool:
            if c not in tracked_set:
                st.write("-".join(f"{x:02d}" for x in c))

    # Downloads
import pandas as pd

def fmt_combo(c):
    return "-".join(f"{int(x):02d}" for x in sorted(c))

df_out = pd.DataFrame({"numbers": [fmt_combo(c) for c in pool]})

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

