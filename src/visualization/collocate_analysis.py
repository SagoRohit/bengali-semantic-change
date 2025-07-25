import streamlit as st
import json
import pandas as pd
import altair as alt

# --- Settings
TOP_K = 10

# --- Load data (adjust path as needed)
@st.cache_data
def load_neighbors():
    with open("results/semantic_neighbors_drift.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_freqs():
    # Load frequency file per era, e.g., {"word": {"era1": freq1, "era2": freq2, ...}}
    with open("results/freqs_by_era.json", "r", encoding="utf-8") as f:
        return json.load(f)

neighbors = load_neighbors()
freqs = load_freqs()  # Dict: word → {era: count}

eras = list(next(iter(neighbors.values())).keys())
eras.sort()

# --- User input
word = st.text_input("Enter a target word:", "")
smoothing = st.slider("Smoothing window (eras)", 0, min(3, len(eras)//2), 0)

if word in neighbors:
    st.header(f"Analysis for '{word}'")

    # --- 1. Frequency Line Chart
    if word in freqs:
        freq_vals = [freqs[word].get(era, 0) for era in eras]
        # Smoothing (moving average)
        smoothed = []
        for i in range(len(freq_vals)):
            left = max(0, i - smoothing)
            right = min(len(freq_vals), i + smoothing + 1)
            smoothed_val = sum(freq_vals[left:right]) / (right - left)
            smoothed.append(smoothed_val)
        freq_data = pd.DataFrame({
            "Era": eras,
            "Frequency": smoothed
        }).sort_values("Era")
        st.subheader("Frequency Trend")
        st.altair_chart(
            alt.Chart(freq_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X('Era', sort=eras),
                    y='Frequency',
                    tooltip=['Era', 'Frequency']
                ).properties(width=600, height=300)
        )
    else:
        st.info("No frequency data found for this word.")


    # --- 2. Collocate Bar Charts by Era
    st.subheader("Top Collocates by Era")
    collocates_df = []
    for era in eras:
        era_colloc = neighbors[word].get(era, [])
        for coll, score in era_colloc:
            collocates_df.append({"Era": era, "Collocate": coll, "PMI": score})

    collocates_df = pd.DataFrame(collocates_df)
    for era in eras:
        era_df = collocates_df[collocates_df["Era"] == era]
        st.markdown(f"**{era}:**")
        st.altair_chart(
            alt.Chart(era_df).mark_bar().encode(
                x=alt.X('Collocate', sort='-y'),
                y='PMI',
                tooltip=['Collocate', 'PMI']
            ).properties(width=600, height=300)
        )

    # --- 3. Comparative Collocate Table (Highlight New/Stable/Vanished)
    st.subheader("Collocate Change Over Time")

    def get_coll_set(era):
        return set(coll for coll, score in neighbors[word].get(era, []))

    for idx, era in enumerate(eras):
        curr_set = get_coll_set(era)
        prev_set = get_coll_set(eras[idx - 1]) if idx > 0 else set()
        new = curr_set - prev_set
        vanished = prev_set - curr_set
        stable = curr_set & prev_set
        st.markdown(f"**{era}:**")
        st.markdown(
            f"- 🟢 **New Collocates:** {', '.join(new) if new else '-'}"
        )
        st.markdown(
            f"- 🔴 **Vanished Collocates:** {', '.join(vanished) if vanished else '-'}"
        )
        st.markdown(
            f"- 🔵 **Stable Collocates:** {', '.join(stable) if stable else '-'}"
        )
else:
    st.info("Enter a target word present in your dataset.")
