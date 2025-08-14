import streamlit as st
import pandas as pd
import os

DATA_DIR = "corrected_data"
DATA_FILES = ["iliad_1.tsv", "odyssey_1.tsv", "posthomerica_1.tsv"]

@st.cache_data
def downloadData(data_dir=DATA_DIR, files=DATA_FILES):
    '''load tsv files and return a single dataframe'''
    df = pd.concat(
        pd.read_table(os.path.join(data_dir, filename)) for filename in files
    )

    # add groundtruth col
    df["is_voc"] = (
        (df["eval_spacy"] == "true positive") | 
        (df["eval_spacy"] == "false negative")
    )
    
    return df


@st.cache_data
def getConfusionMatrix(df):
    '''calculate confusion matrix for parser validation'''

    # confusion matrix and accuracy
    eval_spacy = df.groupby("speech_id")["eval_spacy"].value_counts().unstack(fill_value=0)
    eval_spacy["F1"] = (
        2 * eval_spacy["true positive"] / 
        (2 * eval_spacy["true positive"] + eval_spacy["false positive"] + eval_spacy["false negative"])
    ).round(2)

    eval_cltk = df.groupby("speech_id")["eval_CLTK"].value_counts().unstack(fill_value=0)
    eval_cltk["F1"] = (
        2 * eval_cltk["true positive"] / 
        (2 * eval_cltk["true positive"] + eval_cltk["false positive"] + eval_cltk["false negative"])
    ).round(2)

    # summarize by speech
    aggregated = (df
        .groupby("speech_id")
        .agg(
            Work = ("Work", "first"),
            Book = ("book", "first"),
            First_Line = ("line", "first"),
            Last_Line = ("line", "last"),
            Speaker = ("spkr", "first"),
            Addressee = ("addr", "first"),
            Tokens = ("string", "count"),
            Vocatives = ("is_voc", "sum"),
            )
        .join(
            eval_spacy.join(eval_cltk, lsuffix="_spacy", rsuffix="_cltk")
        )
    
    )

    return aggregated


def colorByStatus(token, status):
    '''add span tag highlighting tokens to show parser validation'''
    
    styling = {
        "false negative": "font-weight: bold; color: darkgoldenrod;",
        "true positive": "font-weight: bold; color: darkgreen; background: palegreen;",
        "false positive": "background: lightcoral;",
    }

    if status in styling:
        style = f' style="{styling.get(status)}"' 
    else:
        style = ""
        
    return f"<span{style}>{token}</span>"


def displayTokens(df, speech_id, parser):
    '''display the tokens of one speech as an html table'''

    html = "<table>"
    
    for label, group in df.loc[df["speech_id"]==speech_id].groupby("line"):

        html += "<tr>"
        html += f"<td>{label}</td>"
    
        tokens = []
        for _, row in group.iterrows():
            tokens.append(colorByStatus(token=row["string"], status=row[f"eval_{parser}"]))
        html += f"<td>{' '.join(tokens)}</td>"
    
        html += "</tr>"

    html += "</table>"
    
    return html


def displayLegend(df, speech_id, parser):

    speech_mask = (df["speech_id"] == speech_id)

    html = "<table>"
    
    for status in ["true positive", "false positive", "false negative"]:
        count = sum(speech_mask & (df[f"eval_{parser}"] == status))
        label = colorByStatus(token=status, status=status)
        html += f"<tr><td>{label}</td><td>{count}</td></tr>"
    
    tok_count = sum(speech_mask)
    html += f"<tr><td>total tokens</td><td>{tok_count}</td></tr>"
    
    tp = sum(speech_mask & (df[f"eval_{parser}"] == "true positive"))
    fp = sum(speech_mask & (df[f"eval_{parser}"] == "false positive"))
    fn = sum(speech_mask & (df[f"eval_{parser}"] == "false negative"))
    if tp > 0:
        f1 = round(100 * (2 * tp) / (2 * tp + fp + fn))
        f1 = f"{f1} %"
    else:
        f1 = "N/A"
    html += f"<tr><td>accuracy (f1)</td><td>{f1}</td></tr>"
    
    html += "</table>"

    return html


#
# main
#

all_tokens = downloadData()
aggregated = getConfusionMatrix(all_tokens)

# title
st.title("Vocatives in Homer and Quintus")

# filters
text_values = [s for s in aggregated["Work"].unique()]
filter_text = st.selectbox("Work", ["all"] + text_values)
if filter_text != "all":
    allowed_texts = [filter_text]
else:
    allowed_texts = text_values
mask = aggregated["Work"].isin(allowed_texts)

# display table, get row click
selected = st.dataframe(aggregated.loc[mask],
    column_order = ("Work", "Book", "First_Line", "Speaker", "Addressee", "Tokens", "Vocatives", "F1_spacy", "F1_cltk"),
    column_config = {
        "F1_spacy": st.column_config.NumberColumn(format="percent"),
        "F1_cltk": st.column_config.NumberColumn(format="percent"),
    },
    hide_index=True,
    on_select = "rerun",
    selection_mode = "single-row",
)["selection"]["rows"]

# display tokens
if len(selected) > 0:
    idx = selected[-1]
    speech_id = aggregated.loc[mask].index.values[idx]
    wk = aggregated.loc[speech_id, "Work"]
    bk = aggregated.loc[speech_id, "Book"]
    fl = aggregated.loc[speech_id, "First_Line"]
    ll = aggregated.loc[speech_id, "Last_Line"]
    
    spacy_tab, cltk_tab = st.tabs(["spaCy validation", "CLTK validation"])
    
    with spacy_tab:
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:  
            st.subheader(f"{wk} {bk}.{fl}-{ll}")
            st.markdown(displayTokens(all_tokens, speech_id, "spacy"), unsafe_allow_html=True)
        with col2:
            st.subheader("Legend")
            st.markdown(displayLegend(all_tokens, speech_id, "spacy"), unsafe_allow_html=True)
        
    with cltk_tab:
        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            st.subheader(f"{wk} {bk}.{fl}-{ll}")            
            st.markdown(displayTokens(all_tokens, speech_id, "CLTK"), unsafe_allow_html=True)
        with col2:
            st.subheader("Legend")            
            st.markdown(displayLegend(all_tokens, speech_id, "CLTK"), unsafe_allow_html=True)