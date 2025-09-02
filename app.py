import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import math
import unicodedata
import altair as alt
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Respostas", layout="wide", page_icon="📊")

# -------------------- SETTINGS --------------------
CSV_PATH = "respostas.csv" # Ensure this file exists for the app to run without errors

HEADER_IMAGE = "header_viseu.png" # Assuming you have this image for the header

ALIASES = {
    "cultura": "Cultura e Património",
    "patrimonio": "Cultura e Património",
    "educacao": "Educação, Desporto e Juventude",
    "desporto": "Educação, Desporto e Juventude",
    "juventude": "Educação, Desporto e Juventude",
    "criancas": "Educação, Desporto e Juventude",
    "obras": "Obras, Infraestruturas e Serviços Municipais",
    "infraestruturas": "Obras, Infraestruturas e Serviços Municipais",
    "servicos municipais": "Obras, Infraestruturas e Serviços Municipais",
    "territorio": "Território, Urbanismo, Habitação",
    "urbanismo": "Território, Urbanismo, Habitação",
    "habitacao": "Território, Urbanismo, Habitação",
    "governanca": "Governança e Participação Cívica",
    "participacao civica": "Governança e Participação Cívica",
    "empresas, investimento, comercio": "Empresas, Investimento, Comércio e Turismo",
    "turismo": "Empresas, Investimento, Comércio e Turismo",
    "transicao digital": "Transição digital/energética/ambiental",
    "energetica": "Transição digital/energética/ambiental",
    "ambiental": "Transição digital/energética/ambiental",
    "agua": "Água",
}

MAIN_AREAS = [
    "Educação, Desporto e Juventude",
    "Transição digital/energética/ambiental",
    "Cultura e Património",
    "Obras, Infraestruturas e Serviços Municipais",
    "Território, Urbanismo, Habitação",
    "Freguesias",
    "Governança e Participação Cívica",
    "Mobilidade",
    "Empresas, Investimento, Comércio e Turismo",
    "Água",
    "Outros",
]

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found. Please ensure the CSV file is in the correct directory.")
        # Create a dummy DataFrame to prevent further errors
        return pd.DataFrame(columns=['Idade', 'Freguesia', 'Área', 'Género', 'Resposta'])

df = load_csv(CSV_PATH)
# If df is empty (due to FileNotFoundError), initialize work with appropriate columns
if df.empty:
    work = pd.DataFrame(columns=['Idade', 'Freguesia', 'Área', 'Género', 'Resposta'])
else:
    work = df.copy()

# -------------------- UTILITIES --------------------
def find_column(df, patterns):
    for c in df.columns:
        lc = str(c).lower()
        for p in patterns:
            if p in lc:
                return c
    return None

def normalize_str(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s

def normalize_age(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip()
    m = re.search(r"(\d{1,3})", v)
    if m:
        age = int(m.group(1))
        if 18 <= age <= 24: return "18-24"
        elif 25 <= age <= 34: return "25-34"
        elif 35 <= age <= 44: return "35-44"
        elif 45 <= age <= 54: return "45-54"
        elif 55 <= age <= 64: return "55-64"
        elif age >= 65: return "65+"
    if re.search(r"18-24", v): return "18-24"
    if re.search(r"25-34", v): return "25-34"
    if re.search(r"35-44", v): return "35-44"
    if re.search(r"45-54", v): return "45-54"
    if re.search(r"55-64", v): return "55-64"
    if re.search(r"65|\+|mais", v, re.IGNORECASE): return "65+"
    return np.nan

def split_multiselect(val):
    """
    Splits a string containing multiple selections, handling commas within category names
    and using a predefined list of main areas for matching.
    """
    if pd.isna(val) or not isinstance(val, str):
        return []

    val_normalized = normalize_str(val)
    found_areas = []

    # Try to match full main areas first
    for area in MAIN_AREAS:
        if normalize_str(area) in val_normalized:
            found_areas.append(area)
            val_normalized = val_normalized.replace(normalize_str(area), "").strip()

    # If there are still parts left, try splitting by common delimiters
    # and then mapping to aliases or main areas
    remaining_parts = re.split(r";|/|\||\+|\s+e\s+|,", val) # Added comma to split delimiters
    for part in remaining_parts:
        stripped_part = part.strip()
        if not stripped_part:
            continue

        normalized_stripped_part = normalize_str(stripped_part)

        # Check if the part (or its alias) is in MAIN_AREAS
        mapped_area = None
        if normalized_stripped_part in AREA_MAP:
            mapped_area = AREA_MAP[normalized_stripped_part]
        elif normalized_stripped_part in ALIASES:
            mapped_area = ALIASES[normalized_stripped_part]
        elif normalized_stripped_part.startswith("outra"):
            mapped_area = "Outros"
        
        if mapped_area and mapped_area not in found_areas:
            found_areas.append(mapped_area)
        elif not mapped_area and stripped_part not in found_areas and stripped_part != '': # If not mapped, add as is, might be a custom entry
             # Only add to found areas if it isn't already there from the full main areas check
            if stripped_part not in [a for a in found_areas]:
                found_areas.append(stripped_part)

    # Filter out empty strings and ensure uniqueness, re-map "Outros" if it's the only one and not explicitly selected
    final_areas = []
    for area in found_areas:
        mapped = area # Default to the area itself
        norm_area = normalize_str(area)
        if norm_area in AREA_MAP:
            mapped = AREA_MAP[norm_area]
        elif norm_area in ALIASES:
            mapped = ALIASES[norm_area]
        elif norm_area.startswith("outra"):
            mapped = "Outros"
        
        if mapped not in final_areas and mapped != '':
            final_areas.append(mapped)

    return final_areas if final_areas else []


def tokenize_pt(text):
    text = str(text).lower()
    text = re.sub(r"[^\wáéíóúâêôãõàç]+"," ", text, flags=re.UNICODE)
    tokens = text.split()
    stop = set("""a o os as de da do das dos e em para por com sem um uma umas uns ao aos à às no na nos nas que se ou ser ter tem têm foi era foram mais menos muito muita muitos muitas pouco poucos poucas já não sim como quando onde porque qual quais sobre entre até desde contra cada pelo pela pelos pelas este esta estes estas esse essa esses essas aquele aquela aqueles aquela isto isso aquilo lhe lhes eu tu ele ela nós voces vocês eles elas meu minha meus minhas teu tua teus tuas seu sua seus suas nosso nossa nossos nossas vosso vossa vossos vossas dele dela deles delas aqui ali lá também só tão quanto tudo todo toda todos todas""".split())
    return [t for t in tokens if len(t)>2 and t not in stop and not t.isdigit()]

def top_keywords_per_area(grouped_text_df, k=10):
    if grouped_text_df.empty: return pd.DataFrame(columns=["Área","Palavra","Score","Frequência"])
    docs = {}
    for _, row in grouped_text_df.iterrows():
        area = str(row["Área"])
        texts = row["Respostas Abertas"]
        toks = []
        for t in texts: toks.extend(tokenize_pt(t))
        docs[area] = toks
    vocab = set().union(*docs.values())
    dfreq = Counter({term: sum(1 for a in docs if term in set(docs[a])) for term in vocab})
    N = max(len(docs),1)
    rows = []
    for area,toks in docs.items():
        tf = Counter(toks)
        L = max(len(toks),1)
        for term,f in tf.items():
            idf = math.log((N+1)/(dfreq[term]+1))+1
            score = (f/L)*idf
            rows.append((area,term,score,f))
    kw = pd.DataFrame(rows, columns=["Área","Palavra","Score","Frequência"])
    kw_top = kw.sort_values(["Área","Score"],ascending=[True,False]).groupby("Área").head(k).reset_index(drop=True)
    return kw_top

def plot_horizontal(dist_df, category, value, color="#2ca02c", chart_width=600, bar_height=30, title="", sort_by_value=False):
    if dist_df.empty or dist_df[value].sum() == 0:
        st.write(f"Não há dados para exibir em '{title}'.")
        return

    total_responses = dist_df[value].sum()
    if total_responses > 0:
        dist_df['percentage'] = (dist_df[value] / total_responses * 100).round(1)
        dist_df['label'] = dist_df.apply(lambda row: f"{row['percentage']:.1f}%", axis=1)
    else:
        dist_df['percentage'] = 0.0
        dist_df['label'] = "0.0%"

    if sort_by_value:
        dist_df = dist_df.sort_values(by=value, ascending=False)

    # Calculate chart height dynamically based on number of categories and bar_height
    # Add some padding for title and axis labels
    chart_height = max(bar_height * len(dist_df), 150) + 50 # Increased min height and added padding

    base = alt.Chart(dist_df).properties(title=title, height=chart_height, width=chart_width)

    bars = base.mark_bar().encode(
        x=alt.X('percentage', title="Percentagem de Respostas", axis=alt.Axis(format='.1f')),
        y=alt.Y(category,
                sort=alt.EncodingSortField(field=value, op="sum", order='descending' if sort_by_value else 'ascending'), # Sort by value dynamically
                title="",
                axis=alt.Axis(labelAngle=0, labelAlign='right', labelOverlap="greedy", labelLimit=300)), # Prevent label overlap, increase limit
        color=alt.value(color),
        tooltip=[category, alt.Tooltip(value, title="Respostas"), alt.Tooltip('percentage', title="Percentagem", format=".1f")]
    )

    text = bars.mark_text(align='left', baseline='middle', dx=3, color='black').encode(
        text='label:N'
    )

    chart = (bars + text).configure_title(fontSize=16, anchor='middle') \
                        .configure_axisY(labelFontSize=12) # Set font size for Y-axis labels
    st.altair_chart(chart, use_container_width=True)

def plot_donut(dist_df, category, value, title="", color_scheme='category20', outerRadius=120, innerRadius=80):
    if dist_df.empty or dist_df[value].sum() == 0:
        st.write(f"Não há dados para exibir em '{title}'.")
        return
    dist_df = dist_df.copy()
    total = dist_df[value].sum()
    if total > 0:
        dist_df['percentage'] = dist_df[value] / total
        dist_df['label'] = dist_df.apply(lambda row: f"{row['percentage']*100:.1f}%", axis=1) # Only percentage
    else:
        dist_df['percentage'] = 0.0
        dist_df['label'] = "0.0%"

    base = alt.Chart(dist_df).encode(
        theta=alt.Theta(field=value, type='quantitative'),
        color=alt.Color(field=category, type='nominal', scale=alt.Scale(scheme=color_scheme)),
        tooltip=[category, alt.Tooltip(value, title="Respostas"), alt.Tooltip('percentage', title="Percentagem", format=".1%")]
    ).properties(title=title)
    pie = base.mark_arc(innerRadius=innerRadius, outerRadius=outerRadius, stroke="#fff", strokeWidth=1) # Added stroke for better separation
    text = base.mark_text(radius=outerRadius + 20, size=12, fill='black').encode( # Position labels outside the donut
        text='label:N', # Show only percentage label
        order=alt.Order(field=value, sort='descending')
    )
    chart = (pie + text).properties(width=400, height=300) # Increased height a bit
    chart = chart.configure_title(fontSize=16, anchor='middle')
    st.altair_chart(chart, use_container_width=True)


# -------------------- DETECT COLUMNS --------------------
# Ensure work is not empty before attempting to find columns
if not work.empty:
    col_age = "Idade" 
    col_freg = find_column(work, ["freguesia","bairro","localidade"])
    col_area = find_column(work, ["área","area","áreas"])
    col_gender = find_column(work, ["genero", "género"])

    col_text = None
    best_avg = 0
    for c in work.select_dtypes(include="object").columns:
        if c in (col_age,col_freg,col_area,col_gender): continue
        s = work[c].dropna().astype(str)
        if s.empty: continue
        avg_len = s.map(len).mean()
        uniq_ratio = s.nunique()/len(s)
        if avg_len>best_avg and avg_len>20 and uniq_ratio>0.2:
            best_avg = avg_len
            col_text = c
else: # If work is empty, set columns to None to prevent errors
    col_age = None
    col_freg = None
    col_area = None
    col_gender = None
    col_text = None

# -------------------- PROCESS DATA --------------------
if col_age and not work.empty:
    work["faixa_etaria"] = work[col_age].map(normalize_age)
else:
    work["faixa_etaria"] = np.nan

AREA_MAP = {normalize_str(a): a for a in MAIN_AREAS}

if col_area and not work.empty:
    tmp = work[[col_area]].copy()
    tmp["row_id"] = tmp.index
    tmp["_area_list"] = tmp[col_area].map(split_multiselect)
    areas_long = tmp.explode("_area_list").rename(columns={"_area_list":"Área"}).dropna(subset=["Área"])

    def map_area(a):
        key = normalize_str(str(a))
        if key in AREA_MAP:
            return AREA_MAP[key]
        if key in ALIASES:
            return ALIASES[key]
        if key.startswith("outra"):
            return "Outros"
        return "Outros"

    areas_long["Área"] = areas_long["Área"].map(map_area)
    
    # After mapping, remove duplicates if a row maps to the same area multiple times
    # This ensures each original response contributes to each unique area it mentioned only once
    areas_long = areas_long.drop_duplicates(subset=['row_id', 'Área'])

else:
    areas_long = pd.DataFrame(columns=["row_id","Área"])

age_order = ["18-24","25-34","35-44","45-54","55-64","65+"]

dist_age = work["faixa_etaria"].value_counts().reindex(age_order, fill_value=0).rename_axis("Faixa Etária").reset_index(name="Respostas") if "faixa_etaria" in work.columns and not work["faixa_etaria"].empty else pd.DataFrame(columns=["Faixa Etária", "Respostas"])
dist_freg = work[col_freg].astype(str).value_counts().rename_axis("Freguesia").reset_index(name="Respostas") if col_freg and not work[col_freg].empty else pd.DataFrame(columns=["Freguesia", "Respostas"])
dist_area = areas_long["Área"].value_counts().rename_axis("Área").reset_index(name="Respostas") if not areas_long.empty else pd.DataFrame(columns=["Área", "Respostas"])
dist_gender = work[col_gender].astype(str).value_counts().rename_axis("Género").reset_index(name="Respostas") if col_gender and not work[col_gender].empty else pd.DataFrame(columns=["Género", "Respostas"])

if col_text and not areas_long.empty:
    text_df = work[[col_text]].copy(); text_df["row_id"] = text_df.index
    grouped_text = areas_long.merge(text_df, on="row_id").groupby("Área")[col_text].apply(lambda s: [str(x).strip() for x in s.dropna() if str(x).strip()]).reset_index(name="Respostas Abertas")
else:
    grouped_text = pd.DataFrame(columns=["Área","Respostas Abertas"])

kw_top = top_keywords_per_area(grouped_text, k=12)

# -------------------- DASHBOARD --------------------
col1, col2, col3 = st.columns([1,1,1])  # Colunas para centralizar
with col2:
    st.image(HEADER_IMAGE)

# Centralized buttons
if "tab" not in st.session_state:
    st.session_state.tab = "gerais"

with st.container():
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])

    with button_col2:
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("📊 Gerais", key="btn_gerais", use_container_width=True):
                st.session_state.tab = "gerais"
        with btn2:
            if st.button("🔎 Filtros", key="btn_filtros", use_container_width=True):
                st.session_state.tab = "filtros"

st.markdown("---")

# Show content
if st.session_state.tab=="gerais":
    st.subheader("Distribuição por Idade e Género")
    col_age_d, col_gender_d = st.columns([1,1])
    with col_age_d:
        plot_donut(dist_age, "Faixa Etária", "Respostas", title="", color_scheme='reds', outerRadius=100, innerRadius=60)
    with col_gender_d:
        plot_donut(dist_gender, "Género", "Respostas", title="", color_scheme='greens', outerRadius=100, innerRadius=60)


    st.subheader("Distribuição por Freguesia e Área de Resposta")
    col_freg_b, col_area_b = st.columns([1,1])
    with col_freg_b:
        # Adjusted chart_width for better fit within a column
        plot_horizontal(dist_freg, "Freguesia", "Respostas", title="", color="#2ca02c", chart_width=600, bar_height=25, sort_by_value=True)
    with col_area_b:
        # Adjusted chart_width for better fit within a column
        plot_horizontal(dist_area, "Área", "Respostas", title="", color="#d62728", chart_width=600, bar_height=25, sort_by_value=True) # Sort by value for areas too


elif st.session_state.tab=="filtros":
    
    st.subheader(f"Ideias, Contributos e Críticas")

    age_options = ["Todas"] + age_order
    freg_options = ["Todas"] + dist_freg["Freguesia"].tolist() if not dist_freg.empty else ["Todas"]
    area_options = ["Todas"] + [a for a in MAIN_AREAS if a in dist_area["Área"].tolist()] if not dist_area.empty else ["Todas"]
    gender_options = ["Todas"] + dist_gender["Género"].tolist() if not dist_gender.empty else ["Todas"]

    f1,f2,f3,f4 = st.columns(4)
    with f1: sel_age = st.selectbox("Faixa Etária", age_options)
    with f2: sel_freg = st.selectbox("Freguesia", freg_options)
    with f3: sel_area = st.selectbox("Área", area_options)
    with f4: sel_gender = st.selectbox("Género", gender_options)

    filtered_idx = set(work.index)
    if sel_age != "Todas": filtered_idx &= set(work[work["faixa_etaria"]==sel_age].index)
    if sel_freg != "Todas": filtered_idx &= set(work[work[col_freg]==sel_freg].index if col_freg else [])
    if sel_area != "Todas": filtered_idx &= set(areas_long[areas_long["Área"]==sel_area]["row_id"].unique().tolist())
    if sel_gender != "Todas": filtered_idx &= set(work[work[col_gender]==sel_gender].index if col_gender else [])

    work_filtered = work.loc[sorted(list(filtered_idx))] if filtered_idx and not work.empty else pd.DataFrame(columns=work.columns)

    st.markdown("---")

    if col_text and not work_filtered.empty:
        
        # ---------------- Top 5 Words ----------------
        filtered_texts = work_filtered[col_text].dropna().astype(str).tolist()
        all_tokens = []
        for t in filtered_texts:
            all_tokens.extend(tokenize_pt(t))
        if all_tokens:
            counter = Counter(all_tokens)
            top5 = dict(counter.most_common(3))
            keywords_list = ", ".join(top5.keys())

        # ---------------- Raw Responses ----------------
            if keywords_list:
                st.write(f"**Palavras-chave:** *{keywords_list}*")  
            st.text("—"*30)
            for row in filtered_texts:
                st.text(row)          # raw text
                st.text("—" )    # visual separator

        filtered_text_df = work_filtered[[col_text]].copy()
        if not areas_long.empty:
            filtered_text_df = filtered_text_df.merge(
                areas_long[["row_id","Área"]], left_index=True, right_on="row_id", how="left"
            ).drop(columns=["row_id"])
            filtered_text_df["Área"] = sel_area if sel_area!="Todas" else filtered_text_df["Área"].fillna("(Não Especificado)")
        else:
            filtered_text_df["Área"] = "Todas"

        grouped_filtered_text = filtered_text_df.groupby("Área")[col_text].apply(
            lambda s: [str(x).strip() for x in s.dropna() if str(x).strip()]
        ).reset_index(name="Respostas Abertas")

    else:
        st.write("Não há respostas abertas para os filtros selecionados.")