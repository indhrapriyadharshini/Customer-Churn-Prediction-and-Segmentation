
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #111a31 100%);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #10182b;
    border-right: 1px solid #293653;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
p, label, [data-testid="stCaptionContainer"] {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background: #18243c;
    border: 1px solid #30415f;
    padding: 18px;
    border-radius: 14px;
}
div[data-testid="stMetricLabel"] {
    color: #a5b4fc !important;
}
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}
div.stButton > button {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 9px 18px;
}
div.stButton > button:hover {
    background: #6d28d9;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("# 📊 ChurnIQ")
st.markdown("### Customer Churn Prediction & Retention Analytics")
st.caption("Explore customer behaviour, identify churn patterns, and predict churn risk.")

# ---------------- LOAD DATA ----------------
DEFAULT_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

@st.cache_data
def read_csv(file):
    return pd.read_csv(file)

df = None
csv_path = Path(DEFAULT_FILE)

if csv_path.exists():
    df = read_csv(str(csv_path))
else:
    # Automatically try another CSV in the project folder
    csv_files = [
        p for p in Path(".").glob("*.csv")
        if p.name != DEFAULT_FILE
    ]
    if csv_files:
        df = read_csv(str(csv_files[0]))

if df is None:
    st.warning("Customer churn CSV file not found.")
    uploaded = st.file_uploader("Upload your customer churn CSV file", type=["csv"])
    if uploaded is not None:
        df = read_csv(uploaded)
    else:
        st.info(
            "Place your Telco churn CSV file in the same folder as app.py, "
            "or upload it above."
        )
        st.stop()

# ---------------- DATA CLEANING ----------------
df.columns = df.columns.str.strip()

required = ["Churn"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("This dataset needs a 'Churn' column. Please upload the Telco Customer Churn dataset.")
    st.stop()

if "customerID" not in df.columns:
    df["customerID"] = ["Customer-" + str(i + 1) for i in range(len(df))]

for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["Churn"] = df["Churn"].astype(str).str.strip()
df = df[df["Churn"].isin(["Yes", "No"])].copy()

if df.empty:
    st.error("No valid Yes/No values were found in the Churn column.")
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.caption("Filter the customer data")

filtered = df.copy()

filter_columns = [
    ("Contract", "Contract Type"),
    ("InternetService", "Internet Service"),
    ("PaymentMethod", "Payment Method"),
    ("gender", "Gender"),
    ("Churn", "Churn Status")
]

for col, label in filter_columns:
    if col in filtered.columns:
        values = sorted(filtered[col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(
            label,
            options=values,
            default=values,
            key="filter_" + col
        )
        if selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]
        else:
            filtered = filtered.iloc[0:0]

if "tenure" in filtered.columns and not filtered.empty:
    valid_tenure = filtered["tenure"].dropna()
    if not valid_tenure.empty:
        min_tenure = int(valid_tenure.min())
        max_tenure = int(valid_tenure.max())
        if min_tenure < max_tenure:
            tenure_range = st.sidebar.slider(
                "Customer Tenure (months)",
                min_value=min_tenure,
                max_value=max_tenure,
                value=(min_tenure, max_tenure)
            )
            filtered = filtered[
                filtered["tenure"].between(tenure_range[0], tenure_range[1])
            ]

if st.sidebar.button("🔄 Reset chart selection"):
    st.session_state.pop("selected_contract", None)
    st.session_state.pop("selected_service", None)
    st.rerun()

# ---------------- KPIs ----------------
total_customers = len(filtered)
churned = int((filtered["Churn"] == "Yes").sum())
churn_rate = (churned / total_customers * 100) if total_customers else 0

if "MonthlyCharges" in filtered.columns:
    avg_monthly = filtered["MonthlyCharges"].mean()
else:
    avg_monthly = 0

if "tenure" in filtered.columns:
    avg_tenure = filtered["tenure"].mean()
else:
    avg_tenure = 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Churned Customers", f"{churned:,}")
k3.metric("Churn Rate", f"{churn_rate:.1f}%")
k4.metric("Average Monthly Charge", f"${avg_monthly:,.2f}")

st.divider()

# ---------------- EMPTY FILTER CHECK ----------------
if filtered.empty:
    st.warning("No customers match the selected filters. Change your sidebar filters.")
    st.stop()

# ---------------- CHARTS ----------------
st.subheader("📈 Customer Insights")

left, right = st.columns(2)

# Churn distribution
with left:
    st.markdown("#### Churn Distribution")
    churn_counts = (
        filtered["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(name="Customers")
    )
    fig_churn = px.pie(
        churn_counts,
        names="Churn",
        values="Customers",
        hole=0.62,
        color="Churn",
        color_discrete_map={"Yes": "#f43f5e", "No": "#22c55e"}
    )
    fig_churn.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Churn Status",
        margin=dict(t=20, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_churn, use_container_width=True)

# Churn by contract
with right:
    st.markdown("#### Churn by Contract Type")
    if "Contract" in filtered.columns:
        contract_data = (
            filtered.groupby(["Contract", "Churn"], observed=False)
            .size()
            .reset_index(name="Customers")
        )
        fig_contract = px.bar(
            contract_data,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            color_discrete_map={"Yes": "#f43f5e", "No": "#6366f1"},
            custom_data=["Contract"]
        )
        fig_contract.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Contract",
            yaxis_title="Number of Customers",
            legend_title_text="Churn",
            margin=dict(t=20, b=10, l=10, r=10)
        )
        try:
            event = st.plotly_chart(
                fig_contract,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points"
            )
            points = event.selection.points
            if points:
                clicked_contract = points[0].get("x")
                if clicked_contract in filtered["Contract"].astype(str).unique():
                    st.session_state["selected_contract"] = clicked_contract
        except (TypeError, AttributeError):
            st.plotly_chart(fig_contract, use_container_width=True)

# Apply contract chart selection
if (
    "selected_contract" in st.session_state
    and "Contract" in filtered.columns
):
    selected_contract = st.session_state["selected_contract"]
    st.info(f"Chart selection: Contract = {selected_contract}")
    filtered = filtered[
        filtered["Contract"].astype(str) == str(selected_contract)
    ]

# Tenure chart
left, right = st.columns(2)

with left:
    st.markdown("#### Customer Tenure Distribution")
    if "tenure" in filtered.columns:
        fig_tenure = px.histogram(
            filtered,
            x="tenure",
            color="Churn",
            nbins=20,
            barmode="overlay",
            opacity=0.8,
            color_discrete_map={"Yes": "#f43f5e", "No": "#6366f1"}
        )
        fig_tenure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Tenure (months)",
            yaxis_title="Number of Customers",
            margin=dict(t=20, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_tenure, use_container_width=True)
    else:
        st.info("Tenure column is not available in this dataset.")

with right:
    st.markdown("#### Monthly Charges vs Churn")
    if "MonthlyCharges" in filtered.columns:
        fig_charges = px.box(
            filtered,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"Yes": "#f43f5e", "No": "#6366f1"},
            points=False
        )
        fig_charges.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Churn Status",
            yaxis_title="Monthly Charges ($)",
            showlegend=False,
            margin=dict(t=20, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_charges, use_container_width=True)

# ---------------- CUSTOMER TABLE ----------------
st.divider()
st.subheader("👥 Customer Details")

search_text = st.text_input(
    "Search customer ID",
    placeholder="Enter a customer ID..."
)

table_df = filtered.copy()
if search_text:
    table_df = table_df[
        table_df["customerID"].astype(str).str.contains(
            search_text, case=False, na=False
        )
    ]

preferred_columns = [
    "customerID", "gender", "tenure", "Contract",
    "MonthlyCharges", "TotalCharges", "Churn"
]
display_columns = [c for c in preferred_columns if c in table_df.columns]

st.dataframe(
    table_df[display_columns],
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "⬇️ Download filtered customer data",
    data=table_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_churn_customers.csv",
    mime="text/csv"
)

# ---------------- MACHINE LEARNING MODEL ----------------
st.divider()
st.subheader("🤖 Customer Churn Prediction")
st.caption("Enter customer details to estimate the likelihood of churn.")

if len(df) < 10 or df["Churn"].nunique() < 2:
    st.warning("The dataset needs at least 10 rows and both Yes and No churn classes.")
    st.stop()

model_df = df.drop(columns=["customerID"], errors="ignore").copy()
y = model_df.pop("Churn").map({"No": 0, "Yes": 1})
valid_rows = y.notna()
model_df = model_df.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

# Remove columns that contain no usable values
model_df = model_df.dropna(axis=1, how="all")

numeric_cols = model_df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [
    c for c in model_df.columns if c not in numeric_cols
]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if numeric_cols:
    transformers.append(("numeric", numeric_pipe, numeric_cols))
if categorical_cols:
    transformers.append(("categorical", categorical_pipe, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

try:
    X_train, X_test, y_train, y_test = train_test_split(
        model_df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    st.metric("Model Test Accuracy", f"{accuracy * 100:.1f}%")

    st.markdown("#### Enter customer information")

    input_values = {}
    input_columns = st.columns(3)

    for i, col in enumerate(model_df.columns):
        with input_columns[i % 3]:
            if col in numeric_cols:
                median_value = model_df[col].median()
                if pd.isna(median_value):
                    median_value = 0
                input_values[col] = st.number_input(
                    col,
                    value=float(median_value),
                    key="predict_" + col
                )
            else:
                options = model_df[col].dropna().astype(str).unique().tolist()
                if not options:
                    options = ["Unknown"]
                input_values[col] = st.selectbox(
                    col,
                    options=options,
                    key="predict_" + col
                )

    if st.button("🔍 Predict Churn Risk", type="primary"):
        new_customer = pd.DataFrame([input_values])
        probability = model.predict_proba(new_customer)[0][1]
        prediction = model.predict(new_customer)[0]

        if prediction == 1:
            st.error(f"⚠️ High churn prediction — estimated probability: {probability * 100:.1f}%")
            st.write("Suggested action: contact the customer and review their service or contract options.")
        else:
            st.success(f"✅ Low churn prediction — estimated probability: {probability * 100:.1f}%")
            st.write("Suggested action: continue engagement and monitor customer satisfaction.")

except Exception as e:
    st.error("The model could not be trained. Check the dataset columns and values.")
    st.exception(e)

# ---------------- FOOTER ----------------
st.divider()
st.caption("ChurnIQ | Customer Analytics & Machine Learning Portfolio Project")
