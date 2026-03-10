"""
MSIS 522 HW1 – Streamlit App
==============================
End-to-end mortgage rate spread analysis and interactive predictor.
Run: streamlit run app.py
"""

import os, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import joblib
import torch
import torch.nn as nn

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mortgage Rate Spread Predictor",
    layout="wide",
    page_icon="🏠",
)

# ── Feature lists (must match train_models.py) ─────────────────────────────
NUM_FEATURES = [
    'Credit Score',
    'Mortgage Insurance Percentage (MI %)',
    'Original Combined Loan-to-Value (CLTV)',
    'Original Debt-to-Income (DTI) Ratio',
    'Original UPB',
    'Original Loan-to-Value (LTV)',
    'Number of Borrowers',
]
CAT_FEATURES = [
    'Channel',
    'Loan Purpose',
    'First Time Homebuyer Flag',
    'Property State',
]
TARGET = 'rate_spread'

MODELS_DIR  = 'models'
FIGURES_DIR = 'figures'

# Pre-computed from full dataset (avoids bundling the large CSV in the app)
REF_CLTV_MEDIAN = 80.0
REF_UPB_MEDIAN  = 200000.0
REF_STATE_MODE  = 'CA'

# ── MLP definition (must match train_models.py) ────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# ── Load all assets (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_assets():
    preprocessor = joblib.load(f'{MODELS_DIR}/preprocessor.joblib')
    lr_m  = joblib.load(f'{MODELS_DIR}/lr.joblib')
    dt_m  = joblib.load(f'{MODELS_DIR}/dt.joblib')
    rf_m  = joblib.load(f'{MODELS_DIR}/rf.joblib')
    xgb_m = joblib.load(f'{MODELS_DIR}/xgb.joblib')

    ckpt      = torch.load(f'{MODELS_DIR}/mlp.pt', map_location='cpu', weights_only=False)
    mlp_m     = MLP(ckpt['input_dim'])
    mlp_m.load_state_dict(ckpt['state_dict'])
    mlp_m.eval()

    with open(f'{MODELS_DIR}/results.json') as f:
        results = json.load(f)

    feature_names = np.load(f'{MODELS_DIR}/feature_names.npy', allow_pickle=True).tolist()
    explainer     = shap.TreeExplainer(xgb_m)

    return preprocessor, lr_m, dt_m, rf_m, xgb_m, mlp_m, results, feature_names, explainer

preprocessor, lr_m, dt_m, rf_m, xgb_m, mlp_m, results, feature_names, explainer = load_assets()

# ── Inference helper ───────────────────────────────────────────────────────
def predict_all(X_pp: np.ndarray) -> dict:
    preds = {}
    preds['Linear Regression'] = float(lr_m.predict(X_pp)[0])
    preds['Decision Tree']     = float(dt_m.predict(X_pp)[0])
    preds['Random Forest']     = float(rf_m.predict(X_pp)[0])
    preds['XGBoost']           = float(xgb_m.predict(X_pp)[0])
    with torch.no_grad():
        t = torch.tensor(X_pp, dtype=torch.float32)
        preds['MLP (PyTorch)'] = float(mlp_m(t).item())
    return preds

def show_fig(fname: str, caption: str):
    path = f'{FIGURES_DIR}/{fname}'
    if os.path.exists(path):
        st.image(path, use_container_width=True)
        st.caption(caption)
    else:
        st.warning(f"Figure not found: {fname}. Run train_models.py first.")

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 – EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🏠 Mortgage Rate Spread Predictor")
    st.subheader("MSIS 522 HW1 — End-to-End Data Science Workflow")
    st.divider()

    st.markdown("""
    ### What Is This Dataset?

    This project analyzes a **Freddie Mac single-family mortgage dataset** containing **42,126 loans**
    originated in 2014. Every row represents one 30-year, fixed-rate mortgage on a single-family
    primary residence somewhere in the United States. The dataset captures a rich snapshot of the
    conforming mortgage market: borrower financials (credit score, debt-to-income ratio), collateral
    characteristics (loan-to-value ratio, property state), origination details (channel, loan purpose,
    mortgage insurance), and market context (Treasury rate at origination).

    The **prediction target** is `rate_spread` — the difference in percentage points between the
    loan's original interest rate and the 10-year U.S. Treasury rate on the origination date.
    For example, if the Treasury rate is 2.54% and the mortgage rate is 4.25%, the rate spread
    is 1.71 pp. Features used in modeling include credit score, loan-to-value (LTV) ratio,
    combined LTV (CLTV), debt-to-income (DTI) ratio, loan amount (UPB), mortgage insurance
    percentage, number of borrowers, origination channel, loan purpose, first-time homebuyer
    status, and property state.
    """)

    st.markdown("""
    ### Why Does This Problem Matter?

    The mortgage rate spread is one of the most consequential numbers in a homebuyer's financial life.
    A difference of just 0.25 percentage points on a $300,000, 30-year mortgage translates to roughly
    **$15,000 in extra interest payments** over the life of the loan. Yet the spread a borrower receives
    is often opaque — determined by an algorithmic mix of risk factors that most consumers do not
    understand. Predicting the spread from observable loan characteristics enables several high-impact
    applications:

    - **Consumer empowerment**: Borrowers can benchmark whether their quoted rate is fair given their
      credit profile, flagging potential overcharging before signing.
    - **Regulatory oversight**: Regulators can use model predictions to detect systematic pricing
      disparities after controlling for legitimate risk factors — a key tool for fair-lending enforcement.
    - **Lender consistency**: Lenders can audit their own pricing models for inconsistency or implicit
      bias, improving both compliance and reputation.
    - **Systemic risk**: Mispriced mortgage spreads contributed to the 2008 financial crisis; better
      prediction models can help identify under-priced risk before it accumulates.
    """)

    st.markdown("""
    ### Approach & Key Findings

    Five machine learning models were trained and evaluated on a **70/30 train/test split**
    (`random_state=42`). Tree-based models (Decision Tree, Random Forest, XGBoost) were tuned with
    **5-fold cross-validated GridSearchCV**. The neural network is a two-hidden-layer MLP built in
    PyTorch (128 units per layer, ReLU activations, Adam optimizer, 100 epochs). SHAP TreeExplainer
    was applied to the best tree model (XGBoost) on a held-out sample of 2,000 loans.
    """)

    # Key metrics row
    best_name = min(results, key=lambda m: results[m]['RMSE'])
    best      = results[best_name]
    baseline  = results['Linear Regression']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model",     best_name)
    col2.metric("Best RMSE",      f"{best['RMSE']:.4f} pp")
    col3.metric("Best R²",        f"{best['R2']:.4f}")
    col4.metric("Baseline RMSE",  f"{baseline['RMSE']:.4f} pp",
                delta=f"{best['RMSE'] - baseline['RMSE']:.4f} pp vs baseline",
                delta_color="inverse")

    st.markdown(f"""
    **Key findings:** **{best_name}** achieved the lowest test-set RMSE of {best['RMSE']:.4f} pp
    with R² = {best['R2']:.4f}, outperforming the linear regression baseline (RMSE = {baseline['RMSE']:.4f}).
    SHAP analysis confirmed that **Credit Score** is by far the dominant pricing driver — lower scores
    strongly push the spread upward. **Mortgage Insurance Percentage** and **LTV** are the next
    most impactful, reflecting the well-known relationship between leverage and risk pricing.
    **Origination channel** also matters: broker-originated loans tend to carry higher spreads than
    retail loans. Collectively, these findings validate that lenders price mortgage risk in a largely
    systematic, explainable way — and that machine learning can faithfully capture this process.
    """)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 – DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Descriptive Analytics")
    st.markdown("""
    **Dataset:** 42,126 Freddie Mac 30-year fixed-rate single-family mortgages (2014).
    **Target:** `rate_spread` (interest rate minus 10-yr Treasury, in percentage points).
    **Features:** 7 numerical + 4 categorical (after dropping constant columns).
    """)

    # 1.1 Basic stats
    with st.expander("1.1 Dataset Statistics", expanded=False):
        stat_data = {
            'Statistic': ['Rows', 'Numerical features', 'Categorical features', 'Missing values',
                          'Target mean', 'Target std', 'Target min', 'Target max'],
            'Value':     ['42,126', '7', '4', '0 (after cleaning)',
                          '1.837 pp', '0.238 pp', '0.710 pp', '3.335 pp'],
        }
        st.table(pd.DataFrame(stat_data))

    st.header("1.2 Target Distribution")
    show_fig('fig_target.png',
             "**Rate spread distribution** is approximately bell-shaped, centered at 1.84 pp with a "
             "standard deviation of 0.24 pp. The distribution is slightly right-skewed: a small "
             "proportion of borrowers pay noticeably higher spreads (>2.5 pp), likely reflecting "
             "high-LTV or low-credit-score loans where lenders charge a substantial risk premium.")

    st.header("1.3 Feature Distributions and Relationships")
    col1, col2 = st.columns(2)

    with col1:
        show_fig('fig_credit_score.png',
                 "**Credit Score vs Rate Spread** reveals a clear negative relationship: borrowers "
                 "with lower credit scores consistently receive higher rate spreads. The scatter "
                 "thickens around scores 700–780 (where most borrowers fall), and spreads compress "
                 "tightly above 760 — confirming that prime borrowers are priced near the market floor.")

        show_fig('fig_ltv.png',
                 "**LTV Ratio vs Rate Spread** shows that loans above 80% LTV (which require mortgage "
                 "insurance) cluster at higher spread values, consistent with lenders pricing the "
                 "elevated default risk. The relationship is positive but diffuse, reflecting that "
                 "LTV is one component of a multi-factor pricing model rather than the sole driver.")

        show_fig('fig_channel.png',
                 "**Origination Channel boxplot** shows that broker-originated loans (B) carry a "
                 "slightly higher median spread than retail (R) or correspondent (C) loans. This may "
                 "reflect the cost of broker intermediation, or that brokers systematically serve "
                 "borrowers with marginally riskier profiles in this dataset.")

    with col2:
        show_fig('fig_dti.png',
                 "**DTI Ratio vs Rate Spread** reveals a weak positive trend — higher debt burdens "
                 "correlate with marginally higher spreads. The wide vertical scatter at each DTI "
                 "level suggests lenders do not price DTI in isolation; it is weighted alongside "
                 "credit score, LTV, and loan purpose in a holistic risk assessment.")

        show_fig('fig_loan_purpose.png',
                 "**Loan Purpose boxplot** shows that Cash-Out Refinances (C) carry slightly higher "
                 "median spreads than purchase loans (P) or no-cash-out refinances (N). Cash-out "
                 "transactions increase borrower leverage and are historically associated with "
                 "greater financial stress, justifying a modest risk premium from the lender.")

        show_fig('fig_mi.png',
                 "**MI Percentage vs Rate Spread**: Loans with 30% mortgage insurance coverage have "
                 "the widest spread distribution, indicating these are the riskiest borrowers who "
                 "need the most insurance protection. Loans with 0% MI (LTV ≤ 80%) consistently "
                 "show lower and tighter spreads, reflecting their lower default risk profile.")

    st.header("1.4 Correlation Heatmap")
    show_fig('fig_corr.png',
             "**Correlation heatmap** of numerical features. LTV and CLTV are near-perfectly "
             "correlated (r ≈ 0.97), carrying nearly identical information — either alone is "
             "sufficient for modeling. Credit Score has the strongest negative correlation with "
             "rate_spread (r ≈ –0.31), confirming it as the dominant borrower risk signal. "
             "MI Percentage correlates positively with both LTV and rate_spread: higher-leverage "
             "loans require more insurance and command larger spreads from lenders.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🤖 Model Performance")

    # ── 2.7 Comparison table ──────────────────────────────────────────────
    st.header("2.7 Model Comparison Summary")
    table_rows = []
    for name, m in results.items():
        row = {
            'Model':  name,
            'MAE':    m['MAE'],
            'RMSE':   m['RMSE'],
            'R²':     m['R2'],
        }
        if 'best_params' in m:
            row['Best Hyperparameters'] = str(m['best_params'])
        else:
            row['Best Hyperparameters'] = 'N/A'
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows).set_index('Model')
    st.dataframe(df_table, use_container_width=True)

    show_fig('fig_model_comparison.png',
             "**Model comparison bar chart** (RMSE on 30% held-out test set, lower is better). "
             "XGBoost achieves the best performance, with Random Forest close behind. The neural "
             "network MLP performs comparably to tree ensembles despite being architecturally very "
             "different. Decision Tree is the most interpretable but sacrifices accuracy. Linear "
             "Regression provides the baseline — still surprisingly competitive given the "
             "near-linear credit-score/spread relationship.")

    st.markdown("""
    **Discussion:** XGBoost delivers the lowest RMSE by capturing non-linear interactions among
    credit score, LTV, MI percentage, and origination channel that linear regression cannot represent.
    Random Forest is nearly as accurate and benefits from averaging across hundreds of diverse trees.
    The MLP neural network, trained from scratch with PyTorch, achieves competitive performance
    without any feature engineering — demonstrating that deep learning can match tree ensembles
    on structured tabular data when given sufficient data (42K rows). The Decision Tree is the
    most interpretable: its top splits on Credit Score and MI % are clearly legible as a pricing
    rubric. The trade-off is reduced accuracy compared to ensembles. For deployment, XGBoost is
    the best choice; for regulatory explanation, the Decision Tree provides the most transparent
    rationale.
    """)

    # ── Predicted vs Actual ───────────────────────────────────────────────
    st.header("Predicted vs Actual Plots")
    figs_info = [
        ('fig_lr_pred.png',  'Linear Regression', None),
        ('fig_dt_pred.png',  'Decision Tree',     results.get('Decision Tree',  {}).get('best_params')),
        ('fig_rf_pred.png',  'Random Forest',     results.get('Random Forest',  {}).get('best_params')),
        ('fig_xgb_pred.png', 'XGBoost',           results.get('XGBoost',        {}).get('best_params')),
        ('fig_mlp_pred.png', 'MLP (PyTorch)',      None),
    ]
    cols = st.columns(3)
    for i, (fname, name, params) in enumerate(figs_info):
        with cols[i % 3]:
            m = results[name]
            caption = (f"**{name}** | MAE={m['MAE']} | RMSE={m['RMSE']} | R²={m['R2']}"
                       + (f"  \nBest params: `{params}`" if params else ""))
            show_fig(fname, caption)

    # ── MLP training curves ───────────────────────────────────────────────
    st.header("MLP Training History")
    show_fig('fig_mlp_history.png',
             "**MLP training and validation curves** over 100 epochs. Training loss decreases "
             "continuously, while validation loss begins to plateau and slightly rise after ~epoch 40, "
             "indicating mild overfitting. The best generalization occurs around 40–50 epochs; "
             "early stopping or dropout could further improve out-of-sample performance.")

    # ── Decision Tree visualization ───────────────────────────────────────
    st.header("Decision Tree Structure (top 3 levels)")
    show_fig('fig_dt_tree.png',
             "**Best decision tree** (top 3 splitting levels). The tree first splits on Credit Score, "
             "then on MI Percentage and LTV — the three dominant risk pricing factors. Each leaf "
             "shows the predicted rate spread for loans matching that path. This structure is directly "
             "interpretable: a lender or regulator can trace exactly which criteria lead to each "
             "pricing tier.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 – EXPLAINABILITY & INTERACTIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("🔍 Explainability & Interactive Prediction")

    # ── 3.1 SHAP plots ────────────────────────────────────────────────────
    st.header("3.1 SHAP Analysis (XGBoost)")

    col1, col2 = st.columns(2)
    with col1:
        show_fig('fig_shap_summary.png',
                 "**SHAP Beeswarm Plot** — each dot is one loan. Red = high feature value, "
                 "blue = low. Features are ranked by mean |SHAP|. Credit Score dominates: "
                 "low scores (blue dots on the right) strongly increase predicted spread. "
                 "High MI% and LTV (red) also push predictions upward, while high credit scores "
                 "(red on the left) reduce predicted spread below the baseline.")
    with col2:
        show_fig('fig_shap_bar.png',
                 "**SHAP Feature Importance Bar Chart** ranks features by their mean absolute SHAP "
                 "contribution. Credit Score is the single most important feature, followed by MI "
                 "Percentage and LTV. Property State and Channel contribute meaningfully but are "
                 "secondary to the core borrower and collateral risk factors.")

    show_fig('fig_shap_waterfall.png',
             "**SHAP Waterfall Plot** for the loan with the highest predicted rate spread in the "
             "test set. Each colored bar shows how a feature pushes the prediction above (red) or "
             "below (blue) the model's expected baseline value. A combination of very low credit "
             "score, high LTV, and 30% MI coverage drives this loan's predicted spread far above "
             "the average — the waterfall makes each contributing factor visually transparent.")

    st.markdown("""
    **Interpretation for decision-makers:**

    - **Credit Score** is the single strongest driver of rate spread — improving a borrower's score
      from 620 to 720 can meaningfully reduce their quoted rate. Counseling programs targeting credit
      repair would have a measurable financial impact on borrowers.
    - **MI Percentage** (a proxy for LTV > 80%) has the second-largest effect. Borrowers who can
      make a larger down payment to avoid mortgage insurance save on both the MI premium and the
      spread itself.
    - **Origination channel** matters: broker-originated loans systematically carry higher spreads.
      Regulators should investigate whether this reflects genuine risk differences or intermediary
      mark-ups that harm borrowers.
    - The SHAP waterfall enables **individual-loan explanations** — any underwriting decision can
      be attributed to specific factors, a key requirement for fair-lending compliance and adverse
      action notices.
    """)

    st.divider()

    # ── Interactive Prediction ─────────────────────────────────────────────
    st.header("Interactive Prediction")
    st.markdown("""
    Set feature values below to get a **real-time rate spread prediction** from all five models.
    *(CLTV, UPB, and Property State are fixed at their dataset medians/mode for simplicity.)*
    """)

    # Reference medians (pre-computed from full dataset)
    cltv_median = REF_CLTV_MEDIAN
    upb_median  = REF_UPB_MEDIAN
    state_mode  = REF_STATE_MODE

    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("Credit Score", 560, 832, 720, step=5)
        dti          = st.slider("DTI Ratio (%)", 1, 50, 36)
        ltv          = st.slider("LTV Ratio (%)", 7, 103, 80)
    with col2:
        mi_pct      = st.selectbox("MI Percentage (%)", [0, 12, 16, 25, 30], index=0)
        n_borrowers = st.selectbox("Number of Borrowers", [1, 2], index=1)
        channel     = st.selectbox("Origination Channel",
                                   ['R – Retail', 'C – Correspondent', 'B – Broker'])
    with col3:
        loan_purpose = st.selectbox("Loan Purpose",
                                    ['P – Purchase', 'C – Cash-Out Refi', 'N – No Cash-Out Refi'])
        first_time   = st.selectbox("First-Time Homebuyer?", ['N', 'Y'])
        model_choice = st.selectbox("Model for prediction", list(results.keys()))

    channel_val = channel.split('–')[0].strip()
    purpose_val = loan_purpose.split('–')[0].strip()

    user_input = {
        'Credit Score':                              credit_score,
        'Mortgage Insurance Percentage (MI %)':      mi_pct,
        'Original Combined Loan-to-Value (CLTV)':    cltv_median,
        'Original Debt-to-Income (DTI) Ratio':       dti,
        'Original UPB':                              upb_median,
        'Original Loan-to-Value (LTV)':              ltv,
        'Number of Borrowers':                       n_borrowers,
        'Channel':                                   channel_val,
        'Loan Purpose':                              purpose_val,
        'First Time Homebuyer Flag':                 first_time,
        'Property State':                            state_mode,
    }

    user_df  = pd.DataFrame([user_input])
    X_user   = preprocessor.transform(user_df)
    all_preds = predict_all(X_user)
    chosen   = all_preds[model_choice]

    # Prediction output
    st.markdown(f"### Prediction — **{model_choice}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Rate Spread", f"{chosen:.4f} pp",
              help="Percentage points above the 10-year Treasury rate")
    c2.metric("Model RMSE (test set)", f"±{results[model_choice]['RMSE']:.4f} pp",
              help="Expected prediction error on held-out test data")
    est_rate = chosen + 2.54  # approximate — illustrative only
    c3.metric("Approx. Mortgage Rate", f"~{est_rate:.2f}%",
              help="Illustrative only (rate_spread + 2.54% Treasury reference)")

    st.markdown("**All-model predictions for your input:**")
    pred_df = pd.DataFrame([
        {'Model': k, 'Predicted Rate Spread (pp)': round(v, 4)}
        for k, v in all_preds.items()
    ]).set_index('Model')
    st.dataframe(pred_df, use_container_width=True)

    # ── Live SHAP Waterfall ────────────────────────────────────────────────
    st.markdown("### SHAP Waterfall for Your Input (XGBoost)")
    try:
        expl_user = explainer(X_user)
        fig_wf, _ = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(expl_user[0], show=False)
        plt.title("SHAP Waterfall – Your Custom Input", fontsize=12)
        st.pyplot(fig_wf)
        plt.close(fig_wf)
        st.caption(
            "This waterfall plot shows how each feature shifts the XGBoost prediction "
            "above (red) or below (blue) the model's baseline expected value. The final "
            "prediction equals the baseline plus all SHAP contributions summed together. "
            "Adjust the sliders above and this plot updates in real time."
        )
    except Exception as e:
        st.error(f"Could not generate SHAP waterfall: {e}")
