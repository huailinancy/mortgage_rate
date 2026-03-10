
# Run: streamlit run app.py

import os, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import joblib
import torch
import torch.nn as nn

st.set_page_config(page_title="Mortgage Rate Spread Predictor", layout="wide", page_icon=":house:")

NUM_FEATURES = [
    'Credit Score',
    'Mortgage Insurance Percentage (MI %)',
    'Original Combined Loan-to-Value (CLTV)',
    'Original Debt-to-Income (DTI) Ratio',
    'Original UPB',
    'Original Loan-to-Value (LTV)',
    'Number of Borrowers',
]
CAT_FEATURES = ['Channel', 'Loan Purpose', 'First Time Homebuyer Flag', 'Property State']
MODELS_DIR  = 'models'
FIGURES_DIR = 'figures'
REF_CLTV_MEDIAN = 80.0
REF_UPB_MEDIAN  = 200000.0
REF_STATE_MODE  = 'CA'

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

@st.cache_resource(show_spinner="Loading models...")
def load_assets():
    preprocessor = joblib.load(f'{MODELS_DIR}/preprocessor.joblib')
    lr_m  = joblib.load(f'{MODELS_DIR}/lr.joblib')
    dt_m  = joblib.load(f'{MODELS_DIR}/dt.joblib')
    rf_m  = joblib.load(f'{MODELS_DIR}/rf.joblib')
    xgb_m = joblib.load(f'{MODELS_DIR}/xgb.joblib')
    ckpt  = torch.load(f'{MODELS_DIR}/mlp.pt', map_location='cpu', weights_only=False)
    mlp_m = MLP(ckpt['input_dim'])
    mlp_m.load_state_dict(ckpt['state_dict'])
    mlp_m.eval()
    with open(f'{MODELS_DIR}/results.json') as f:
        results = json.load(f)
    feature_names = np.load(f'{MODELS_DIR}/feature_names.npy', allow_pickle=True).tolist()
    explainer = shap.TreeExplainer(xgb_m)
    return preprocessor, lr_m, dt_m, rf_m, xgb_m, mlp_m, results, feature_names, explainer

preprocessor, lr_m, dt_m, rf_m, xgb_m, mlp_m, results, feature_names, explainer = load_assets()

def predict_all(X_pp):
    preds = {}
    preds['Linear Regression'] = float(lr_m.predict(X_pp)[0])
    preds['Decision Tree']     = float(dt_m.predict(X_pp)[0])
    preds['Random Forest']     = float(rf_m.predict(X_pp)[0])
    preds['XGBoost']           = float(xgb_m.predict(X_pp)[0])
    with torch.no_grad():
        t = torch.tensor(X_pp, dtype=torch.float32)
        preds['MLP (PyTorch)'] = float(mlp_m(t).item())
    return preds

def show_fig(fname, caption):
    path = f'{FIGURES_DIR}/{fname}'
    if os.path.exists(path):
        st.image(path, use_container_width=True)
        st.caption(caption)
    else:
        st.warning(f"Figure not found: {fname}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary", "Descriptive Analytics", "Model Performance", "Explainability & Prediction"
])

with tab1:
    st.title("Mortgage Rate Spread Predictor")
    st.markdown('<p style="font-size:1.25rem;font-weight:600;margin-top:-0.5rem;">End-to-End Data Science Workflow</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown(
        "### What Is This Dataset?\n\n"
        "This project analyzes a **Freddie Mac single-family mortgage dataset** containing **42,126 loans** "
        "originated in 2014. Every row represents one 30-year, fixed-rate mortgage on a single-family "
        "primary residence in the United States. The dataset captures a rich snapshot of the conforming "
        "mortgage market: borrower financials (credit score, DTI ratio), collateral characteristics "
        "(LTV ratio, property state), and origination details (channel, loan purpose, mortgage insurance).\n\n"
        "The **prediction target** is `rate_spread` - the difference in percentage points between the "
        "loan original interest rate and the 10-year U.S. Treasury rate at origination. "
        "Features used in modeling include credit score, LTV, CLTV, DTI, loan amount (UPB), "
        "mortgage insurance percentage, number of borrowers, origination channel, loan purpose, "
        "first-time homebuyer status, and property state."
    )
    st.markdown(
        "### Why Does This Problem Matter?\n\n"
        "The mortgage rate spread is one of the most consequential numbers in a homebuyer's financial life. "
        "A difference of just 0.25 percentage points on a $300,000 30-year mortgage translates to roughly "
        "**$15,000 in extra interest payments** over the life of the loan. Predicting the spread from "
        "observable loan characteristics enables:\n\n"
        "- **Consumer empowerment**: Borrowers can benchmark whether their quoted rate is fair.\n"
        "- **Regulatory oversight**: Detect systematic pricing disparities for fair-lending enforcement.\n"
        "- **Lender consistency**: Audit pricing models for inconsistency or implicit bias.\n"
        "- **Systemic risk**: Mispriced mortgage spreads contributed to the 2008 financial crisis."
    )
    st.markdown(
        "### Approach & Key Findings\n\n"
        "Five machine learning models were trained on a **70/30 train/test split** (random_state=42). "
        "Tree-based models (Decision Tree, Random Forest, XGBoost) were tuned with **5-fold GridSearchCV**. "
        "The MLP was built in PyTorch (2x128 hidden layers, ReLU, Adam optimizer, 100 epochs). "
        "SHAP TreeExplainer was applied to XGBoost on 2,000 held-out loans."
    )
    best_name = min(results, key=lambda m: results[m]['RMSE'])
    best      = results[best_name]
    baseline  = results['Linear Regression']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model",    best_name)
    col2.metric("Best RMSE",     f"{best['RMSE']:.4f} pp")
    col3.metric("Best R2",       f"{best['R2']:.4f}")
    col4.metric("Baseline RMSE", f"{baseline['RMSE']:.4f} pp",
                delta=f"{best['RMSE'] - baseline['RMSE']:.4f} pp vs baseline", delta_color="inverse")
    st.markdown(
        f"**Key findings:** **{best_name}** achieved RMSE={best['RMSE']:.4f} pp (R2={best['R2']:.4f}), "
        f"outperforming the linear regression baseline (RMSE={baseline['RMSE']:.4f}). "
        "SHAP confirmed Credit Score as the dominant driver, followed by MI Percentage and LTV. "
        "Origination channel also matters: broker loans carry higher spreads than retail. "
        "These findings validate that lenders price mortgage risk in a systematic, explainable way."
    )

with tab2:
    st.title("Descriptive Analytics")
    st.markdown(
        "**Dataset:** 42,126 Freddie Mac 30-year fixed-rate single-family mortgages (2014).  \n"
        "**Target:** rate_spread (interest rate minus 10-yr Treasury, in percentage points).  \n"
        "**Features:** 7 numerical + 4 categorical (after dropping constant columns)."
    )
    with st.expander("1.1 Dataset Statistics", expanded=False):
        st.table(pd.DataFrame({
            'Statistic': ['Rows','Numerical features','Categorical features','Missing values',
                          'Target mean','Target std','Target min','Target max'],
            'Value':     ['42,126','7','4','0 (after cleaning)',
                          '1.837 pp','0.238 pp','0.710 pp','3.335 pp'],
        }))
    st.header("1.2 Target Distribution")
    show_fig('fig_target.png',
        "**Rate spread distribution** is approximately bell-shaped, centered at 1.84 pp (std=0.24 pp). "
        "The distribution is slightly right-skewed: a small proportion of borrowers pay spreads above "
        "2.5 pp, likely reflecting high-LTV or low-credit-score loans with a substantial risk premium.")
    st.header("1.3 Feature Distributions and Relationships")
    col1, col2 = st.columns(2)
    with col1:
        show_fig('fig_credit_score.png',
            "**Credit Score vs Rate Spread** reveals a clear negative relationship: lower credit scores "
            "consistently yield higher spreads. The scatter thickens around 700-780 (where most borrowers "
            "fall), and spreads compress above 760 - confirming that prime borrowers are near the floor.")
        show_fig('fig_ltv.png',
            "**LTV Ratio vs Rate Spread** shows that loans above 80% LTV cluster at higher spread values, "
            "consistent with lenders pricing elevated default risk. The positive relationship is diffuse, "
            "reflecting that LTV is one factor in a multi-factor pricing model.")
        show_fig('fig_channel.png',
            "**Origination Channel boxplot** shows broker-originated loans (B) carry a slightly higher "
            "median spread than retail (R) or correspondent (C) loans, possibly reflecting broker "
            "intermediation costs or riskier borrower profiles in the broker channel.")
    with col2:
        show_fig('fig_dti.png',
            "**DTI Ratio vs Rate Spread** reveals a weak positive trend - higher debt burdens correlate "
            "with marginally higher spreads. The wide vertical scatter at each DTI level suggests lenders "
            "weigh DTI alongside credit score and LTV in a holistic risk assessment.")
        show_fig('fig_loan_purpose.png',
            "**Loan Purpose boxplot** shows Cash-Out Refinances (C) carry slightly higher median spreads "
            "than purchase (P) or no-cash-out refinance (N) loans. Cash-out transactions increase borrower "
            "leverage and are historically associated with greater financial stress.")
        show_fig('fig_mi.png',
            "**MI Percentage vs Rate Spread**: Loans with 30% MI coverage have the widest spread "
            "distribution, indicating the riskiest borrowers. Loans with 0% MI (LTV <= 80%) show "
            "lower and tighter spreads, reflecting lower default risk.")
    st.header("1.4 Correlation Heatmap")
    show_fig('fig_corr.png',
        "**Correlation heatmap** of numerical features. LTV and CLTV are near-perfectly correlated "
        "(r ~ 0.97) and carry nearly identical information. Credit Score has the strongest negative "
        "correlation with rate_spread (r ~ -0.31), confirming it as the dominant risk signal. "
        "MI Percentage correlates positively with both LTV and rate_spread.")

with tab3:
    st.title("Model Performance")
    st.header("2.7 Model Comparison Summary")
    table_rows = []
    for name, m in results.items():
        table_rows.append({
            'Model': name, 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
            'Best Hyperparameters': str(m['best_params']) if 'best_params' in m else 'N/A',
        })
    st.dataframe(pd.DataFrame(table_rows).set_index('Model'), use_container_width=True)
    show_fig('fig_model_comparison.png',
        "**Model comparison bar chart** (RMSE on 30% held-out test set, lower is better). "
        "XGBoost achieves the best performance with Random Forest close behind. The MLP performs "
        "comparably to tree ensembles. Decision Tree is most interpretable but sacrifices accuracy. "
        "Linear Regression is the baseline - competitive given the near-linear credit-score relationship.")
    st.markdown(
        "**Discussion:** XGBoost delivers the lowest RMSE by capturing non-linear interactions that "
        "linear regression cannot represent. Random Forest is nearly as accurate. The MLP matches "
        "tree ensembles without feature engineering - showing deep learning works on tabular data. "
        "The Decision Tree is most interpretable, with top splits on Credit Score and MI % readable "
        "as a pricing rubric. XGBoost is best for deployment; Decision Tree for regulatory explanation."
    )
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
            cap = (f"**{name}** | MAE={m['MAE']} | RMSE={m['RMSE']} | R2={m['R2']}"
                   + (f"  \nBest params: `{params}`" if params else ""))
            show_fig(fname, cap)
    st.header("MLP Training History")
    show_fig('fig_mlp_history.png',
        "**MLP training and validation curves** over 100 epochs. Training loss decreases continuously "
        "while validation loss plateaus and slightly rises after ~epoch 40 (mild overfitting). "
        "Best generalization is around epoch 40-50; early stopping or dropout could help.")
    st.header("Decision Tree Structure (top 3 levels)")
    show_fig('fig_dt_tree.png',
        "**Best decision tree** (top 3 levels). First split on Credit Score, then MI Percentage and LTV "
        "- the three dominant risk pricing factors. Each leaf shows the predicted spread for loans "
        "matching that path, making the pricing logic fully transparent for lenders and regulators.")

with tab4:
    st.title("Explainability & Interactive Prediction")
    st.header("3.1 SHAP Analysis (XGBoost)")
    col1, col2 = st.columns(2)
    with col1:
        show_fig('fig_shap_summary.png',
            "**SHAP Beeswarm Plot** - each dot is one loan. Red = high feature value, blue = low. "
            "Features ranked by mean |SHAP|. Credit Score dominates: low scores (blue dots) strongly "
            "increase predicted spread. High MI% and LTV (red) also push predictions upward.")
    with col2:
        show_fig('fig_shap_bar.png',
            "**SHAP Feature Importance Bar Chart** ranks features by mean absolute SHAP contribution. "
            "Credit Score is the most important feature, followed by MI Percentage and LTV. "
            "Property State and Channel contribute meaningfully but are secondary.")
    show_fig('fig_shap_waterfall.png',
        "**SHAP Waterfall Plot** for the loan with the highest predicted rate spread in the test set. "
        "Each bar shows how a feature pushes the prediction above (red) or below (blue) the baseline. "
        "A combination of low credit score, high LTV, and 30% MI drives this loan far above average.")
    st.markdown(
        "**Interpretation for decision-makers:**\n\n"
        "- **Credit Score** is the single strongest driver - improving from 620 to 720 meaningfully "
        "reduces the quoted rate. Credit repair counseling programs have measurable financial impact.\n"
        "- **MI Percentage** (proxy for LTV > 80%) has the second-largest effect. A larger down payment "
        "saves on both the MI premium and the rate spread itself.\n"
        "- **Origination channel** matters: broker loans carry higher spreads. Regulators should "
        "investigate whether this reflects genuine risk differences or intermediary mark-ups.\n"
        "- The SHAP waterfall enables **individual-loan explanations** - a key requirement for "
        "fair-lending compliance and adverse action notices."
    )
    st.divider()
    st.header("Interactive Prediction")
    st.markdown(
        "Set feature values to get a **real-time rate spread prediction** from all five models.  \n"
        "*(CLTV, UPB, and Property State are fixed at dataset medians/mode for simplicity.)*"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("Credit Score", 560, 832, 720, step=5)
        dti          = st.slider("DTI Ratio (%)", 1, 50, 36)
        ltv          = st.slider("LTV Ratio (%)", 7, 103, 80)
    with col2:
        mi_pct      = st.selectbox("MI Percentage (%)", [0, 12, 16, 25, 30], index=0)
        n_borrowers = st.selectbox("Number of Borrowers", [1, 2], index=1)
        channel     = st.selectbox("Origination Channel",
                                   ['R - Retail', 'C - Correspondent', 'B - Broker'])
    with col3:
        loan_purpose = st.selectbox("Loan Purpose",
                                    ['P - Purchase', 'C - Cash-Out Refi', 'N - No Cash-Out Refi'])
        first_time   = st.selectbox("First-Time Homebuyer?", ['N', 'Y'])
        model_choice = st.selectbox("Model for prediction", list(results.keys()))

    channel_val = channel.split(' - ')[0].strip()
    purpose_val = loan_purpose.split(' - ')[0].strip()
    user_input = {
        'Credit Score':                           credit_score,
        'Mortgage Insurance Percentage (MI %)':   mi_pct,
        'Original Combined Loan-to-Value (CLTV)': REF_CLTV_MEDIAN,
        'Original Debt-to-Income (DTI) Ratio':    dti,
        'Original UPB':                           REF_UPB_MEDIAN,
        'Original Loan-to-Value (LTV)':           ltv,
        'Number of Borrowers':                    n_borrowers,
        'Channel':                                channel_val,
        'Loan Purpose':                           purpose_val,
        'First Time Homebuyer Flag':              first_time,
        'Property State':                         REF_STATE_MODE,
    }
    user_df   = pd.DataFrame([user_input])
    X_user    = preprocessor.transform(user_df)
    all_preds = predict_all(X_user)
    chosen    = all_preds[model_choice]

    st.markdown(f"### Prediction - **{model_choice}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Rate Spread", f"{chosen:.4f} pp",
              help="Percentage points above the 10-year Treasury rate")
    c2.metric("Model RMSE (test set)", f"+/- {results[model_choice]['RMSE']:.4f} pp",
              help="Expected prediction error on held-out test data")
    c3.metric("Approx. Mortgage Rate", f"~{chosen + 2.54:.2f}%",
              help="Illustrative only (rate_spread + 2.54% Treasury reference)")
    st.markdown("**All-model predictions for your input:**")
    st.dataframe(pd.DataFrame([
        {'Model': k, 'Predicted Rate Spread (pp)': round(v, 4)}
        for k, v in all_preds.items()
    ]).set_index('Model'), use_container_width=True)

    st.markdown("### SHAP Waterfall for Your Input (XGBoost)")
    try:
        expl_user = explainer(X_user)
        fig_wf    = plt.figure(figsize=(10, 7))
        shap.waterfall_plot(expl_user[0], show=False)
        plt.title("SHAP Waterfall - Your Custom Input", fontsize=12)
        st.pyplot(fig_wf)
        plt.close(fig_wf)
        st.caption(
            "This waterfall plot shows how each feature shifts the XGBoost prediction above (red) or "
            "below (blue) the model baseline. Adjust sliders above and the plot updates in real time."
        )
    except Exception as e:
        st.error(f"Could not generate SHAP waterfall: {e}")
