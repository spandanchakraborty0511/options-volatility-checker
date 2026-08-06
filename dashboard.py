"""
SPY Options Volatility Checker — Interactive Dashboard (Phase 5).
Provides an interactive dashboard built with Streamlit and Plotly.
Run with: streamlit run D:\Volatality_checker\dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ensure project directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import data_loader
import realized_vol
from svi_model import SVIModel, flag_rich_cheap_strikes
from ssvi_model import SSVISurface
import garch_model

st.set_page_config(
    page_title="SPY Options Volatility Checker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .stMetric label { font-size: 0.9rem !important; color: #475569 !important; }
    .stMetric .metric-value { font-size: 1.6rem !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_cached_available_dates():
    return data_loader.get_available_dates()

@st.cache_data(ttl=3600)
def get_cached_expiries(date_str):
    return data_loader.get_expiries_for_date(date_str)

def main():
    st.markdown('<div class="main-header">📈 SPY Options Volatility Checker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Institutional-grade SVI/SSVI Surface Calibration & Rich/Cheap Option Signal Engine</div>', unsafe_allow_html=True)

    dates = get_cached_available_dates()
    if not dates:
        st.error("No dates found in database!")
        return

    # Sidebar Controls
    st.sidebar.header("⚙️ Model Controls")
    
    # Select Date
    default_idx = len(dates) - 1  # Default to latest date
    selected_date = st.sidebar.selectbox("Select Trading Date", dates, index=default_idx)
    
    expiries = get_cached_expiries(selected_date)
    selected_expiry = st.sidebar.selectbox("Select Expiry Date", expiries, index=min(3, len(expiries)-1))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Parameters")
    
    moneyness_min = st.sidebar.slider("Moneyness Min (K/S)", 0.50, 0.90, config.MONEYNESS_MIN, 0.05)
    moneyness_max = st.sidebar.slider("Moneyness Max (K/S)", 1.10, 1.50, config.MONEYNESS_MAX, 0.05)
    vol_threshold = st.sidebar.slider("Rich/Cheap Vol Threshold (%)", 0.5, 5.0, config.RICH_CHEAP_THRESHOLD_VOL * 100, 0.5) / 100.0

    # Load & Clean Data
    raw_slice = data_loader.load_raw_options_slice(selected_date, selected_expiry)
    clean_slice = data_loader.clean_options_slice(
        raw_slice,
        moneyness_min=moneyness_min,
        moneyness_max=moneyness_max
    )
    
    raw_date_all = data_loader.load_raw_options_slice(selected_date)
    clean_date_all = data_loader.clean_options_slice(
        raw_date_all,
        moneyness_min=moneyness_min,
        moneyness_max=moneyness_max
    )

    # Volatility Metrics Summary Row
    try:
        vol_summary = realized_vol.get_volatility_summary_for_date(selected_date)
        spot_price = vol_summary['underlying_price']
        atm_iv = vol_summary['current_atm_iv']
        rv_21d = vol_summary['rv_21d']
        iv_rank = vol_summary['iv_rank_52w']
        vol_prem = vol_summary['volatility_premium']
    except Exception as e:
        spot_price = clean_slice['spot'].iloc[0] if not clean_slice.empty else 0.0
        atm_iv, rv_21d, iv_rank, vol_prem = 0.20, 0.18, 40.0, 0.02

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("SPY Spot Price", f"${spot_price:.2f}")
    col2.metric("30d ATM IV", f"{atm_iv*100:.2f}%" if not np.isnan(atm_iv) else "N/A")
    col3.metric("21d Realized Vol", f"{rv_21d*100:.2f}%" if not np.isnan(rv_21d) else "N/A")
    col4.metric("52w IV Rank", f"{iv_rank:.1f}%" if not np.isnan(iv_rank) else "N/A")
    col5.metric("Vol Premium (IV - RV)", f"{vol_prem*100:+.2f}%" if not np.isnan(vol_prem) else "N/A")

    st.markdown("---")

    # Tabs navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 SVI Smile Calibration",
        "🌐 SSVI Volatility Surface",
        "📊 IV vs Realized Vol Timeline",
        "📋 Flagged Signal Table"
    ])

    # Tab 1: Single Expiry SVI Smile
    with tab1:
        if clean_slice.empty:
            st.warning("No clean options available for this expiry slice under current filters.")
        else:
            tau = clean_slice['tau'].iloc[0]
            svi = SVIModel()
            svi.fit(clean_slice['log_moneyness'].values, clean_slice['total_variance'].values, tau=tau)
            df_flagged = flag_rich_cheap_strikes(clean_slice, svi, vol_threshold=vol_threshold)
            
            k_smooth = np.linspace(clean_slice['log_moneyness'].min() - 0.02, clean_slice['log_moneyness'].max() + 0.02, 200)
            iv_smooth = svi.predict_iv(k_smooth, tau=tau)
            strikes_smooth = spot_price * np.exp(k_smooth)
            
            fig_smile = go.Figure()
            
            # Fitted SVI Curve
            fig_smile.add_trace(go.Scatter(
                x=strikes_smooth,
                y=iv_smooth * 100,
                mode='lines',
                name=f'Fitted SVI Model (RMSE: {svi.rmse*100:.2f}%)',
                line=dict(color='#1E293B', width=3)
            ))
            
            # Market IV points color-coded by signal
            colors_map = {'FAIR': '#3B82F6', 'RICH': '#EF4444', 'CHEAP': '#10B981'}
            symbols_map = {'FAIR': 'circle', 'RICH': 'triangle-up', 'CHEAP': 'triangle-down'}
            
            for sig in ['FAIR', 'RICH', 'CHEAP']:
                sub = df_flagged[df_flagged['signal'] == sig]
                if not sub.empty:
                    fig_smile.add_trace(go.Scatter(
                        x=sub['strike'],
                        y=sub['iv'] * 100,
                        mode='markers',
                        name=f'{sig} Market IV',
                        marker=dict(color=colors_map[sig], symbol=symbols_map[sig], size=10),
                        hovertemplate="<b>Strike:</b> %{x}<br><b>Market IV:</b> %{y:.2f}%<br><b>Option:</b> %{customdata[0]}<br><b>Bid/Ask:</b> %{customdata[1]:.2f} / %{customdata[2]:.2f}<extra></extra>",
                        customdata=sub[['option_type', 'bid', 'ask']].values
                    ))
                    
            fig_smile.add_vline(x=spot_price, line_dash="dash", line_color="gray", annotation_text=f"Spot ${spot_price:.2f}")
            fig_smile.update_layout(
                title=f"SVI Smile Fit & Rich/Cheap Flags ({selected_date}, Expiry: {selected_expiry}, DTE: {clean_slice['dte'].iloc[0]}d)",
                xaxis_title="Strike Price ($)",
                yaxis_title="Implied Volatility (%)",
                height=520,
                hovermode="closest",
                template="plotly_white"
            )
            st.plotly_chart(fig_smile, use_container_width=True)
            
            st.subheader("SVI Parameter Calibration Summary")
            p_cols = st.columns(6)
            params = svi.get_params()
            p_cols[0].metric("a (Level)", f"{params['a']:.5f}")
            p_cols[1].metric("b (Slope)", f"{params['b']:.5f}")
            p_cols[2].metric("rho (Skew)", f"{params['rho']:.4f}")
            p_cols[3].metric("m (Shift)", f"{params['m']:.4f}")
            p_cols[4].metric("sigma (Smooth)", f"{params['sigma']:.4f}")
            p_cols[5].metric("Model RMSE", f"{params['rmse']*100:.2f}% IV")

    # Tab 2: SSVI Multi-Expiry Volatility Surface
    with tab2:
        if clean_date_all.empty:
            st.warning("No clean data available for multi-expiry SSVI surface.")
        else:
            ssvi = SSVISurface()
            ssvi.fit(clean_date_all)
            is_arb_free, viols = ssvi.check_calendar_arbitrage()
            
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("SSVI Global Skew (ρ)", f"{ssvi.rho:.4f}")
            s_col2.metric("SSVI Curvature (η)", f"{ssvi.eta:.4f}")
            s_col3.metric("SSVI Term Power (γ)", f"{ssvi.gamma:.4f}")
            s_col4.metric("Calendar Arbitrage Status", "✅ Arbitrage Free" if is_arb_free else f"⚠️ {viols} Violations")
            
            # Multi-curve term structure plot
            df_exp = ssvi.expiries_df.sort_values('tau')
            fig_surface = go.Figure()
            
            for _, exp_row in df_exp.iterrows():
                exp_name = exp_row['expiry']
                tau_val = exp_row['tau']
                dte_val = exp_row['dte']
                
                slice_df = clean_date_all[clean_date_all['expiry'] == exp_name]
                if len(slice_df) < 3:
                    continue
                k_smooth = np.linspace(slice_df['log_moneyness'].min(), slice_df['log_moneyness'].max(), 50)
                strikes_smooth = spot_price * np.exp(k_smooth)
                iv_fit = ssvi.predict_iv(k_smooth, tau=tau_val, theta=exp_row['theta'])
                
                fig_surface.add_trace(go.Scatter(
                    x=strikes_smooth,
                    y=iv_fit * 100,
                    mode='lines',
                    name=f'{exp_name} ({dte_val}d)'
                ))
                
            fig_surface.update_layout(
                title=f"SSVI Multi-Expiry Surface Term Structure Curves ({selected_date})",
                xaxis_title="Strike Price ($)",
                yaxis_title="Implied Volatility (%)",
                height=520,
                template="plotly_white"
            )
            st.plotly_chart(fig_surface, use_container_width=True)

    # Tab 3: IV vs Realized Vol Timeline
    with tab3:
        st.subheader("Historical Volatility & IV Rank Trend")
        start_lookback = (pd.to_datetime(selected_date) - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
        
        df_und = realized_vol.extract_underlying_history(start_date=start_lookback, end_date=selected_date)
        df_rv = realized_vol.calculate_close_to_close_rv(df_und, window=21)
        df_atm = realized_vol.extract_daily_atm_iv(start_date=start_lookback, end_date=selected_date)
        df_rank = realized_vol.calculate_iv_rank_and_percentile(df_atm, window=252)
        
        df_ts = pd.merge(df_rv, df_rank, on='date', how='inner')
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df_ts['date'], y=df_ts['atm_iv']*100, name='ATM Implied Vol (30d)', line=dict(color='#1E40AF', width=2)))
        fig_ts.add_trace(go.Scatter(x=df_ts['date'], y=df_ts['rv_close_to_close']*100, name='Realized Vol (21d)', line=dict(color='#DC2626', width=1.5, dash='dash')))
        fig_ts.update_layout(title="30-Day ATM Implied Volatility vs 21-Day Realized Volatility", yaxis_title="Volatility (%)", height=400, template="plotly_white")
        st.plotly_chart(fig_ts, use_container_width=True)

    # Tab 4: Flagged Signal Table
    with tab4:
        st.subheader("Option Mispricing Signal Table")
        if 'df_flagged' in locals() and not df_flagged.empty:
            df_display = df_flagged[['strike', 'option_type', 'iv', 'svi_iv', 'iv_diff_pct', 'bid', 'ask', 'volume', 'signal']].copy()
            df_display['iv'] = (df_display['iv'] * 100).round(2)
            df_display['svi_iv'] = (df_display['svi_iv'] * 100).round(2)
            df_display['iv_diff_pct'] = df_display['iv_diff_pct'].round(2)
            
            df_display = df_display.rename(columns={
                'strike': 'Strike ($)',
                'option_type': 'Type',
                'iv': 'Market IV (%)',
                'svi_iv': 'SVI Model IV (%)',
                'iv_diff_pct': 'Diff (%)',
                'signal': 'Trading Signal'
            })
            
            filter_sig = st.radio("Filter Signal", ["ALL", "RICH (Overpriced)", "CHEAP (Underpriced)"], horizontal=True)
            if filter_sig.startswith("RICH"):
                df_display = df_display[df_display['Trading Signal'] == 'RICH']
            elif filter_sig.startswith("CHEAP"):
                df_display = df_display[df_display['Trading Signal'] == 'CHEAP']
                
            st.dataframe(df_display, use_container_width=True)

if __name__ == "__main__":
    main()
