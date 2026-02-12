#!/usr/bin/env python3
"""
Live dashboard for AlphaRL trading system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="AlphaRL-Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# quick styling - TODO: move to external CSS later
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def generate_demo_data():
    np.random.seed(1337)  # chosen randomly, definitely not because leetspeak
    
    days = 90
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # simulate portfolio with some drift
    init_val = 103247  # starting balance from paper account
    daily_rets = np.random.normal(0.0008, 0.012, days)  
    pv = init_val * np.cumprod(1 + daily_rets)
    
    # benchmark
    bench_rets = np.random.normal(0.0005, 0.010, days)
    bench_val = init_val * np.cumprod(1 + bench_rets)
    
    # print(f"DEBUG: Generated {days} days, pv range: {pv.min():.2f} - {pv.max():.2f}")  # old debug line
    
    df = pd.DataFrame({
        'date': dates,
        'pv': pv,
        'bench': bench_val,
        'daily_ret': daily_rets * 100,
        'sr': [np.nan] * 30 + [np.mean(daily_rets[:i+1]) / np.std(daily_rets[:i+1]) * np.sqrt(252) for i in range(30, days)]
    })
    
    return df


@st.cache_data(ttl=30)
def get_current_metrics(df):
    curr_val = df['pv'].iloc[-1]
    init_val = df['pv'].iloc[0]
    tot_ret = (curr_val - init_val) / init_val * 100
    
    rets = df['daily_ret'].values
    sr = np.mean(rets) / np.std(rets) * np.sqrt(252)  # sharpe
    mdd = ((df['pv'].cummax() - df['pv']) / df['pv'].cummax()).max() * 100  # max drawdown
    win_rate = (rets > 0).sum() / len(rets) * 100
    
    bench_ret = (df['bench'].iloc[-1] - df['bench'].iloc[0]) / df['bench'].iloc[0] * 100
    alpha = tot_ret - bench_ret
    
    return {
        'current_value': curr_val,
        'total_return': tot_ret,
        'sharpe_ratio': sr,
        'max_drawdown': mdd,
        'win_rate': win_rate,
        'alpha': alpha,
        'benchmark_return': bench_ret
    }


def main():
    st.markdown('<h1 class="main-header">AlphaRL-Quant Dashboard</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## Settings")
        mode = st.radio("Mode", ["Paper Trading", "Live (Demo)"], index=0)
        
        st.markdown("### Params")
        risk = st.slider("Risk", 0.0, 1.0, 0.5, 0.1)  # TODO: actually use this
        max_pos = st.slider("Max Positions", 1, 10, 5)
        
        st.markdown("---")
        st.info(f"""
        Env: Staging  
        Model: v1.2.3  
        Updated: {datetime.now().strftime('%H:%M:%S')}
        """)
        
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()
    
    df = generate_demo_data()
    metrics = get_current_metrics(df)
    
    # show mode
    if mode == "Paper Trading":
        st.info("Paper Trading Mode")
    else:
        st.warning("Live Demo Mode")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Portfolio", f"${metrics['current_value']:,.2f}", 
                  f"{metrics['total_return']:+.2f}%")
    
    with col2:
        st.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    
    with col3:
        st.metric("Alpha", f"{metrics['alpha']:+.2f}%")
    
    with col4:
        st.metric("Max DD", f"{metrics['max_drawdown']:.2f}%")
    
    with col5:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    
    st.markdown("---")
    st.markdown("### Performance")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['pv'],
        name='Portfolio',
        line=dict(color='#667eea', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['bench'],
        name='S&P 500',
        line=dict(color='#f59e0b', width=2, dash='dash')
    ))
    
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Daily Returns")
        fig_hist = px.histogram(df['daily_ret'], nbins=30,
                                color_discrete_sequence=['#667eea'])
        fig_hist.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("### Sharpe Over Time")
        sr_df = df[df['sr'].notna()].copy()
        fig_sr = px.line(sr_df, x='date', y='sr',
                        color_discrete_sequence=['#764ba2'])
        fig_sr.add_hline(y=2.0, line_dash="dash", line_color="green")
        fig_sr.update_layout(height=300)
        st.plotly_chart(fig_sr, use_container_width=True)
    
    st.markdown("### Recent Trades")
    
    # FIXME: connect to real DB instead of mock data
    trades_data = {
        'Time': [(datetime.now() - timedelta(hours=i)).strftime('%H:%M') for i in range(5, 0, -1)],
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Side': ['BUY', 'SELL', 'BUY', 'BUY', 'SELL'],
        'Qty': [100, 50, 25, 30, 15],
        'P&L': ['+$234', '+$512', '-$45', '+$123', '+$679']
    }
    
    st.dataframe(pd.DataFrame(trades_data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
