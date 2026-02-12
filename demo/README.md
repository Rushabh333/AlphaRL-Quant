# AlphaRL-Quant Live Demo Dashboard

Interactive web dashboard showcasing the AlphaRL-Quant trading system's capabilities.

## 🚀 Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r demo/requirements.txt

# Run the dashboard
streamlit run demo/dashboard.py
```

The dashboard will open at `http://localhost:8501`

### Deploy to Streamlit Cloud (Get Shareable Link)

1. **Push to GitHub** (if not already):
   ```bash
   git add demo/
   git commit -m "Add demo dashboard"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository: `Rushabh333/AlphaRL-Quant`
   - Main file path: `demo/dashboard.py`
   - Click "Deploy"

3. **Get your shareable link**:
   - You'll get a URL like: `https://rushabh333-alpharl-quant.streamlit.app`
   - Share this link with anyone!

## 📊 Features

The dashboard includes:

- **📈 Real-time Portfolio Performance**: 90-day performance chart vs S&P 500
- **💰 Key Metrics**: Portfolio value, Sharpe ratio, Alpha, Max drawdown, Win rate
- **📊 Analytics**: Daily returns distribution, Sharpe ratio evolution
- **💼 Recent Trades**: Simulated trading activity with P&L
- **🎛️ Interactive Controls**: Risk tolerance, position limits
- **🔴 Live Updates**: Auto-refresh capability

## 🎨 Screenshots

### Main Dashboard
![Dashboard](https://img.shields.io/badge/Status-Live-success?style=for-the-badge)

**Key Metrics Row**:
- Portfolio Value: $XXX,XXX (+X.XX%)
- Sharpe Ratio: X.XX
- Alpha vs S&P500: +X.XX%
- Max Drawdown: X.XX%
- Win Rate: XX.X%

**Performance Chart**:
- Interactive line chart comparing portfolio vs benchmark
- 90-day historical performance

**Analytics**:
- Daily returns histogram
- Sharpe ratio evolution over time

## 🔧 Customization

### Change Demo Data

Edit `generate_demo_data()` in `dashboard.py`:

```python
# Modify these parameters
initial_value = 100000  # Starting portfolio value
daily_returns = np.random.normal(0.0008, 0.012, days)  # Return distribution
```

### Add Real Data

Connect to your actual trading database:

```python
import psycopg2

def get_real_data():
    conn = psycopg2.connect(...)
    df = pd.read_sql("SELECT * FROM portfolio_history", conn)
    return df
```

### Customize Styling

Modify the CSS in the `st.markdown()` section:

```python
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #YOUR_COLOR 0%, #YOUR_COLOR 100%);
    }
</style>
""", unsafe_allow_html=True)
```

## 📱 Mobile Responsive

The dashboard is fully responsive and works on:
- 💻 Desktop
- 📱 Mobile phones
- 📱 Tablets

## 🌐 Share Your Demo

Once deployed to Streamlit Cloud, you can:

1. **Share the link** with investors, team members, recruiters
2. **Embed in website**: Use iframe to embed the dashboard
3. **Present live**: Use during presentations and demos

Example shareable link:
```
https://rushabh333-alpharl-quant.streamlit.app
```

## 🔒 Security Note

This demo uses **simulated data** for demonstration purposes. For production:

1. Add authentication
2. Use real trading data
3. Implement rate limiting
4. Add access controls

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Charts](https://plotly.com/python/)
- [AlphaRL-Quant Main README](../README.md)

## 🆘 Troubleshooting

### Port already in use
```bash
streamlit run demo/dashboard.py --server.port 8502
```

### Dependencies error
```bash
pip install --upgrade streamlit plotly pandas numpy
```

### Deployment issues
- Check that `demo/requirements.txt` is in your repo
- Ensure `demo/dashboard.py` path is correct
- Verify your GitHub repo is public (or grant Streamlit access)

## 🎯 Next Steps

1. ✅ Run the demo locally
2. ✅ Deploy to Streamlit Cloud
3. ✅ Share your link
4. 🚀 Impress your audience!
