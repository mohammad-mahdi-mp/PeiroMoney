# app.py
import streamlit as st
from core.system import CryptoAnalysisSystem
from ui.visualizer import TradingViewVisualizer

def main():
    st.set_page_config(
        page_title="PeiroTahlil v4.0 - Intelligent Trading Engine",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("PeiroTahlil v4.0 - Intelligent Crypto Trading Engine")
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Analysis Parameters")
        symbol = st.text_input("Symbol", value="BTC/USDT")
        timeframe = st.selectbox("Timeframe", options=['1m','5m','15m','30m','1h','4h','1d','1w'], index=5)
        exchange = st.selectbox("Exchange", options=['kucoin', 'bybit', 'okx', 'binance'], index=1)
        total_capital = st.number_input("Total Capital ($)", min_value=100.0, value=10000.0, step=1000.0)
        
        mode = st.radio(
            "Display Mode",
            ['Beginner Mode (Simple)', 'Pro Mode (Advanced)'],
            index=1
        )
    
    tab1, tab2 = st.tabs(["Live Analysis", "Backtesting"])
    
    with tab1:
        st.header("Live Market Analysis")
        if st.button("Run Live Analysis", key="live_analysis"):
            with st.spinner("Executing multi-factor market analysis... Please wait."):
                system = CryptoAnalysisSystem(
                    symbol,
                    timeframe,
                    exchange,
                    total_capital
                )
                report, tech_analyzer = system.run_live_analysis()
                
            if report and tech_analyzer:
                st.success("Analysis complete!")
                
                trade_idea = report.get('trade_idea')
                if trade_idea:
                    if mode == 'Beginner Mode (Simple)':
                        if trade_idea['signal'] == 'BUY':
                            st.markdown("### 📈 Suggested Action: BUY")
                        else:
                            st.markdown("### 📉 Suggested Action: SELL")
                        
                        st.info(f"**Summary:** {trade_idea['summary']}")
                        
                        plot_suggestion = trade_idea.copy()
                        plot_suggestion['targets'] = [t['price'] for t in trade_idea.get('targets', [])]

                        visualizer = TradingViewVisualizer(tech_analyzer, symbol, timeframe)
                        beginner_fig = visualizer.plot_beginner_chart(
                            plot_suggestion, 
                            report.get('price_forecast', [])
                        )
                        if beginner_fig:
                            st.plotly_chart(beginner_fig, use_container_width=True)
                            
                        st.subheader("Trade Details")
                        col1, col2 = st.columns(2)
                        col1.metric("Entry Price", f"${trade_idea['entry_price']:.4f}")
                        col2.metric("Stop Loss", f"${trade_idea['stop_loss']:.4f}")
                        
                        st.subheader("Take Profit Targets")
                        for i, target in enumerate(trade_idea['targets']):
                            st.metric(f"Target {i+1}", f"${target['price']:.4f}", help=target.get('reason'))
                            
                        st.metric("Calculated Position Size", f"${report['position_size']:.2f}")
                    
                    else:
                        st.subheader("💡 Intelligent Trade Idea")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Signal", trade_idea['signal'])
                        col2.metric("Confidence Score", f"{trade_idea['confidence_score']:.1f}%")
                        col3.metric("Calculated Position Size", f"${report['position_size']:.2f}")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Entry Price", f"${trade_idea['entry_price']:.4f}")
                        col2.metric("Stop Loss (Structural)", f"${trade_idea['stop_loss']:.4f}")
                        
                        st.subheader("Take Profit Targets (Based on Liquidity/Structure)")
                        targets = trade_idea.get('targets', [])
                        if targets:
                            cols = st.columns(len(targets))
                            for i, target in enumerate(targets):
                                cols[i].metric(f"Target {i+1}", f"${target['price']:.4f}", help=target['reason'])
                        
                        st.subheader("Reasoning Log (Confluence Factors)")
                        st.code('\n'.join(trade_idea['reasoning_log']), language='text')
                        st.info(f"**Summary:** {trade_idea['summary']}")
                        
                        visualizer = TradingViewVisualizer(tech_analyzer, symbol, timeframe) 
                        pro_fig = visualizer.plot_interactive_chart()
                        if pro_fig:
                            st.plotly_chart(pro_fig, use_container_width=True)

                else:
                    st.warning("⚖️ No high-confidence trade setup found at this time.")
                    st.info("The trading engine analyzed the market but did not find a setup that meets the minimum confidence threshold.")
                    
                    visualizer = TradingViewVisualizer(tech_analyzer, symbol, timeframe) 
                    pro_fig = visualizer.plot_interactive_chart()
                    if pro_fig:
                        st.plotly_chart(pro_fig, use_container_width=True)
                
                with st.expander("Fundamental & Sentiment Analysis"):
                    st.subheader("Whale Activity (Aggregated)")
                    whale_data = report.get('whale_activity', {})
                    if whale_data:
                        st.write(f"Total Buy Volume: ${whale_data.get('total_whale_buy', 0):,.2f}")
                        st.write(f"Total Sell Volume: ${whale_data.get('total_whale_sell', 0):,.2f}")
                        st.write(f"Buy/Sell Ratio: {whale_data.get('ratio', 0):.2f}")
                        st.write(f"Market Dominance: **{whale_data.get('dominance', 'Unknown')}**")
                    else:
                        st.write("Could not retrieve whale activity data.")
                        
                with st.expander("Legacy AI Recommendation (DeepSeek)"):
                    ai_rec = report.get('ai_recommendation', {})
                    if ai_rec:
                        st.write(f"**Recommendation:** {ai_rec.get('recommendation', 'N/A')}")
                        st.write(f"**Confidence:** {ai_rec.get('confidence', 0)*100:.1f}%")
                        st.write(f"**Reason:** {ai_rec.get('reason', 'N/A')}")
                    else:
                        st.write("Could not retrieve AI recommendation.")
            else:
                st.error("Failed to complete analysis. Please check the console/logs for details.")

if __name__ == "__main__":
    main()