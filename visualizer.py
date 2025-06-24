# ui/visualizer.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import Config

class TradingViewVisualizer:
    def __init__(self, tech_analyzer, symbol: str , timeframe: str):
        self.tech = tech_analyzer
        self.df = tech_analyzer.df
        self.symbol = symbol
        self.timeframe = timeframe
        
    def plot_interactive_chart(self):
        """Create interactive TradingView-like chart with Plotly"""
        if len(self.df) < 50:
            return None
            
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03,
                           row_heights=[0.5, 0.15, 0.15, 0.2])
        
        # Price chart
        fig.add_trace(go.Candlestick(x=self.df['timestamp'],
                                    open=self.df['open'],
                                    high=self.df['high'],
                                    low=self.df['low'],
                                    close=self.df['close'],
                                    name='Price'), row=1, col=1)
        
        # Add moving averages
        for ma in Config.MA_PERIODS:
            col = f'ma_{ma}'
            if col in self.df:
                fig.add_trace(go.Scatter(x=self.df['timestamp'], 
                                        y=self.df[col],
                                        name=f'MA {ma}',
                                        line=dict(width=1)), row=1, col=1)
        
        # Add key levels
        if hasattr(self.tech, 'significant_levels') and self.tech.significant_levels:
            for level in self.tech.significant_levels:
                fig.add_hline(y=level, 
                             line=dict(color='blue', dash='dash'),
                             row=1, col=1)
        
        # Add supply/demand zones
        if hasattr(self.tech, 'supply_zones'):
            for zone in self.tech.supply_zones[-3:]:  # Last 3 zones
                fig.add_hrect(y0=zone['price']*0.99, y1=zone['price']*1.01,
                             fillcolor='rgba(255,0,0,0.2)', line_width=0,
                             row=1, col=1)
        
        if hasattr(self.tech, 'demand_zones'):
            for zone in self.tech.demand_zones[-3:]:  # Last 3 zones
                fig.add_hrect(y0=zone['price']*0.99, y1=zone['price']*1.01,
                             fillcolor='rgba(0,255,0,0.2)', line_width=0,
                             row=1, col=1)
        
        # Add BOS and ChoCH events
        if 'bos' in self.df.columns and 'choch' in self.df.columns:
            bos_events = self.df[self.df['bos']]
            choch_events = self.df[self.df['choch']]

            for idx, row in bos_events.iterrows():
                fig.add_annotation(
                    x=row['timestamp'],
                    y=row['high'],
                    text="BOS",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    row=1,
                    col=1
                )

            for idx, row in choch_events.iterrows():
                fig.add_annotation(
                    x=row['timestamp'],
                    y=row['low'],
                    text="ChoCH",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=40,
                    row=1,
                    col=1
                )
        
        # Volume chart
        colors = np.where(self.df['close'] > self.df['open'], 'green', 'red')
        fig.add_trace(go.Bar(x=self.df['timestamp'], 
                            y=self.df['volume'],
                            marker_color=colors,
                            name='Volume'), row=2, col=1)
        
        # RSI chart
        if 'rsi' in self.df:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], 
                                    y=self.df['rsi'],
                                    name='RSI',
                                    line=dict(color='purple')), row=3, col=1)
            fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=3, col=1)
            fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=3, col=1)
        
        # MACD chart
        if 'macd' in self.df and 'signal' in self.df:
            fig.add_trace(go.Scatter(x=self.df['timestamp'], 
                                    y=self.df['macd'],
                                    name='MACD',
                                    line=dict(color='blue')), row=4, col=1)
            fig.add_trace(go.Scatter(x=self.df['timestamp'], 
                                    y=self.df['signal'],
                                    name='Signal',
                                    line=dict(color='orange')), row=4, col=1)
            
            # Histogram
            hist_colors = np.where(self.df['macd'] > self.df['signal'], 'green', 'red')
            fig.add_trace(go.Bar(x=self.df['timestamp'], 
                                y=self.df['macd'] - self.df['signal'],
                                marker_color=hist_colors,
                                name='Histogram'), row=4, col=1)
            fig.add_hline(y=0, line=dict(color='gray'), row=4, col=1)
        
        # Update layout
        fig.update_layout(title=f'Advanced Analysis: {self.symbol}',
                         height=800,
                         showlegend=True,
                         xaxis_rangeslider_visible=False)
        
        return fig

    def plot_beginner_chart(self, trade_suggestion=None, forecast=None):
        """Create simplified chart for beginner mode"""
        if len(self.df) < 50:
            return None

        fig = go.Figure()

        # Price candlesticks
        fig.add_trace(go.Candlestick(
            x=self.df['timestamp'],
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            name='Price'
        ))

        # Add 50-period moving average
        if 'ma_50' in self.df:
            fig.add_trace(go.Scatter(
                x=self.df['timestamp'],
                y=self.df['ma_50'],
                name='MA 50',
                line=dict(color='blue', width=2)
            ))

        # Add forecast if available
        if forecast is not None and len(forecast) > 0:
            # Generate timestamps for the forecast
            last_timestamp = self.df['timestamp'].iloc[-1]
            if self.timeframe == '1d':  
                delta = timedelta(days=1)
            elif self.timeframe == '1h': 
                delta = timedelta(hours=1)
            else:
                # Default to 1 hour for now
                delta = timedelta(hours=1)

            forecast_timestamps = [last_timestamp + i * delta for i in range(1, len(forecast)+1)]

            fig.add_trace(go.Scatter(
                x=forecast_timestamps,
                y=forecast,
                name='Forecast',
                line=dict(color='purple', width=2, dash='dot')
            ))

        # Add trade suggestion if available
        if trade_suggestion:
            # Entry line
            fig.add_hline(
                y=trade_suggestion['entry_price'],
                line=dict(color='blue', width=2),
                annotation_text="Entry"
            )
            # Stop loss line
            fig.add_hline(
                y=trade_suggestion['stop_loss'],
                line=dict(color='red', width=2),
                annotation_text="Stop Loss"
            )
            # Take profit targets (up to 3)
            for i, target in enumerate(trade_suggestion['targets'][:3]):
                fig.add_hline(
                    y=target,
                    line=dict(color='green', width=2, dash='dash'),
                    annotation_text=f"TP{i+1}"
                )

        fig.update_layout(
            title=f"Beginner Mode: {self.symbol}",
            xaxis_rangeslider_visible=False,
            height=600
        )

        return fig
