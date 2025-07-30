import os
import sys

# Set working directory to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
print("Enhanced Trading Backtest Visualization")
print(f"Working Directory: {os.getcwd()}")

import asyncio
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_position_size(cash_available, price, max_position_pct=0.1, min_shares=1):
    max_cash_to_use = cash_available * max_position_pct
    max_shares = int(max_cash_to_use // price)
    return max(min_shares, max_shares)

def create_sequences(data, seq_length):
    sequences = []
    for i in range(seq_length, len(data)):
        sequences.append(data[i - seq_length:i])
    return np.array(sequences)

def simulate_enhanced_trading(df, initial_capital=10000, max_position_pct=0.1,
                              transaction_cost=0.0, confidence_threshold=0.6):
    cash = initial_capital
    shares = 0
    portfolio_values = []
    cash_history = []
    shares_history = []
    positions = []
    trades = []
    last_signal = 0
    last_trade_price = 0

    for idx, row in df.iterrows():
        price = row['close']
        signal = row['signal']
        confidence = row.get('proba', 0.5)

        if confidence < confidence_threshold and signal != 0:
            signal = 0

        position_value = shares * price
        portfolio_value = cash + position_value

        if signal == 1 and last_signal != 1 and cash > price:
            position_size = calculate_position_size(cash, price, max_position_pct)
            if position_size > 0:
                cost = position_size * price * (1 + transaction_cost)
                if cost <= cash:
                    cash -= cost
                    shares += position_size
                    last_trade_price = price
                    trades.append({
                        'type': 'BUY',
                        'timestamp': row['ts'],
                        'price': price,
                        'shares': position_size,
                        'cost': cost,
                        'confidence': confidence
                    })

        elif signal == -1 and last_signal != -1 and shares > 0:
            proceeds = shares * price * (1 - transaction_cost)
            cash += proceeds
            profit = proceeds - (shares * last_trade_price)
            trades.append({
                'type': 'SELL',
                'timestamp': row['ts'],
                'price': price,
                'shares': shares,
                'proceeds': proceeds,
                'profit': profit,
                'confidence': confidence
            })
            shares = 0

        last_signal = signal
        portfolio_value = cash + shares * price

        portfolio_values.append(portfolio_value)
        cash_history.append(cash)
        shares_history.append(shares)
        positions.append(shares * price)

    df['cash'] = cash_history
    df['shares'] = shares_history
    df['position_value'] = positions
    df['portfolio_value'] = portfolio_values

    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    max_portfolio_value = max(portfolio_values)
    max_drawdown = min([(pv - max_portfolio_value) / max_portfolio_value
                        for pv in portfolio_values])

    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() > 0 else 0

    profitable_trades = [t for t in trades if t['type'] == 'SELL' and t.get('profit', 0) > 0]
    total_sell_trades = [t for t in trades if t['type'] == 'SELL']
    win_rate = len(profitable_trades) / len(total_sell_trades) if total_sell_trades else 0

    performance = {
        'initial_capital': initial_capital,
        'final_value': portfolio_values[-1],
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'trades': trades
    }

    return df, performance

def create_enhanced_plot(df, performance, symbol, model_name):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f"{symbol} - {model_name.upper()} Model Backtest",
            "Portfolio Value & Drawdown",
            "Position Value & Cash",
            "Trading Signals & Confidence"
        ),
        row_heights=[0.4, 0.25, 0.2, 0.15]
    )

    fig.add_trace(
        go.Scatter(
            x=df['ts'], y=df['close'],
            name='Price',
            line=dict(color='black', width=1),
            hovertemplate='<b>Price</b>: $%{y:.2f}<br><b>Time</b>: %{x}<extra></extra>'
        ),
        row=1, col=1
    )

    buys = df[df['signal'] == 1]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys['ts'], y=buys['close'],
                mode='markers', name='Buy Signal',
                marker=dict(symbol='triangle-up', color='green', size=8),
                hovertemplate='<b>BUY</b><br>Price: $%{y:.2f}<br>Confidence: %{customdata:.1%}<extra></extra>',
                customdata=buys['proba']
            ),
            row=1, col=1
        )

    sells = df[df['signal'] == -1]
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells['ts'], y=sells['close'],
                mode='markers', name='Sell Signal',
                marker=dict(symbol='triangle-down', color='red', size=8),
                hovertemplate='<b>SELL</b><br>Price: $%{y:.2f}<br>Confidence: %{customdata:.1%}<extra></extra>',
                customdata=sells['proba']
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=df['ts'], y=df['portfolio_value'],
            name='Portfolio Value',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Portfolio</b>: $%{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_hline(
        y=performance['initial_capital'],
        line_dash="dash", line_color="gray",
        annotation_text="Initial Capital",
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['ts'], y=df['position_value'],
            name='Position Value',
            line=dict(color='orange', width=1),
            fill='tonexty',
            hovertemplate='<b>Position</b>: $%{y:,.0f}<extra></extra>'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['ts'], y=df['cash'],
            name='Cash',
            line=dict(color='green', width=1),
            hovertemplate='<b>Cash</b>: $%{y:,.0f}<extra></extra>'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['ts'], y=df['proba'],
            name='Model Confidence',
            line=dict(color='purple', width=1),
            hovertemplate='<b>Confidence</b>: %{y:.1%}<extra></extra>'
        ),
        row=4, col=1
    )

    fig.add_hline(
        y=0.6,
        line_dash="dash", line_color="red",
        annotation_text="Confidence Threshold",
        row=4, col=1
    )

    fig.update_layout(
        height=1000,
        width=1400,
        title=dict(
            text=f"{symbol} Trading Backtest - {model_name.upper()} Model<br>"
                 f"Return: {performance['total_return']:.1%} | "
                 f"Sharpe: {performance['sharpe_ratio']:.2f} | "
                 f"Max DD: {performance['max_drawdown']:.1%} | "
                 f"Win Rate: {performance['win_rate']:.1%} | "
                 f"Trades: {performance['total_trades']}",
            x=0.5
        ),
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=3, col=1)
    fig.update_yaxes(title_text="Confidence", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)

    return fig

def print_performance_summary(symbol, model_name, performance):
    print(f"\nPERFORMANCE SUMMARY - {symbol} ({model_name.upper()})")
    print("=" * 60)
    print(f"Initial Capital:    ${performance['initial_capital']:>10,.0f}")
    print(f"Final Value:        ${performance['final_value']:>10,.0f}")
    print(f"Total Return:       {performance['total_return']:>10.1%}")
    print(f"Max Drawdown:       {performance['max_drawdown']:>10.1%}")
    print(f"Sharpe Ratio:       {performance['sharpe_ratio']:>10.2f}")
    print(f"Win Rate:           {performance['win_rate']:>10.1%}")
    print(f"Total Trades:       {performance['total_trades']:>10}")
    if performance['trades']:
        sell_trades = [t for t in performance['trades'] if t['type'] == 'SELL']
        if sell_trades:
            profits = [t.get('profit', 0) for t in sell_trades]
            avg_profit = np.mean(profits)
            print(f"Avg Profit/Trade:   ${avg_profit:>10.2f}")
            print(f"Best Trade:         ${max(profits):>10.2f}")
            print(f"Worst Trade:        ${min(profits):>10.2f}")
    print("=" * 60)

def plot_signals_and_profit(
    pipeline, symbol, model_name='xgb', initial_capital=10000,
    max_position_pct=0.1, confidence_threshold=0.6, transaction_cost=0.001
):
    print(f"\nAnalyzing {symbol} with {model_name.upper()} model...")

    # 1. Load and validate data
    df = asyncio.run(pipeline.load_data(symbol))
    if df is None or df.empty:
        print(f"No data available for {symbol}")
        return

    print(f"Loaded {len(df)} data points")

    # 2. Create features
    try:
        df_feat = pipeline.create_all_features(df)
        df_feat = df_feat.dropna().reset_index(drop=True)
        feature_columns = pipeline.feature_names
        print(f"Created {len(feature_columns)} features")
    except Exception as e:
        print(f"Feature creation failed: {e}")
        return

    # 3. Load trained model
    try:
        model, scaler, selector = pipeline.load_trained_model(symbol, model_name)
        print(f"Loaded {model_name} model successfully")
    except FileNotFoundError:
        print(f"Model {model_name} not found for {symbol}")
        symbol_dir = os.path.join(pipeline.config.model_dir, symbol)
        if os.path.exists(symbol_dir):
            models = [f.replace('.joblib', '') for f in os.listdir(symbol_dir) if f.endswith('.joblib')]
            for model in models:
                print(f"   - {model}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Prepare features and make predictions
    try:
        X = df_feat[feature_columns].values
        if scaler is not None:
            X = scaler.transform(X)
        if selector is not None:
            X = selector.transform(X)

        if model_name == "lstm":
            sequence_length = 60  # Use the same as in training
            if len(X) <= sequence_length:
                print("Not enough data for LSTM prediction.")
                return
            X_seq = create_sequences(X, sequence_length)
            predictions = (model.predict(X_seq) > 0.5).astype(int).flatten()
            probabilities = model.predict(X_seq).flatten()
            # Align DataFrame with predictions
            df_feat = df_feat.iloc[sequence_length:].reset_index(drop=True)
            df_feat['prediction'] = predictions
            df_feat['proba'] = probabilities
        else:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.ones(len(X)) * 0.5
            df_feat['prediction'] = predictions
            df_feat['proba'] = probabilities

        print(f"Generated predictions for {len(df_feat)} data points")

    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    # 5. Generate trading signals
    df_feat['signal'] = 0
    df_feat.loc[df_feat['prediction'] == 1, 'signal'] = 1
    df_feat.loc[df_feat['prediction'] == 0, 'signal'] = -1

    buy_signals = (df_feat['signal'] == 1).sum()
    sell_signals = (df_feat['signal'] == -1).sum()
    avg_confidence = df_feat['proba'].mean()

    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    print(f"Average confidence: {avg_confidence:.1%}")

    # 6. Run enhanced trading simulation
    print(f"Running trading simulation...")
    df_result, performance = simulate_enhanced_trading(
        df_feat,
        initial_capital=initial_capital,
        max_position_pct=max_position_pct,
        transaction_cost=transaction_cost,
        confidence_threshold=confidence_threshold
    )

    # 7. Print performance summary
    print_performance_summary(symbol, model_name, performance)

    # 8. Create and show enhanced plot
    fig = create_enhanced_plot(df_result, performance, symbol, model_name)
    fig.show()

    # 9. Save results and plot
    try:
        results_dir = os.path.join(pipeline.config.model_dir, 'backtest_results')
        os.makedirs(results_dir, exist_ok=True)

        # Save performance metrics
        performance_file = os.path.join(results_dir, f"{symbol}_{model_name}_performance.json")
        import json
        with open(performance_file, 'w') as f:
            performance_json = {}
            for key, value in performance.items():
                if key == 'trades':
                    performance_json[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    performance_json[key] = value.item()
                else:
                    performance_json[key] = value
            json.dump(performance_json, f, indent=2, default=str)

        # Save plot as HTML
        plot_file = os.path.join(results_dir, f"{symbol}_{model_name}_plot.html")
        fig.write_html(plot_file)
        print(f"Plot saved to {plot_file}")

    except Exception as e:
        print(f"Could not save results: {e}")

if __name__ == "__main__":
    from Utils.Config_Loader import load_config
    from Logic.Train import AdvancedTrainerPipeline

    config = load_config()
    pg_dsn = config.get("POSTGRES_DSN")
    model_dir = config.get("MODEL_DIR", "Models")
    pipeline = AdvancedTrainerPipeline(pg_dsn=pg_dsn, model_dir=model_dir)

    MODEL_ALIASES = {
        "xgboost": "xgb", "xgb": "xgb",
        "lightgbm": "lgbm", "lgbm": "lgbm", "light": "lgbm",
        "catboost": "catboost", "cat": "catboost",
        "voting_soft": "voting_soft", "voting": "voting_soft",
        "stacking": "stacking", "stack": "stacking",
        "lstm": "lstm", "neural": "lstm"
    }

    print("Enhanced Trading Strategy Backtester")
    print("=" * 50)

    symbols_input = input("Enter symbols (comma-separated): ").strip().upper()
    symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]

    model_input = input("Model [xgb/lgbm/catboost/voting_soft/stacking/lstm]: ").strip().lower() or "xgb"
    model_name = MODEL_ALIASES.get(model_input, model_input)

    capital_input = input("Initial capital [$10,000]: ").strip()
    initial_capital = float(capital_input.replace(',', '').replace(' ', '')) if capital_input else 10000

    position_input = input("Max position size (% of portfolio) [10%]: ").strip().replace('%', '')
    max_position_pct = float(position_input) / 100 if position_input else 0.1

    confidence_input = input("Confidence threshold (%) [60%]: ").strip().replace('%', '')
    confidence_threshold = float(confidence_input) / 100 if confidence_input else 0.6

    cost_input = input("Transaction cost (%) [0.1%]: ").strip().replace('%', '')
    transaction_cost = float(cost_input) / 100 if cost_input else 0.001

    batch_mode = input("Run batch analysis? [y/N]: ").strip().lower() == 'y'

    print(f"\nCONFIGURATION")
    print(f"   Capital: ${initial_capital:,.0f}")
    print(f"   Max Position: {max_position_pct:.1%}")
    print(f"   Confidence Threshold: {confidence_threshold:.1%}")
    print(f"   Transaction Cost: {transaction_cost:.1%}")
    print(f"   Model: {model_name.upper()}")

    if batch_mode and len(symbols) > 1:
        for symbol in symbols:
            plot_signals_and_profit(
                pipeline, symbol,
                model_name=model_name,
                initial_capital=initial_capital,
                max_position_pct=max_position_pct,
                confidence_threshold=confidence_threshold,
                transaction_cost=transaction_cost
            )
    else:
        for symbol in symbols:
            plot_signals_and_profit(
                pipeline, symbol,
                model_name=model_name,
                initial_capital=initial_capital,
                max_position_pct=max_position_pct,
                confidence_threshold=confidence_threshold,
                transaction_cost=transaction_cost
            )
