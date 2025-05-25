"""
Simple Forex Dashboard - Clean Zero Baselines
Just the dashboard without complex strategy calculations
"""

import os
import pandas as pd
import json
from flask import Flask, jsonify, render_template

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

@app.route('/')
def index():
    """Main dashboard page with clean zero baselines"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Forex Backtest Dashboard</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="container mt-4">
            <h1 class="text-center mb-4">Forex Backtest Dashboard</h1>
            
            <div class="row">
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Total P&L</h5>
                            <h3 class="text-success">$0.00</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Total Trades</h5>
                            <h3>0</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Win Rate</h5>
                            <h3>0%</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card">
                        <div class="card-body text-center">
                            <h5 class="card-title">Max Drawdown</h5>
                            <h3>0%</h3>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Backtest Configuration</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3">
                                    <label class="form-label">Currency Pair</label>
                                    <select class="form-select">
                                        <option>EURUSD</option>
                                        <option>GBPUSD</option>
                                        <option>USDJPY</option>
                                    </select>
                                </div>
                                <div class="col-md-3">
                                    <label class="form-label">Timeframe</label>
                                    <select class="form-select">
                                        <option>30_M</option>
                                        <option>5_M</option>
                                    </select>
                                </div>
                                <div class="col-md-3">
                                    <label class="form-label">Strategy</label>
                                    <select class="form-select">
                                        <option>MA Crossover</option>
                                    </select>
                                </div>
                                <div class="col-md-3">
                                    <button class="btn btn-primary mt-4">Run Backtest</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Performance Chart</h5>
                        </div>
                        <div class="card-body">
                            <canvas id="performanceChart" width="400" height="200"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Clean zero baseline chart
            const ctx = document.getElementById('performanceChart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Start'],
                    datasets: [{
                        label: 'Account Balance',
                        data: [10000],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 9500,
                            max: 10500
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/api/backtest-results')
def backtest_results():
    """Return clean zero baseline results"""
    return jsonify({
        'total_pnl': 0.0,
        'total_trades': 0,
        'win_rate': 0.0,
        'avg_trade': 0.0,
        'max_drawdown': 0.0,
        'profit_factor': 0.0,
        'trades': []
    })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)