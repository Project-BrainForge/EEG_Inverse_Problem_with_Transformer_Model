import React from 'react';
import './StatsPanel.css';

function StatsPanel({ statistics, numSamples, numSources, currentSample, fileName }) {
  if (!statistics) return null;

  return (
    <div className="stats-panel">
      <h2>Statistics</h2>

      <div className="stat-grid">
        <div className="stat-item">
          <span className="stat-label">Total Samples</span>
          <span className="stat-value">{numSamples}</span>
        </div>

        <div className="stat-item">
          <span className="stat-label">Source Regions</span>
          <span className="stat-value">{numSources}</span>
        </div>

        <div className="stat-item">
          <span className="stat-label">Current Sample</span>
          <span className="stat-value">{currentSample + 1}</span>
        </div>

        {fileName && (
          <div className="stat-item full-width">
            <span className="stat-label">File</span>
            <span className="stat-value file-name">{fileName}</span>
          </div>
        )}
      </div>

      <div className="stats-divider"></div>

      <h3>Activation Statistics</h3>

      <div className="stat-grid">
        <div className="stat-item">
          <span className="stat-label">Min</span>
          <span className="stat-value">{statistics.min.toFixed(6)}</span>
        </div>

        <div className="stat-item">
          <span className="stat-label">Max</span>
          <span className="stat-value">{statistics.max.toFixed(6)}</span>
        </div>

        <div className="stat-item">
          <span className="stat-label">Mean</span>
          <span className="stat-value">{statistics.mean.toFixed(6)}</span>
        </div>

        <div className="stat-item">
          <span className="stat-label">Std Dev</span>
          <span className="stat-value">{statistics.std.toFixed(6)}</span>
        </div>
      </div>

      <div className="color-legend">
        <h3>Color Scale</h3>
        <div className="legend-bar">
          <div className="legend-gradient"></div>
          <div className="legend-labels">
            <span>0.0</span>
            <span>0.5</span>
            <span>1.0</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default StatsPanel;

