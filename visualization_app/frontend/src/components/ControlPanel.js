import React from 'react';
import './ControlPanel.css';

function ControlPanel({
  subjects,
  selectedSubject,
  onSubjectChange,
  currentSample,
  totalSamples,
  onSampleChange,
  threshold,
  onThresholdChange,
  normalize,
  onNormalizeChange,
  loading
}) {
  const handlePrevSample = () => {
    if (currentSample > 0) {
      onSampleChange(currentSample - 1);
    }
  };

  const handleNextSample = () => {
    if (currentSample < totalSamples - 1) {
      onSampleChange(currentSample + 1);
    }
  };

  return (
    <div className="control-panel">
      <h2>Controls</h2>

      <div className="control-group">
        <label>Subject</label>
        <select
          value={selectedSubject || ''}
          onChange={(e) => onSubjectChange(e.target.value)}
          disabled={loading}
        >
          {subjects.map((subject) => (
            <option key={subject.name} value={subject.name}>
              {subject.name} ({subject.num_files} files)
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label>
          Sample: {currentSample + 1} / {totalSamples}
        </label>
        <div className="sample-controls">
          <button
            onClick={handlePrevSample}
            disabled={currentSample === 0 || loading}
            className="btn-nav"
          >
            ◀ Prev
          </button>
          <input
            type="range"
            min="0"
            max={Math.max(0, totalSamples - 1)}
            value={currentSample}
            onChange={(e) => onSampleChange(parseInt(e.target.value))}
            disabled={loading}
            className="slider"
          />
          <button
            onClick={handleNextSample}
            disabled={currentSample >= totalSamples - 1 || loading}
            className="btn-nav"
          >
            Next ▶
          </button>
        </div>
      </div>

      <div className="control-group">
        <label>
          Threshold: {threshold.toFixed(2)}
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={threshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          disabled={loading}
          className="slider"
        />
        <div className="slider-labels">
          <span>0.0</span>
          <span>0.5</span>
          <span>1.0</span>
        </div>
      </div>

      <div className="control-group">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={normalize}
            onChange={(e) => onNormalizeChange(e.target.checked)}
            disabled={loading}
          />
          <span>Normalize Activations</span>
        </label>
      </div>

      <div className="info-box">
        <h3>ℹ️ Instructions</h3>
        <ul>
          <li>Select a subject to view predictions</li>
          <li>Use sample controls to navigate</li>
          <li>Adjust threshold to filter weak activations</li>
          <li>Rotate: Left mouse drag</li>
          <li>Zoom: Mouse wheel</li>
          <li>Pan: Right mouse drag</li>
        </ul>
      </div>
    </div>
  );
}

export default ControlPanel;

