import React from 'react';
import './AnimationControls.css';

function AnimationControls({
  isPlaying,
  currentSample,
  totalSamples,
  playbackSpeed,
  onPlayPause,
  onStop,
  onSpeedChange,
  disabled
}) {
  const speeds = [
    { label: '0.5x', value: 2000 },
    { label: '1x', value: 1000 },
    { label: '2x', value: 500 },
    { label: '4x', value: 250 },
    { label: '8x', value: 125 },
  ];

  const getCurrentSpeedLabel = () => {
    const speed = speeds.find(s => s.value === playbackSpeed);
    return speed ? speed.label : '1x';
  };

  const progress = totalSamples > 0 ? (currentSample / (totalSamples - 1)) * 100 : 0;

  return (
    <div className="animation-controls">
      <h3>🎬 Animation Controls</h3>
      
      <div className="playback-info">
        <div className="time-display">
          <span className="time-current">{currentSample + 1}</span>
          <span className="time-separator">/</span>
          <span className="time-total">{totalSamples}</span>
        </div>
        <div className="progress-track">
          <div 
            className="progress-indicator" 
            style={{ left: `${progress}%` }}
          ></div>
        </div>
      </div>

      <div className="control-buttons">
        <button
          className="btn-control btn-play"
          onClick={onPlayPause}
          disabled={disabled || totalSamples === 0}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        
        <button
          className="btn-control btn-stop"
          onClick={onStop}
          disabled={disabled || totalSamples === 0 || currentSample === 0}
          title="Stop and Reset"
        >
          ⏹
        </button>

        <div className="speed-control">
          <label>Speed:</label>
          <div className="speed-buttons">
            {speeds.map((speed) => (
              <button
                key={speed.value}
                className={`btn-speed ${playbackSpeed === speed.value ? 'active' : ''}`}
                onClick={() => onSpeedChange(speed.value)}
                disabled={disabled}
              >
                {speed.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="playback-hint">
        <p>
          {isPlaying ? (
            <>🎥 Playing at {getCurrentSpeedLabel()}</>
          ) : (
            <>Press ▶ to animate through time points</>
          )}
        </p>
      </div>
    </div>
  );
}

export default AnimationControls;

