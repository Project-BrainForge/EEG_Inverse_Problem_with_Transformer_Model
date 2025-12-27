import React, { useState, useEffect } from 'react';
import './App.css';
import CortexVisualization from './components/CortexVisualization';
import ControlPanel from './components/ControlPanel';
import StatsPanel from './components/StatsPanel';
import { fetchSubjects, fetchPredictions, fetchCortexMesh } from './api/api';

function App() {
  const [subjects, setSubjects] = useState([]);
  const [selectedSubject, setSelectedSubject] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [cortexMesh, setCortexMesh] = useState(null);
  const [currentSample, setCurrentSample] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(0.1);
  const [normalize, setNormalize] = useState(true);

  // Load subjects on mount
  useEffect(() => {
    loadSubjects();
    loadCortexMesh();
  }, []);

  const loadSubjects = async () => {
    try {
      const data = await fetchSubjects();
      setSubjects(data);
      if (data.length > 0) {
        setSelectedSubject(data[0].name);
      }
    } catch (err) {
      setError('Failed to load subjects: ' + err.message);
    }
  };

  const loadCortexMesh = async () => {
    try {
      const mesh = await fetchCortexMesh();
      setCortexMesh(mesh);
    } catch (err) {
      setError('Failed to load cortex mesh: ' + err.message);
    }
  };

  const loadPredictions = async (subject, sampleIdx = null) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPredictions(subject, sampleIdx);
      setPredictions(data);
      setCurrentSample(0);
    } catch (err) {
      setError('Failed to load predictions: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Load predictions when subject changes
  useEffect(() => {
    if (selectedSubject) {
      loadPredictions(selectedSubject);
    }
  }, [selectedSubject]);

  const handleSubjectChange = (subject) => {
    setSelectedSubject(subject);
  };

  const handleSampleChange = (sampleIdx) => {
    setCurrentSample(sampleIdx);
  };

  const handleThresholdChange = (value) => {
    setThreshold(value);
  };

  const handleNormalizeChange = (value) => {
    setNormalize(value);
  };

  const getCurrentPrediction = () => {
    if (!predictions || !predictions.predictions) return null;
    return predictions.predictions[currentSample];
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧠 EEG Source Localization Visualization</h1>
        <p>Transformer Model Predictions</p>
      </header>

      <div className="App-content">
        <div className="left-panel">
          <ControlPanel
            subjects={subjects}
            selectedSubject={selectedSubject}
            onSubjectChange={handleSubjectChange}
            currentSample={currentSample}
            totalSamples={predictions?.num_samples || 0}
            onSampleChange={handleSampleChange}
            threshold={threshold}
            onThresholdChange={handleThresholdChange}
            normalize={normalize}
            onNormalizeChange={handleNormalizeChange}
            loading={loading}
          />

          {predictions && (
            <StatsPanel
              statistics={predictions.statistics}
              numSamples={predictions.num_samples}
              numSources={predictions.num_sources}
              currentSample={currentSample}
              fileName={predictions.file_names?.[currentSample]}
            />
          )}
        </div>

        <div className="visualization-container">
          {error && (
            <div className="error-message">
              <p>⚠️ {error}</p>
            </div>
          )}

          {loading && (
            <div className="loading-message">
              <div className="spinner"></div>
              <p>Loading predictions...</p>
            </div>
          )}

          {!loading && cortexMesh && getCurrentPrediction() && (
            <CortexVisualization
              vertices={cortexMesh.vertices}
              faces={cortexMesh.faces}
              activations={getCurrentPrediction()}
              threshold={threshold}
              normalize={normalize}
            />
          )}

          {!loading && !cortexMesh && (
            <div className="loading-message">
              <p>Loading cortex mesh...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;

