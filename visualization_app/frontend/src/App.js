import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import CortexVisualization from './components/CortexVisualization';
import ControlPanel from './components/ControlPanel';
import StatsPanel from './components/StatsPanel';
import FileUpload from './components/FileUpload';
import AnimationControls from './components/AnimationControls';
import { fetchSubjects, fetchPredictions, fetchCortexMesh, uploadAndPredict } from './api/api';

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
  
  // Animation state
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000); // milliseconds per frame
  const animationRef = useRef(null);

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

  // Handle file upload
  const handleFileUpload = async (file) => {
    setLoading(true);
    setError(null);
    setIsPlaying(false); // Stop any animation
    
    try {
      const data = await uploadAndPredict(file);
      setPredictions(data);
      setCurrentSample(0);
      setSelectedSubject(null); // Clear subject selection when uploading
    } catch (err) {
      setError('Failed to process uploaded file: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Animation functions
  const startAnimation = useCallback(() => {
    if (!predictions || predictions.num_samples === 0) return;
    
    setIsPlaying(true);
    
    const animate = () => {
      setCurrentSample((prev) => {
        const next = prev + 1;
        if (next >= predictions.num_samples) {
          setIsPlaying(false);
          return 0; // Loop back to start
        }
        return next;
      });
    };
    
    animationRef.current = setInterval(animate, playbackSpeed);
  }, [predictions, playbackSpeed]);

  const stopAnimation = useCallback(() => {
    if (animationRef.current) {
      clearInterval(animationRef.current);
      animationRef.current = null;
    }
    setIsPlaying(false);
    setCurrentSample(0);
  }, []);

  const togglePlayPause = useCallback(() => {
    if (isPlaying) {
      if (animationRef.current) {
        clearInterval(animationRef.current);
        animationRef.current = null;
      }
      setIsPlaying(false);
    } else {
      startAnimation();
    }
  }, [isPlaying, startAnimation]);

  // Clean up animation on unmount or when predictions change
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        clearInterval(animationRef.current);
      }
    };
  }, []);

  // Restart animation when speed changes
  useEffect(() => {
    if (isPlaying) {
      if (animationRef.current) {
        clearInterval(animationRef.current);
      }
      startAnimation();
    }
  }, [playbackSpeed, isPlaying, startAnimation]);

  // Stop animation when predictions change
  useEffect(() => {
    if (animationRef.current) {
      clearInterval(animationRef.current);
      animationRef.current = null;
      setIsPlaying(false);
    }
  }, [predictions]);

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
            <>
              <AnimationControls
                isPlaying={isPlaying}
                currentSample={currentSample}
                totalSamples={predictions.num_samples}
                playbackSpeed={playbackSpeed}
                onPlayPause={togglePlayPause}
                onStop={stopAnimation}
                onSpeedChange={setPlaybackSpeed}
                disabled={loading}
              />

              <StatsPanel
                statistics={predictions.statistics}
                numSamples={predictions.num_samples}
                numSources={predictions.num_sources}
                currentSample={currentSample}
                fileName={predictions.file_names?.[currentSample]}
              />
            </>
          )}

          <FileUpload
            onUploadComplete={handleFileUpload}
            loading={loading}
          />
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

