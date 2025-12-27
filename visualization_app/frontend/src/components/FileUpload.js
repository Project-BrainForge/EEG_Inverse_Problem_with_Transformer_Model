import React, { useState, useRef } from 'react';
import './FileUpload.css';

function FileUpload({ onUploadComplete, loading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.name.endsWith('.mat')) {
        setSelectedFile(file);
      } else {
        alert('Please select a .mat file');
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first');
      return;
    }

    onUploadComplete(selectedFile);
    setUploadProgress(0);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.name.endsWith('.mat')) {
      setSelectedFile(file);
    } else {
      alert('Please drop a .mat file');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleClear = () => {
    setSelectedFile(null);
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="file-upload-container">
      <h3>📁 Upload EEG Data</h3>
      
      <div 
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".mat"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        {!selectedFile ? (
          <div className="drop-zone-content">
            <div className="upload-icon">📤</div>
            <p>Click or drag & drop MAT file here</p>
            <span className="file-hint">Accepts .mat files with EEG data (75 channels)</span>
          </div>
        ) : (
          <div className="file-selected">
            <div className="file-icon">📄</div>
            <div className="file-info">
              <p className="file-name">{selectedFile.name}</p>
              <p className="file-size">{(selectedFile.size / 1024).toFixed(2)} KB</p>
            </div>
            <button 
              className="btn-clear"
              onClick={(e) => {
                e.stopPropagation();
                handleClear();
              }}
              disabled={loading}
            >
              ✕
            </button>
          </div>
        )}
      </div>

      {uploadProgress > 0 && uploadProgress < 100 && (
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${uploadProgress}%` }}
          ></div>
        </div>
      )}

      <button
        className="btn-upload"
        onClick={handleUpload}
        disabled={!selectedFile || loading}
      >
        {loading ? (
          <>
            <span className="spinner-small"></span>
            Processing...
          </>
        ) : (
          <>
            🚀 Upload & Predict
          </>
        )}
      </button>

      <div className="upload-info">
        <p className="info-text">
          <strong>Expected format:</strong>
        </p>
        <ul>
          <li>MAT file with 'data' or 'eeg_data' field</li>
          <li>Shape: (time_points, 75 channels)</li>
          <li>Will be resized to 500 time points if needed</li>
        </ul>
      </div>
    </div>
  );
}

export default FileUpload;

