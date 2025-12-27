# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User's Browser                          │
│                     http://localhost:3000                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │              React Frontend (Port 3000)               │    │
│  ├───────────────────────────────────────────────────────┤    │
│  │                                                       │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │    │
│  │  │ Control     │  │ Cortex       │  │ Stats      │ │    │
│  │  │ Panel       │  │ Visualization│  │ Panel      │ │    │
│  │  │             │  │              │  │            │ │    │
│  │  │ - Subject   │  │ - Three.js   │  │ - Min/Max  │ │    │
│  │  │ - Sample    │  │ - WebGL      │  │ - Mean/Std │ │    │
│  │  │ - Threshold │  │ - 3D Mesh    │  │ - Counts   │ │    │
│  │  │ - Normalize │  │ - Colors     │  │            │ │    │
│  │  └─────────────┘  └──────────────┘  └────────────┘ │    │
│  │                                                       │    │
│  │                    ↕ API Calls (Axios)               │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/REST
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │                    API Endpoints                      │    │
│  ├───────────────────────────────────────────────────────┤    │
│  │                                                       │    │
│  │  GET  /api/cortex-mesh        → Cortex geometry     │    │
│  │  GET  /api/subjects           → List subjects        │    │
│  │  GET  /api/predictions/{id}   → Get predictions      │    │
│  │  POST /api/predict/{id}       → Run inference        │    │
│  │  GET  /api/health             → Health check         │    │
│  │                                                       │    │
│  └───────────────────────────────────────────────────────┘    │
│                              ↕                                  │
│  ┌───────────────────────────────────────────────────────┐    │
│  │                  Data Processing                      │    │
│  ├───────────────────────────────────────────────────────┤    │
│  │                                                       │    │
│  │  • Load MAT files (scipy.io.loadmat)                │    │
│  │  • Cache predictions (in-memory)                     │    │
│  │  • Process activations                               │    │
│  │  • Run model inference (PyTorch)                     │    │
│  │                                                       │    │
│  └───────────────────────────────────────────────────────┘    │
│                              ↕                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↕ File I/O
┌─────────────────────────────────────────────────────────────────┐
│                         File System                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  anatomy/                                                       │
│  └── fs_cortex_20k.mat          (Cortex mesh: pos, tri)       │
│                                                                 │
│  source/                                                        │
│  └── VEP/                                                       │
│      ├── data*.mat              (EEG data)                     │
│      └── transformer_predictions_*.mat  (Predictions)          │
│                                                                 │
│  checkpoints/                                                   │
│  └── best_model.pt              (Trained model)                │
│                                                                 │
│  models/                                                        │
│  └── transformer_model.py       (Model definition)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend Components

#### App.js (Main Controller)
```javascript
State Management:
- subjects: List of available subjects
- selectedSubject: Currently selected subject
- predictions: Loaded prediction data
- cortexMesh: Brain mesh geometry
- currentSample: Current sample index
- threshold: Activation threshold
- normalize: Normalization flag

Responsibilities:
- Coordinate all components
- Manage application state
- Handle API calls
- Update visualization
```

#### CortexVisualization.js (3D Renderer)
```javascript
Props:
- vertices: Array of 3D coordinates
- faces: Array of triangle indices
- activations: Current activation values
- threshold: Filter threshold
- normalize: Normalization flag

Responsibilities:
- Create Three.js geometry
- Map activations to colors
- Render 3D mesh
- Handle camera controls
```

#### ControlPanel.js (User Interface)
```javascript
Props:
- subjects: Available subjects
- currentSample: Current index
- totalSamples: Total count
- threshold: Current threshold
- normalize: Normalization state

Responsibilities:
- Subject selection
- Sample navigation
- Threshold adjustment
- Settings control
```

#### StatsPanel.js (Statistics Display)
```javascript
Props:
- statistics: Activation stats
- numSamples: Sample count
- numSources: Source count
- currentSample: Current index
- fileName: Data file name

Responsibilities:
- Display statistics
- Show metadata
- Color legend
```

### Backend Components

#### FastAPI Application
```python
Main Components:
- app: FastAPI instance
- CORS middleware: Enable cross-origin requests
- Global caches: Store loaded data
- Route handlers: API endpoints

Data Flow:
1. Request received
2. Check cache
3. Load data if needed
4. Process data
5. Return response
```

#### Data Loading
```python
Functions:
- load_cortex_mesh(): Load brain geometry
- load_model(): Load PyTorch model
- load_predictions(): Load MAT predictions
- preprocess_eeg(): Normalize EEG data

Caching:
- CORTEX_DATA: Mesh geometry
- PREDICTIONS_CACHE: Predictions
- MODEL: Loaded model
```

## Data Flow

### Loading Predictions

```
User selects subject
        ↓
Frontend: fetchPredictions(subject)
        ↓
Backend: GET /api/predictions/{subject}
        ↓
Check PREDICTIONS_CACHE
        ↓
If not cached:
  - Find MAT file
  - Load with scipy.io.loadmat
  - Extract 'all_out' array
  - Calculate statistics
  - Cache result
        ↓
Return JSON response
        ↓
Frontend: Update state
        ↓
CortexVisualization: Render with new data
```

### Changing Sample

```
User moves slider
        ↓
ControlPanel: onSampleChange(index)
        ↓
App: setCurrentSample(index)
        ↓
App: getCurrentPrediction()
        ↓
CortexVisualization: Re-render
  - Extract activations[index]
  - Apply threshold
  - Map to colors
  - Update geometry
        ↓
Display updated 3D view
```

### Adjusting Threshold

```
User moves threshold slider
        ↓
ControlPanel: onThresholdChange(value)
        ↓
App: setThreshold(value)
        ↓
CortexVisualization: Re-render
  - Filter activations < threshold
  - Update colors
  - Refresh display
        ↓
Immediate visual update (no API call)
```

## Technology Stack

### Backend Stack
```
FastAPI          → Web framework
Uvicorn          → ASGI server
NumPy            → Array operations
SciPy            → MAT file loading
PyTorch          → Model inference
Pydantic         → Data validation
```

### Frontend Stack
```
React            → UI framework
Three.js         → 3D rendering
React Three Fiber → React + Three.js
Axios            → HTTP client
React Hooks      → State management
```

## Performance Optimizations

### Backend
1. **Caching**: Store loaded data in memory
2. **Lazy loading**: Load data on-demand
3. **Batch processing**: Handle multiple samples efficiently
4. **GPU acceleration**: Use CUDA if available

### Frontend
1. **Memoization**: Prevent unnecessary re-renders
2. **useMemo**: Cache computed values
3. **BufferGeometry**: Efficient vertex storage
4. **WebGL**: Hardware acceleration

## Security Considerations

### Backend
- CORS properly configured
- Input validation on all endpoints
- Error handling prevents crashes
- No sensitive data exposure

### Frontend
- Environment variables for config
- Safe data parsing
- XSS prevention (React default)
- No eval() or dangerous operations

## Scalability

### Current Limits
- **Samples**: 1000+ per subject
- **Vertices**: 20,000 (can handle 100k+)
- **Subjects**: Unlimited
- **Concurrent users**: 10-50 (single server)

### Scaling Options
1. **Horizontal**: Multiple backend instances
2. **Vertical**: More powerful server
3. **Caching**: Redis for shared cache
4. **CDN**: Static file delivery
5. **Database**: PostgreSQL for metadata

## Deployment Options

### Development (Current)
```
Backend:  python app.py
Frontend: npm start
Access:   localhost:3000
```

### Production
```
Backend:  uvicorn app:app --host 0.0.0.0 --port 8000
Frontend: npm run build → serve with nginx
Access:   your-domain.com
```

### Docker (Future)
```
docker-compose up
  - backend container
  - frontend container
  - nginx container
```

## Error Handling

### Backend Errors
```python
try:
    # Operation
except FileNotFoundError:
    raise HTTPException(404, "File not found")
except Exception as e:
    raise HTTPException(500, str(e))
```

### Frontend Errors
```javascript
try {
  const data = await fetchPredictions(subject);
  setPredictions(data);
} catch (error) {
  setError(error.message);
}
```

## Testing Strategy

### Backend Tests
- Unit tests: Individual functions
- Integration tests: API endpoints
- Load tests: Performance under load

### Frontend Tests
- Component tests: React Testing Library
- E2E tests: Cypress/Playwright
- Visual tests: Screenshot comparison

## Monitoring

### Backend Metrics
- Request count
- Response time
- Error rate
- Memory usage

### Frontend Metrics
- Page load time
- API call latency
- Render performance
- User interactions

## Future Architecture

### Planned Enhancements
1. **WebSocket**: Real-time updates
2. **Database**: Persistent storage
3. **Authentication**: User accounts
4. **Cloud storage**: S3 for data files
5. **Microservices**: Separate services

### Potential Stack Evolution
```
Current:  Monolithic (FastAPI + React)
Future:   Microservices
  - API Gateway
  - Data Service
  - Inference Service
  - Storage Service
  - Frontend Service
```

## Summary

The architecture is designed for:
- ✅ **Simplicity**: Easy to understand and modify
- ✅ **Performance**: Fast and responsive
- ✅ **Scalability**: Can handle growth
- ✅ **Maintainability**: Clean code structure
- ✅ **Extensibility**: Easy to add features

The separation of concerns (frontend/backend) allows independent development and deployment of each component.

