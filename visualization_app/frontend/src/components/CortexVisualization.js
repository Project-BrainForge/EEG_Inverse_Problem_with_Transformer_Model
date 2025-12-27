import React, { useMemo, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import './CortexVisualization.css';

function CortexMesh({ vertices, faces, activations, threshold, normalize }) {
  const meshRef = useRef();

  // Process activations and create colors
  const { colors, geometry } = useMemo(() => {
    // Create geometry
    const geo = new THREE.BufferGeometry();
    
    // Convert vertices to Float32Array
    const verticesArray = new Float32Array(vertices.length * 3);
    for (let i = 0; i < vertices.length; i++) {
      verticesArray[i * 3] = vertices[i][0];
      verticesArray[i * 3 + 1] = vertices[i][1];
      verticesArray[i * 3 + 2] = vertices[i][2];
    }
    geo.setAttribute('position', new THREE.BufferAttribute(verticesArray, 3));
    
    // Convert faces to Uint32Array
    const facesArray = new Uint32Array(faces.length * 3);
    for (let i = 0; i < faces.length; i++) {
      facesArray[i * 3] = faces[i][0];
      facesArray[i * 3 + 1] = faces[i][1];
      facesArray[i * 3 + 2] = faces[i][2];
    }
    geo.setIndex(new THREE.BufferAttribute(facesArray, 1));
    
    // Compute normals for proper lighting
    geo.computeVertexNormals();
    
    // Process activations
    let processedActivations = [...activations];
    
    // Normalize if requested
    if (normalize) {
      const maxAbs = Math.max(...processedActivations.map(Math.abs));
      if (maxAbs > 0) {
        processedActivations = processedActivations.map(v => Math.abs(v) / maxAbs);
      }
    } else {
      processedActivations = processedActivations.map(Math.abs);
    }
    
    // Apply threshold
    processedActivations = processedActivations.map(v => 
      v < threshold ? 0 : v
    );
    
    // Create color array based on activations
    const colorArray = new Float32Array(vertices.length * 3);
    for (let i = 0; i < vertices.length; i++) {
      const activation = i < processedActivations.length ? processedActivations[i] : 0;
      const color = getColorForActivation(activation);
      colorArray[i * 3] = color.r;
      colorArray[i * 3 + 1] = color.g;
      colorArray[i * 3 + 2] = color.b;
    }
    
    geo.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    
    return { colors: colorArray, geometry: geo };
  }, [vertices, faces, activations, threshold, normalize]);

  // Color mapping function (hot colormap)
  function getColorForActivation(value) {
    // Hot colormap: black -> red -> orange -> yellow -> white
    if (value <= 0) {
      return { r: 0.1, g: 0.1, b: 0.15 }; // Dark blue-gray for inactive
    }
    
    if (value < 0.25) {
      // Black to dark red
      const t = value / 0.25;
      return {
        r: 0.1 + t * 0.4,
        g: 0.1,
        b: 0.15
      };
    } else if (value < 0.5) {
      // Dark red to red
      const t = (value - 0.25) / 0.25;
      return {
        r: 0.5 + t * 0.5,
        g: 0.1 + t * 0.15,
        b: 0.15
      };
    } else if (value < 0.75) {
      // Red to orange/yellow
      const t = (value - 0.5) / 0.25;
      return {
        r: 1.0,
        g: 0.25 + t * 0.5,
        b: 0.15
      };
    } else {
      // Orange to yellow/white
      const t = (value - 0.75) / 0.25;
      return {
        r: 1.0,
        g: 0.75 + t * 0.25,
        b: 0.15 + t * 0.35
      };
    }
  }

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshPhongMaterial
        vertexColors={true}
        side={THREE.DoubleSide}
        shininess={30}
        specular={new THREE.Color(0.2, 0.2, 0.2)}
      />
    </mesh>
  );
}

function CortexVisualization({ vertices, faces, activations, threshold, normalize }) {
  return (
    <div className="cortex-canvas">
      <Canvas>
        <PerspectiveCamera makeDefault position={[0, 0, 200]} fov={50} />
        
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        <directionalLight position={[-10, -10, -5]} intensity={0.4} />
        <pointLight position={[0, 0, 100]} intensity={0.3} />
        
        {/* Cortex mesh */}
        <CortexMesh
          vertices={vertices}
          faces={faces}
          activations={activations}
          threshold={threshold}
          normalize={normalize}
        />
        
        {/* Controls */}
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          rotateSpeed={0.5}
          zoomSpeed={0.8}
          panSpeed={0.5}
          minDistance={50}
          maxDistance={500}
        />
      </Canvas>
      
      <div className="canvas-overlay">
        <div className="view-hint">
          🖱️ Drag to rotate • Scroll to zoom • Right-click to pan
        </div>
      </div>
    </div>
  );
}

export default CortexVisualization;

