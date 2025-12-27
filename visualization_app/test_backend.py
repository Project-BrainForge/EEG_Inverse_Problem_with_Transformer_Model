"""
Test script for the visualization backend
Verifies that all endpoints are working correctly
"""
import requests
import json
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_endpoint(name, url, method="GET", **kwargs):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"Method: {method}")
    
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success!")
            
            # Print relevant info
            if isinstance(data, dict):
                if 'num_vertices' in data:
                    print(f"  Vertices: {data['num_vertices']}")
                    print(f"  Faces: {data['num_faces']}")
                elif 'num_samples' in data:
                    print(f"  Samples: {data['num_samples']}")
                    print(f"  Sources: {data['num_sources']}")
                    if 'statistics' in data:
                        stats = data['statistics']
                        print(f"  Stats: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}")
                elif isinstance(data, list):
                    print(f"  Items: {len(data)}")
                    for item in data:
                        if 'name' in item:
                            print(f"    - {item['name']}: {item.get('num_files', 0)} files")
                else:
                    print(f"  Response: {json.dumps(data, indent=2)[:200]}")
            
            return True
        else:
            print(f"✗ Failed: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection Error: Is the backend running?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("="*60)
    print("EEG Visualization Backend Test Suite")
    print("="*60)
    print(f"\nBackend URL: {API_BASE}")
    print("\nMake sure the backend is running before running this test!")
    print("Start it with: python visualization_app/backend/app.py")
    
    input("\nPress Enter to start tests...")
    
    results = []
    
    # Test 1: Root endpoint
    results.append(test_endpoint(
        "Root Endpoint",
        f"{API_BASE}/"
    ))
    
    # Test 2: Health check
    results.append(test_endpoint(
        "Health Check",
        f"{API_BASE}/api/health"
    ))
    
    # Test 3: Cortex mesh
    results.append(test_endpoint(
        "Cortex Mesh",
        f"{API_BASE}/api/cortex-mesh"
    ))
    
    # Test 4: List subjects
    results.append(test_endpoint(
        "List Subjects",
        f"{API_BASE}/api/subjects"
    ))
    
    # Test 5: Get predictions (if VEP exists)
    results.append(test_endpoint(
        "Get Predictions for VEP",
        f"{API_BASE}/api/predictions/VEP"
    ))
    
    # Test 6: Get specific sample
    results.append(test_endpoint(
        "Get Specific Sample (VEP, sample 0)",
        f"{API_BASE}/api/predictions/VEP",
        params={"sample_idx": 0}
    ))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Backend is working correctly.")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
    
    print("\nNext steps:")
    print("1. If all tests passed, start the frontend: cd visualization_app/frontend && npm start")
    print("2. Open http://localhost:3000 in your browser")
    print("3. Enjoy visualizing your EEG predictions!")


if __name__ == "__main__":
    main()

