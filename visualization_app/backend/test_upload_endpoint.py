"""Quick test to verify upload endpoint exists"""
import requests

try:
    # Test if server is running
    response = requests.get('http://localhost:8000/api/health')
    print(f"✓ Server is running: {response.json()}")
    
    # Check if upload endpoint is registered (will return 422 or 400 for missing file, not 404)
    response = requests.post('http://localhost:8000/api/upload-and-predict')
    
    if response.status_code == 404:
        print("✗ Upload endpoint NOT found (404)")
    elif response.status_code in [422, 400]:
        print("✓ Upload endpoint EXISTS (expecting 422/400 for missing file)")
    else:
        print(f"? Upload endpoint returned: {response.status_code}")
        
except Exception as e:
    print(f"✗ Error: {e}")

