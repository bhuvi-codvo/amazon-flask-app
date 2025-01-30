import requests
import json

def test_bulk_predictions():
    # API endpoint
    url = "http://127.0.0.1:5001/predict_bulk"
    
    # Test data with multiple products
    payload = {
        "products": [
            "Vasagle Nightstand Spaces Modern Farmhouse",
            "Vasagle Storage Shelves Industrial Ulet273B01",
            "Furinno Turn N Tube Bedside Plastic Columbia",
            "Vasagle C Shaped Industrial Bedroom Ulet350B01",
            "Vantic Shaped Storage Bedroom Rusticbrown",
            "Vasagle Charging Station Nightstand Ulet228W01",
            "Yoobure Charging Station Outlets Nightstand",
            "Hoobro Charging Station Nightstand Bf09Ubz01",
            "Lelelinky Storage Industrial Turntable Records",
            "Vasagle Modern Nightstand Fabric Basket"
        ]
    }
    
    # Headers
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        # Make POST request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            # Get the JSON response
            results = response.json()
            
            # Print JSON with proper formatting
            print(json.dumps(results, indent=2))
        else:
            error_response = {
                "error": f"HTTP Error {response.status_code}",
                "message": response.text
            }
            print(json.dumps(error_response, indent=2))
            
    except requests.exceptions.ConnectionError:
        error_response = {
            "error": "Connection Error",
            "message": "Could not connect to the server. Make sure backend_app.py is running on port 5001"
        }
        print(json.dumps(error_response, indent=2))
    except Exception as e:
        error_response = {
            "error": "Unknown Error",
            "message": str(e)
        }
        print(json.dumps(error_response, indent=2))

if __name__ == "__main__":
    # First test if server is running
    try:
        health_check = requests.get("http://127.0.0.1:5001/")
        if health_check.status_code == 200:
            test_bulk_predictions()
        else:
            error_response = {
                "error": "Health Check Failed",
                "message": "Server health check failed"
            }
            print(json.dumps(error_response, indent=2))
    except requests.exceptions.ConnectionError:
        error_response = {
            "error": "Server Not Running",
            "message": "Server is not running. Please start backend_app.py first"
        }
        print(json.dumps(error_response, indent=2)) 