from starlette.testclient import TestClient
import os
import sys
sys.path.append(os.getcwd())
from ria.server import app
from base64 import b64encode

client = TestClient(app)

def test_ping():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
    

def test_invalidsample():

    b64_sample = None

    with open(os.path.join(os.getcwd(), "tests/fixtures/speech.wav"), "rb") as f:
        b64_sample = b64encode(f.read()).decode('utf-8')

    response = client.post(
        "/api/health/v1/ria",
        json={"audio": {"content": b64_sample}}
    )

    assert response.status_code == 200
    assert response.json().get("status") == 'INVALID'

def test_validnegativesample():

    b64_sample = None

    with open(os.path.join(os.getcwd(), "tests/fixtures/cough_noncovid.wav"), "rb") as f:
        b64_sample = b64encode(f.read()).decode('utf-8')

    response = client.post(
        "/api/health/v1/ria",
        json={"audio": {"content": b64_sample}}
    )

    assert response.status_code == 200
    assert response.json().get("status") == 'VALID'
    assert response.json().get("result").get("prediction") == 'NORMAL'

def test_validpositivesample():

    b64_sample = None

    with open(os.path.join(os.getcwd(), "tests/fixtures/cough_covid.wav"), "rb") as f:
        b64_sample = b64encode(f.read()).decode('utf-8')

    response = client.post(
        "/api/health/v1/ria",
        json={"audio": {"content": b64_sample}}
    )

    assert response.status_code == 200
    assert response.json().get("status") == 'VALID'
    assert response.json().get("result").get("prediction") == 'COVID'
    assert response.json().get("result").get("confidence") == 0.88

def test_validpositivesample_uri():

    response = client.post(
        "/api/health/v1/ria",
        json={"audio": {"uri": "random"}}
    )

    assert response.status_code == 200
    assert response.json().get("status") == 'VALID'
    assert response.json().get("result").get("prediction") == 'COVID'
    assert response.json().get("result").get("confidence") == 0.88