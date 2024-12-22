import requests

response = requests.post(
    "https://localhost:8000/predict/",
    files={"file": open("ml project/image/captured_image.png", "rb")},
    verify=False  # Bypass SSL verification (only for development)
)
print(response.json())