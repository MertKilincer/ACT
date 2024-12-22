// Get elements from the DOM
const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("captureButton");
const resultDiv = document.getElementById("result");
const fileInput = document.getElementById("fileUpload");
const uploadButton = document.getElementById("uploadButton");

// Access the camera
navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
        console.log("Camera stream obtained");
        video.srcObject = stream;
        video.style.display = "block"; // Ensure video is displayed
    })
    .catch((err) => {
        console.error("Error accessing the camera: ", err);
        resultDiv.textContent = "Failed to access camera.";
    });

// Capture the image when the Capture button is clicked
captureButton.addEventListener("click", async (event) => {
    event.preventDefault();  // Prevent form submission if any form is used

    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current video frame onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob and send it to the server
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "captured_image.png");

        try {
            resultDiv.textContent = "Processing... Please wait.";  // Show loading text

            const response = await fetch("http://localhost:8000/predict/", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const result = await response.json();
            if (result && result.prediction) {
                resultDiv.textContent = `Prediction: ${result.prediction}`;
            } else {
                resultDiv.textContent = "Prediction not found in the server response.";
            }
        } catch (error) {
            console.error("Error during prediction:", error);
            resultDiv.textContent = "An error occurred. Please try again.";  // Error message
        }
    }, "image/png");
});

// Handle file upload button to open the file explorer
uploadButton.addEventListener("click", () => {
    fileInput.click();  // Triggers the file explorer to open
});

// Handle file input change (when an image is selected)
fileInput.addEventListener("change", async (event) => {
    const file = fileInput.files[0];

    if (!file) return; // If no file is selected, exit the function

    const formData = new FormData();
    formData.append("file", file);

    try {
        resultDiv.textContent = "Processing... Please wait.";  // Show loading text

        const response = await fetch("http://localhost:8000/predict/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const result = await response.json();

        if (result && result.prediction) {
            resultDiv.textContent = `Prediction: ${result.prediction}`;
        } else {
            resultDiv.textContent = "Prediction not found in the server response.";
        }
    } catch (error) {
        console.error("Error during prediction:", error);
        resultDiv.textContent = "An error occurred. Please try again.";  // Error message
    }
});
