// Get elements from the DOM
const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("captureButton");
const resultDiv = document.getElementById("result");
const uploadButton = document.getElementById("uploadButton");
const fileInput = document.getElementById("fileUpload");

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

//read the prediction from static/prediction.txt file when page is loaded
window.onload = async function() {
    const response = await fetch("http://localhost:8000/prediction", {
        method: "GET",
    });

    const result = await response.json();
    resultDiv.textContent = result.prediction;
}

// Capture the image when the Capture button is clicked
captureButton.addEventListener("click", async (event) => {
    event.preventDefault(); // Prevent page refresh

    // Capture the video frame and draw it on the canvas
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
        resultDiv.textContent = "Uploading and processing...";

        // Convert the canvas to a Blob
        canvas.toBlob(async (blob) => {
            // Create a FormData object to hold the image file
            const formData = new FormData();
            formData.append("file", blob, "captured_image.png");

            // Send the image data as a POST request
            const response = await fetch("http://localhost:8000/uploadimg", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("Network response was not ok");

            const result = await response.json();

            // Display the uploaded image
            const imageContainer = document.getElementById("image");
            const imgElement = document.createElement("img");
            imgElement.src = result.image_path; // Use server-returned image path
            imgElement.id = "img";

            imageContainer.innerHTML = ""; // Clear any previous image
            imageContainer.appendChild(imgElement);

            // Display the prediction result
            resultDiv.textContent = result.prediction;
        }, "image/png");
    } catch (error) {
        console.error("Error processing the image: ", error);
        resultDiv.textContent = "Failed to process the image.";
    }
});


// add updload file button code
uploadButton.addEventListener("click", async (event) => {
    event.preventDefault(); // Prevent page refresh
    //open the file upload window       
    
    fileInput.click()    
    console.log("file input clicked")
});


fileInput.addEventListener("change", async (event) => {
    const file = fileInput.files[0];
    
    resultDiv.textContent = "Uploading and processing...";


    if (file) {
        // Handle the file here, for example, upload it or display it
        document.getElementById("result").textContent = `File selected: ${file.name}`;
        
        // send the image to the server only filename using get method
        // take image's filepath
        const image_name = "C:\\Users\\Samet\\Pictures\\"+ file.name;
        console.log()
        const response = await fetch(`http://localhost:8000/upload/?image_data=${image_name}`, {
            method: "GET",
        });

        image_container = document.getElementById("image");

        // create imgElement using the file
        const imgElement = document.createElement("img");
        imgElement.src = URL.createObjectURL(file);
        imgElement.id = "img"


        image_container.innerHTML = ""; // Clear any previous image
        image_container.appendChild(imgElement);

        // set the innerHTML of the result div to the prediction
        const result = await response.json();
        resultDiv.textContent = result.prediction;
    }
});

