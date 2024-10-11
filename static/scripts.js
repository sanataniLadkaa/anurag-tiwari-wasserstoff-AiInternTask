document.getElementById("upload-form").addEventListener("submit", function(event) {
    document.getElementById("upload-button").disabled = true;  // Disable button to prevent multiple clicks
    document.getElementById("loading-spinner").style.display = "block";  // Show the loading spinner
});
