window.onload = init;




document.getElementById("my-form").addEventListener("submit", function(event) {
  // Get the value of the textarea
  var textAreaValue = document.getElementsByName("review")[0].value.trim();
  // If the textarea is empty, prevent submission of the form
  if (textAreaValue === "") {
    event.preventDefault();
    // Display an error message
    document.getElementById("error-message").innerHTML = "Please enter some text.";
  }
});


function checkInput() {
    var fileInput = document.getElementById("file");
    if (fileInput.value == "") {
      alert("Please select a file");
    }
  }