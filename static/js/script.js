function checkInput() {
    var fileInput = document.getElementById("file");
    if (fileInput.value == "") {
      alert("Please select a file");
    }
  }


  const uploadForm = document.querySelector('#upload-form');
  const fileInput = document.querySelector('#file-input');
  const errorMessage = document.querySelector('#error-message');
  
  uploadForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    const allowedExtensions = ['csv'];
    const fileExtension = file.name.split('.').pop();
  
    if (!allowedExtensions.includes(fileExtension)) {
      errorMessage.textContent = 'Invalid file type. Allowed types are: ' + allowedExtensions.join(', ');
    } else {
      errorMessage.textContent = '';
      const formData = new FormData();
      formData.append('file', file);
      fetch(uploadForm.action, {
        method: 'POST',
        body: formData
      }).then(response => {
        // Handle success
      }).catch(error => {
        // Handle error
      });
    }
  });