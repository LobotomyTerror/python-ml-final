document.addEventListener("DOMContentLoaded", function() {
    const danForm = document.getElementById('danForm');
    const danInput = document.getElementById('danInput');
    const danOutput = document.getElementById('danOutput');
  
    danForm.addEventListener('submit', function(event) {
      event.preventDefault();  // Prevent normal form submit
  
      const symptom = encodeURIComponent(danInput.value);
      fetch(`http://127.0.0.1:8000/users/make-predictions?question=${symptom}`, {
        method: 'GET',
        headers: {"Content-Type": "application/json"}
      })
      .then(response => response.json())
      .then(data => {
        danOutput.textContent = data.message;
      })
      .catch(error => console.error("Failed to predict", error));
    });
  
    const jackForm = document.getElementById('jackForm');
    const jackInput = document.getElementById('jackInput');
    const jackOutput = document.getElementById('jackOutput');
  
    jackForm.addEventListener('submit', function(event) {
      event.preventDefault();
  
      const symptom = encodeURIComponent(jackInput.value);
      fetch(`http://127.0.0.1:8000/users/make-predictions-jack?question=${symptom}`, {
        method: 'GET',
        headers: {"Content-Type": "application/json"}
      })
      .then(response => response.json())
      .then(data => {
        jackOutput.textContent = data.message;
      })
      .catch(error => console.error("Failed to predict", error));
    });
  });
  