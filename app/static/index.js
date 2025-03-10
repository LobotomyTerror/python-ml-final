document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById('predictionSubmit');

    function displayOutput(prediction) {
        const diseaseDescription = `Based on the symptoms that you have describe the disease that is closely realted to your illness is: ${prediction}`
        document.getElementById('diseaseType').innerHTML = diseaseDescription;
    }

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData(form);

        const symptom = encodeURIComponent(formData.get('prediction'));
        // const submitButton = form.querySelector('button[type="submit"]');

        fetch(`http://127.0.0.1:8000/users/make-predictions?question=${symptom}`, {
            method: 'GET',
            headers: {"Content-Type": "application/json"}
        })
        .then(response => response.json())
        .then(data => displayOutput(data['message']))
        .catch(error => console.error("Failed to predict symptom", error))
    });
});
