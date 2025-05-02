document.addEventListener("DOMContentLoaded", function () {
    const danForm = document.getElementById('danForm');
    const danInput = document.getElementById('danInput');
    const knnPred = document.getElementById('knnPred');
    const knnConf = document.getElementById('knnConf');
    const rfcPred = document.getElementById('rfcPred');
    const rfcConf = document.getElementById('rfcConf');

    danForm.addEventListener('submit', function (event) {
        event.preventDefault();

        const symptom = encodeURIComponent(danInput.value);
        fetch(`http://127.0.0.1:8000/users/make-predictions?question=${symptom}`, {
            method: 'GET',
            headers: { "Content-Type": "application/json" }
        })
            .then(response => response.json())
            .then(data => {
                // console.log(data.message);
                knnPred.textContent = data.message.knnPrediction;
                knnConf.textContent = `${(data.message.knnConfidence * 100).toFixed(2)}%`;
                rfcPred.textContent = data.message.randomForestPrediction;
                rfcConf.textContent = `${(data.message.randomForestConfidence * 100).toFixed(2)}%`;
            })
            .catch(error => console.error("Failed to predict", error));
    });

    const jackForm = document.getElementById('jackForm');
    const jackInput = document.getElementById('jackInput');
    const jackOutput = document.getElementById('jackOutput');
    const jackConf = document.getElementById('jackConf');

    jackForm.addEventListener('submit', function (event) {
        event.preventDefault();

        const symptom = encodeURIComponent(jackInput.value);
        fetch(`http://127.0.0.1:8000/users/make-predictions-jack?question=${symptom}`, {
            method: 'GET',
            headers: { "Content-Type": "application/json" }
        })
            .then(response => response.json())
            .then(data => {
                jackOutput.textContent = data.message.modelClass;
                jackConf.textContent = `${(data.message.modelConfidence * 100).toFixed(2)}%`
            })
            .catch(error => console.error("Failed to predict", error));
    });
});
