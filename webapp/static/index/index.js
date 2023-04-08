const prediction = document.getElementById('pred');

setInterval(() => {
    fetch('/predict')
        .then(response => response.text())
        .then(text => {
            prediction.innerText = text;
        })
}, 750);
