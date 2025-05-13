function showExplanation(index) {
    fetch(`/explain/${index}`)
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            document.getElementById("lime-explanation").innerHTML = 
                `<img src="${data.image_url}" style="max-width:100%;">`;
            document.getElementById("explanation-modal").style.display = "block";
        }
    })
    .catch(error => console.error("Error fetching explanation:", error));
}

function closeModal() {
    document.getElementById("explanation-modal").style.display = "none";
}
