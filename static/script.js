document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('eeg-file-input');
    const predictBtn = document.getElementById('predict-btn');
    const loadingDiv = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    const errorSection = document.getElementById('error-section');
    const predictedText = document.getElementById('predicted-text');
    const audioPlayer = document.getElementById('audio-player');
    const errorMessage = document.getElementById('error-message');
    const fileLabel = document.querySelector('.file-label');

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileLabel.textContent = fileInput.files[0].name;
        } else {
            fileLabel.textContent = 'Choose File';
        }
    });

    predictBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];

        if (!file) {
            showError('Please select a file first.');
            return;
        }

        // Reset UI
        hideError();
        hideResult();
        showLoading();

        const formData = new FormData();
        formData.append('eeg_file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                predictedText.textContent = data.sentence;
                audioPlayer.src = data.audio_url;
                showResult();
            } else {
                showError(data.error || 'An unknown error occurred.');
            }
        } catch (error) {
            showError('An error occurred while communicating with the server.');
        } finally {
            hideLoading();
        }
    });

    function showLoading() {
        loadingDiv.classList.remove('hidden');
    }

    function hideLoading() {
        loadingDiv.classList.add('hidden');
    }

    function showResult() {
        resultSection.classList.remove('hidden');
    }

    function hideResult() {
        resultSection.classList.add('hidden');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorSection.classList.remove('hidden');
    }

    function hideError() {
        errorSection.classList.add('hidden');
    }
});