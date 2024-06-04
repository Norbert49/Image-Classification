// static/scripts.js
document.addEventListener("DOMContentLoaded", function() {
    const fileInput = document.querySelector('input[type="file"]');
    const fileLabel = document.querySelector('span');

    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileLabel.textContent = this.files[0].name;
        } else {
            fileLabel.textContent = 'No file chosen';
        }
    });
});
