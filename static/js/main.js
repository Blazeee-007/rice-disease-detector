document.addEventListener('DOMContentLoaded', () => {
  const actualBtn = document.getElementById('actual-btn');
  const fileChosen = document.getElementById('file-chosen');
  const previewImg = document.getElementById('preview');
  const cameraBtn = document.getElementById('camera-btn');
  const captureBtn = document.getElementById('capture-btn');
  const stopCameraBtn = document.getElementById('stop-camera-btn');
  const cameraContainer = document.getElementById('camera-container');
  const cameraFeed = document.getElementById('camera-feed');
  let mediaStream = null;

  // File change handler
  if (actualBtn) {
    actualBtn.addEventListener('change', function () {
      const file = this.files[0];
      if (file) {
        if (fileChosen) fileChosen.textContent = file.name;
        const reader = new FileReader();
        reader.onload = function (e) {
          if (previewImg) {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
          }
        };
        reader.readAsDataURL(file);
      }
    });
  }

  // Camera Open
  if (cameraBtn) {
    cameraBtn.addEventListener('click', async () => {
      if (cameraContainer) cameraContainer.style.display = 'block';
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' }
        });
        if (cameraFeed) cameraFeed.srcObject = mediaStream;
      } catch (err) {
        console.error('Camera error:', err);
        alert('Unable to access camera. Please allow camera permissions or upload an image file.');
      }
    });
  }

  // Camera Stop
  if (stopCameraBtn) {
    stopCameraBtn.addEventListener('click', () => {
      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
      }
      if (cameraContainer) cameraContainer.style.display = 'none';
    });
  }

  // Camera Capture
  if (captureBtn) {
    captureBtn.addEventListener('click', () => {
      if (!cameraFeed || !actualBtn) return;
      const canvas = document.createElement('canvas');
      canvas.width = cameraFeed.videoWidth || 640;
      canvas.height = cameraFeed.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(blob => {
        const file = new File([blob], 'captured_leaf.jpg', { type: 'image/jpeg' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        actualBtn.files = dataTransfer.files;

        if (fileChosen) fileChosen.textContent = 'captured_leaf.jpg';
        if (previewImg) {
          previewImg.src = canvas.toDataURL('image/jpeg');
          previewImg.style.display = 'block';
        }
        // Stop stream after capture
        if (mediaStream) {
          mediaStream.getTracks().forEach(track => track.stop());
        }
        if (cameraContainer) cameraContainer.style.display = 'none';
      }, 'image/jpeg', 0.92);
    });
  }
});

// Quick load sample image into the form
function loadSampleImage(url, filename) {
  const actualBtn = document.getElementById('actual-btn');
  const fileChosen = document.getElementById('file-chosen');
  const previewImg = document.getElementById('preview');

  fetch(url)
    .then(res => res.blob())
    .then(blob => {
      const file = new File([blob], filename, { type: 'image/jpeg' });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      if (actualBtn) actualBtn.files = dataTransfer.files;
      if (fileChosen) fileChosen.textContent = filename + ' (Loaded)';
      if (previewImg) {
        previewImg.src = url;
        previewImg.style.display = 'block';
      }
    })
    .catch(err => console.error('Failed to load sample:', err));
}

// Validation function
function validateImage() {
  const fileInput = document.getElementById('actual-btn');
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    alert('Please choose or capture a leaf image before submitting.');
    return false;
  }
  const submitBtn = document.getElementById('submit-btn');
  if (submitBtn) {
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing Image...';
    submitBtn.disabled = true;
  }
  return true;
}

// Google Translate script initializer
function googleTranslateElementInit() {
  new google.translate.TranslateElement({
    pageLanguage: 'en',
    includedLanguages: 'hi,te,ta,kn,ml,bn,mr,gu,pa,en'
  }, 'google_translate_element');
  const el = document.getElementById('google_translate_element');
  if (el) el.style.display = 'block';
}
