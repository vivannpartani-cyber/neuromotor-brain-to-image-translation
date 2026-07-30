import './style.css';

// Number of images generated in Phase 3
const NUM_IMAGES = 5;
const SIMILARITY_SCORES = ['0.976', '0.962', '0.981', '0.955', '0.970'];

document.addEventListener('DOMContentLoaded', () => {
  renderGallery();
  setupInteractions();
});

function renderGallery() {
  const grid = document.getElementById('gallery-grid');
  const template = document.getElementById('card-template') as HTMLTemplateElement;

  if (!grid || !template) return;

  for (let i = 0; i < NUM_IMAGES; i++) {
    // Clone template content
    const clone = template.content.cloneNode(true) as DocumentFragment;
    
    // Select elements within the clone
    const img = clone.querySelector('.card-image') as HTMLImageElement;
    const title = clone.querySelector('.card-title') as HTMLElement;
    const metric = clone.querySelector('.metric-badge') as HTMLElement;
    const viewBtn = clone.querySelector('.view-btn') as HTMLButtonElement;
    
    // Set data
    const paddedIndex = i.toString().padStart(4, '0');
    const imagePath = `/images/comparison_${paddedIndex}.png`;
    
    img.src = imagePath;
    title.textContent = `Mind Read ${i + 1}`;
    metric.textContent = `CosSim: ${SIMILARITY_SCORES[i]}`;
    
    // Setup modal trigger
    viewBtn.addEventListener('click', () => {
      openModal(imagePath);
    });
    
    // Add to grid
    grid.appendChild(clone);
  }
}

function setupInteractions() {
  const triggerBtn = document.getElementById('mock-trigger-btn');
  const statusPanel = document.getElementById('status-panel');
  const statusText = document.getElementById('status-text');
  const progressBar = document.getElementById('progress-bar');
  const modal = document.getElementById('image-modal') as HTMLDialogElement;
  const closeModalBtn = document.getElementById('close-modal');

  // Trigger Mock Translation Button Logic
  if (triggerBtn && statusPanel && statusText && progressBar) {
    triggerBtn.addEventListener('click', () => {
      // Disable button
      (triggerBtn as HTMLButtonElement).disabled = true;
      triggerBtn.style.opacity = '0.5';
      triggerBtn.style.cursor = 'not-allowed';

      // Show panel
      statusPanel.classList.remove('hidden');
      
      // Simulate pipeline steps
      let progress = 0;
      
      statusText.textContent = "Loading synthetic fMRI Z-scores...";
      progressBar.style.width = '10%';
      
      setTimeout(() => {
        statusText.textContent = "Mapping voxels to CLIP embeddings (Ridge R²=0.86)...";
        progressBar.style.width = '40%';
      }, 1500);
      
      setTimeout(() => {
        statusText.textContent = "Projecting 1024-D to 768-D for SD v1.5...";
        progressBar.style.width = '60%';
      }, 3000);
      
      setTimeout(() => {
        statusText.textContent = "Generating image (Stable Diffusion, 30 steps)...";
        progressBar.style.width = '85%';
      }, 4500);
      
      setTimeout(() => {
        statusText.textContent = "Reconstruction Complete!";
        progressBar.style.width = '100%';
        
        // Hide panel after a short delay and reset
        setTimeout(() => {
          statusPanel.classList.add('hidden');
          (triggerBtn as HTMLButtonElement).disabled = false;
          triggerBtn.style.opacity = '1';
          triggerBtn.style.cursor = 'pointer';
          progressBar.style.width = '0%';
        }, 2000);
      }, 7000);
    });
  }

  // Modal logic
  if (modal && closeModalBtn) {
    closeModalBtn.addEventListener('click', () => {
      modal.close();
    });
    
    // Close on backdrop click
    modal.addEventListener('click', (e) => {
      const dialogDimensions = modal.getBoundingClientRect();
      if (
        e.clientX < dialogDimensions.left ||
        e.clientX > dialogDimensions.right ||
        e.clientY < dialogDimensions.top ||
        e.clientY > dialogDimensions.bottom
      ) {
        modal.close();
      }
    });
  }
}

function openModal(imageSrc: string) {
  const modal = document.getElementById('image-modal') as HTMLDialogElement;
  const modalImg = document.getElementById('modal-image') as HTMLImageElement;
  
  if (modal && modalImg) {
    modalImg.src = imageSrc;
    modal.showModal();
  }
}
