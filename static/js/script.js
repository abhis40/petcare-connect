// PetCare Connect - Client-side JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Image preview functionality for all file inputs
    const fileInputs = document.querySelectorAll('input[type="file"][accept*="image"]');
    
    fileInputs.forEach(input => {
        const previewId = input.getAttribute('data-preview');
        const placeholderId = input.getAttribute('data-placeholder');
        
        if (previewId && placeholderId) {
            const imagePreview = document.getElementById(previewId);
            const previewPlaceholder = document.getElementById(placeholderId);
            
            if (imagePreview && previewPlaceholder) {
                input.addEventListener('change', function() {
                    if (this.files && this.files[0]) {
                        const reader = new FileReader();
                        
                        reader.onload = function(e) {
                            imagePreview.src = e.target.result;
                            imagePreview.style.display = 'block';
                            previewPlaceholder.style.display = 'none';
                        }
                        
                        reader.readAsDataURL(this.files[0]);
                    } else {
                        imagePreview.style.display = 'none';
                        previewPlaceholder.style.display = 'flex';
                    }
                });
            }
        }
    });
    
    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });
    
    // Handle pet selection in booking form
    const petSelectors = document.querySelectorAll('.pet-selector');
    const petCards = document.querySelectorAll('.pet-selector-card');
    
    petSelectors.forEach(selector => {
        selector.addEventListener('change', function() {
            petCards.forEach(card => {
                card.classList.remove('selected', 'border-primary');
            });
            
            if (this.checked) {
                const cardId = this.getAttribute('data-card');
                if (cardId) {
                    const card = document.getElementById(cardId);
                    if (card) {
                        card.classList.add('selected', 'border-primary');
                    }
                }
            }
        });
    });
    
    // Booking date/time calculation
    const startTimeInput = document.getElementById('start_time');
    const endTimeInput = document.getElementById('end_time');
    
    if (startTimeInput && endTimeInput) {
        const bookingDuration = document.getElementById('booking-duration');
        const estimatedCost = document.getElementById('estimated-cost');
        const rateElement = document.getElementById('hourly-rate');
        
        function updateBookingSummary() {
            if (startTimeInput.value && endTimeInput.value) {
                const startTime = new Date(startTimeInput.value);
                const endTime = new Date(endTimeInput.value);
                
                if (endTime > startTime) {
                    const durationMs = endTime - startTime;
                    const durationHours = durationMs / (1000 * 60 * 60);
                    
                    if (bookingDuration) {
                        bookingDuration.textContent = `${durationHours.toFixed(1)} hours`;
                    }
                    
                    // Calculate cost if rate element exists
                    if (estimatedCost && rateElement) {
                        const rate = parseFloat(rateElement.getAttribute('data-rate'));
                        const cost = rate * durationHours;
                        estimatedCost.textContent = `₹${cost.toFixed(2)}`;
                    }
                } else if (bookingDuration) {
                    bookingDuration.textContent = 'End time must be after start time';
                    if (estimatedCost) {
                        estimatedCost.textContent = '₹0';
                    }
                }
            } else if (bookingDuration) {
                bookingDuration.textContent = 'Please select dates';
                if (estimatedCost) {
                    estimatedCost.textContent = '₹0';
                }
            }
        }
        
        startTimeInput.addEventListener('change', updateBookingSummary);
        endTimeInput.addEventListener('change', updateBookingSummary);
    }
    
    // Initialize any date range pickers
    const dateRangePickers = document.querySelectorAll('.date-range-picker');
    if (dateRangePickers.length > 0) {
        // This would be implemented if using a date range picker library
        console.log('Date range pickers would be initialized here');
    }
    
    // Handle tab persistence using URL hash
    const triggerTabList = [].slice.call(document.querySelectorAll('a[data-bs-toggle="tab"]'));
    
    triggerTabList.forEach(function(triggerEl) {
        const tabTrigger = new bootstrap.Tab(triggerEl);
        
        triggerEl.addEventListener('click', function(event) {
            event.preventDefault();
            tabTrigger.show();
            history.pushState(null, null, triggerEl.getAttribute('href'));
        });
    });
    
    // Switch to tab based on URL hash
    if (location.hash) {
        const tab = document.querySelector(`a[href="${location.hash}"]`);
        if (tab) {
            const tabInstance = new bootstrap.Tab(tab);
            tabInstance.show();
        }
    }
    
    // Handle filter form submission
    const filterForm = document.getElementById('filter-form');
    const resetFiltersBtn = document.getElementById('reset-filters');
    
    if (filterForm && resetFiltersBtn) {
        resetFiltersBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Reset all form controls
            filterForm.reset();
            
            // Submit the form
            filterForm.submit();
        });
    }
});