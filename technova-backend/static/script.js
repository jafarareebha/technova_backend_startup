// ===== GLOBAL UTILITIES =====

/**
 * Format number as currency
 */
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

/**
 * Format date
 */
function formatDate(date) {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(new Date(date));
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info', duration = 3000) {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
        `;
        document.body.appendChild(toastContainer);
    }
    
    // Create toast
    const toast = document.createElement('div');
    toast.style.cssText = `
        background: white;
        color: #333;
        padding: 12px 24px;
        border-radius: 5px;
        margin-bottom: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        border-left: 4px solid ${type === 'success' ? '#48bb78' : type === 'error' ? '#f56565' : type === 'warning' ? '#ecc94b' : '#667eea'};
        animation: slideIn 0.3s ease;
        min-width: 250px;
    `;
    toast.textContent = message;
    
    // Add animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);
    
    toastContainer.appendChild(toast);
    
    // Remove toast after duration
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, duration);
}

// Add slideOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ===== FORM VALIDATION =====

/**
 * Validate email
 */
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

/**
 * Validate password strength
 */
function validatePassword(password) {
    return {
        length: password.length >= 8,
        uppercase: /[A-Z]/.test(password),
        lowercase: /[a-z]/.test(password),
        number: /[0-9]/.test(password),
        special: /[!@#$%^&*]/.test(password)
    };
}

/**
 * Show password strength
 */
function showPasswordStrength(password) {
    const strength = validatePassword(password);
    const meter = document.getElementById('password-strength');
    if (!meter) return;
    
    let score = 0;
    if (strength.length) score++;
    if (strength.uppercase) score++;
    if (strength.lowercase) score++;
    if (strength.number) score++;
    if (strength.special) score++;
    
    const percentage = (score / 5) * 100;
    meter.style.width = percentage + '%';
    
    if (percentage < 40) {
        meter.style.background = '#f56565';
    } else if (percentage < 70) {
        meter.style.background = '#ecc94b';
    } else {
        meter.style.background = '#48bb78';
    }
}

// ===== LOADING STATES =====

/**
 * Show loading spinner
 */
function showLoading(element) {
    if (!element) return;
    
    const originalContent = element.innerHTML;
    element.dataset.originalContent = originalContent;
    element.disabled = true;
    
    const spinner = document.createElement('span');
    spinner.className = 'spinner';
    spinner.style.cssText = `
        display: inline-block;
        width: 1rem;
        height: 1rem;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 0.5rem;
    `;
    
    element.innerHTML = '';
    element.appendChild(spinner);
    element.appendChild(document.createTextNode(' Loading...'));
}

/**
 * Hide loading spinner
 */
function hideLoading(element) {
    if (!element || !element.dataset.originalContent) return;
    
    element.innerHTML = element.dataset.originalContent;
    element.disabled = false;
}

// ===== MODAL HANDLING =====

/**
 * Open modal
 */
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

/**
 * Close modal
 */
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// ===== API CALLS =====

/**
 * Make API request
 */
async function apiRequest(url, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(url, options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'API request failed');
        }
        
        return result;
    } catch (error) {
        showToast(error.message, 'error');
        throw error;
    }
}

// ===== SIMULATION MANAGER =====
class SimulationManager {
    constructor() {
        this.baseFactors = {};
        this.baseScore = 0;
        this.currentFactors = {};
        this.adjustments = {};
        this.weights = {
            'Market': 0.25,
            'Team': 0.20,
            'Product': 0.20,
            'Financials': 0.20,
            'Competition': 0.15
        };
        
        this.init();
    }
    
    init() {
        // Get base data
        const factorsElement = document.getElementById('base-factors');
        const scoreElement = document.getElementById('base-score');
        
        if (!factorsElement || !scoreElement) return;
        
        try {
            this.baseFactors = JSON.parse(factorsElement.value);
            this.baseScore = parseFloat(scoreElement.value);
            this.currentFactors = {...this.baseFactors};
            
            this.initSliders();
        } catch (error) {
            console.error('Error initializing simulation:', error);
        }
    }
    
    initSliders() {
        const sliders = document.querySelectorAll('.simulation-slider');
        
        sliders.forEach(slider => {
            // Set initial value display
            const factor = slider.dataset.factor;
            const valueDisplay = document.getElementById(`value-${factor}`);
            if (valueDisplay) {
                valueDisplay.textContent = '0';
            }
            
            // Add event listener
            slider.addEventListener('input', (e) => {
                this.handleSliderChange(e);
            });
        });
    }
    
    handleSliderChange(event) {
        const slider = event.target;
        const factor = slider.dataset.factor;
        const adjustment = parseInt(slider.value);
        
        // Update adjustment value
        this.adjustments[factor] = adjustment;
        
        // Update display
        const valueDisplay = document.getElementById(`value-${factor}`);
        if (valueDisplay) {
            valueDisplay.textContent = adjustment > 0 ? `+${adjustment}` : adjustment;
        }
        
        // Update simulation
        this.updateSimulation();
    }
    
    updateSimulation() {
        // Calculate new factor scores
        for (let factor in this.baseFactors) {
            const adj = this.adjustments[factor] || 0;
            let newScore = this.baseFactors[factor] + adj;
            newScore = Math.min(100, Math.max(0, newScore));
            this.currentFactors[factor] = Math.round(newScore);
            
            // Update factor display
            this.updateFactorDisplay(factor, this.currentFactors[factor]);
        }
        
        // Calculate new overall score
        let newScore = 0;
        for (let factor in this.currentFactors) {
            newScore += this.currentFactors[factor] * (this.weights[factor] || 0.2);
        }
        newScore = Math.round(newScore);
        
        // Update overall score display
        this.updateOverallDisplay(newScore);
        
        // Update category
        this.updateCategory(newScore);
    }
    
    updateFactorDisplay(factor, score) {
        // Update value
        const valueElement = document.getElementById(`sim-value-${factor}`);
        if (valueElement) {
            valueElement.textContent = score;
        }
        
        // Update circle color
        const circle = document.getElementById(`sim-circle-${factor}`);
        if (circle) {
            if (score >= 75) {
                circle.style.background = '#00df81';
            } else if (score >= 50) {
                circle.style.background = '#ffc107';
            } else if (score >= 25) {
                circle.style.background = '#ff9800';
            } else {
                circle.style.background = '#ff4444';
            }
        }
    }
    
    updateOverallDisplay(score) {
        const scoreElement = document.getElementById('simulated-score');
        if (scoreElement) {
            scoreElement.textContent = score;
        }
    }
    
    updateCategory(score) {
        let category = '';
        if (score >= 75) category = 'Strong';
        else if (score >= 50) category = 'Moderate';
        else if (score >= 25) category = 'Weak';
        else category = 'Poor';
        
        const badge = document.getElementById('simulated-badge');
        if (badge) {
            badge.textContent = category;
            
            // Update badge color
            if (category === 'Strong') {
                badge.style.background = '#00df81';
                badge.style.color = 'black';
            } else if (category === 'Moderate') {
                badge.style.background = '#ffc107';
                badge.style.color = 'black';
            } else if (category === 'Weak') {
                badge.style.background = '#ff9800';
                badge.style.color = 'black';
            } else {
                badge.style.background = '#ff4444';
                badge.style.color = 'white';
            }
        }
        
        // Update circle color
        const circle = document.getElementById('simulated-circle');
        if (circle) {
            if (category === 'Strong') circle.style.background = '#00df81';
            else if (category === 'Moderate') circle.style.background = '#ffc107';
            else if (category === 'Weak') circle.style.background = '#ff9800';
            else circle.style.background = '#ff4444';
        }
    }
}

// ===== DASHBOARD INTERACTIONS =====

/**
 * Initialize dashboard
 */
function initDashboard() {
    // Add click handlers to analysis cards
    const cards = document.querySelectorAll('.analysis-card');
    cards.forEach(card => {
        card.addEventListener('click', function(e) {
            // Don't navigate if clicking on a button inside the card
            if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A') {
                return;
            }
            window.location = this.dataset.url;
        });
    });
}

// ===== FORM INTERACTIONS =====

/**
 * Initialize form
 */
function initForm() {
    // Initialize sliders
    const sliders = document.querySelectorAll('.score-slider');
    sliders.forEach(slider => {
        const outputId = slider.id + '_val';
        const output = document.getElementById(outputId);
        if (output) {
            output.textContent = slider.value;
        }
        
        slider.addEventListener('input', function() {
            const output = document.getElementById(this.id + '_val');
            if (output) {
                output.textContent = this.value;
            }
        });
    });
    
    // Form validation
    const form = document.getElementById('analysisForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            const startupName = document.getElementById('startup_name')?.value.trim();
            const description = document.getElementById('idea_description')?.value.trim();
            
            if (startupName && startupName.length < 2) {
                e.preventDefault();
                showToast('Please enter a valid startup name', 'error');
                return;
            }
            
            if (description && description.length < 50) {
                const proceed = confirm('Your description is quite short. For better analysis, please provide more details. Do you want to continue anyway?');
                if (!proceed) {
                    e.preventDefault();
                }
            }
        });
    }
}

// ===== INITIALIZE ON PAGE LOAD =====
document.addEventListener('DOMContentLoaded', () => {
    // Initialize simulation on results page
    if (document.getElementById('simulation-section')) {
        new SimulationManager();
    }
    
    // Initialize dashboard
    if (document.querySelector('.analyses-grid')) {
        initDashboard();
    }
    
    // Initialize form
    if (document.getElementById('analysisForm')) {
        initForm();
    }
    
    // Add active class to current nav link
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
    
    // Close modals when clicking outside
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            closeModal(e.target.id);
        }
    });
    
    // Handle escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const activeModal = document.querySelector('.modal.active');
            if (activeModal) {
                closeModal(activeModal.id);
            }
        }
    });
});

// ===== EXPORT FOR USE IN OTHER FILES =====
window.utils = {
    formatCurrency,
    formatDate,
    showToast,
    apiRequest
};