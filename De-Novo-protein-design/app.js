// Protein Design Website JavaScript with Orange Theme

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all functionality
    initializeNavigation();
    initializeTabs();
    initializeScrollEffects();
    initializeInteractiveElements();
    initializeMetricAnimations();
    initializeDNAAnimations();
    initializeCharts();
});

// Chart colors for orange theme
const chartColors = {
    primary: '#f36f21',
    secondary: '#72635c', 
    light: '#aca6a2',
    accent1: '#4A90E2',
    accent2: '#7ED321',
    background: '#ffffff',
    grid: 'rgba(255, 255, 255, 0.3)'
};

// Initialize charts with orange theme
function initializeCharts() {
    // Method Comparison Chart
    const comparisonCtx = document.getElementById('comparisonChart');
    if (comparisonCtx) {
        new Chart(comparisonCtx, {
            type: 'bar',
            data: {
                labels: ['Our Method', 'ProteinMPNN', 'ESM-IF1v', 'Rosetta'],
                datasets: [
                    {
                        label: 'Sequence Recovery (%)',
                        data: [49.2, 52.4, 47.8, 32.9],
                        backgroundColor: chartColors.primary,
                        borderColor: chartColors.primary,
                        borderWidth: 1
                    },
                    {
                        label: 'Success Rate (%)',
                        data: [73.0, 68.0, 65.0, 45.0],
                        backgroundColor: chartColors.secondary,
                        borderColor: chartColors.secondary,
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: chartColors.secondary,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: chartColors.secondary,
                            font: {
                                size: 11
                            }
                        },
                        grid: {
                            color: chartColors.grid
                        }
                    },
                    y: {
                        ticks: {
                            color: chartColors.secondary,
                            font: {
                                size: 11
                            }
                        },
                        grid: {
                            color: chartColors.grid
                        }
                    }
                }
            }
        });
    }

    // Performance Metrics Chart
    const metricsCtx = document.getElementById('metricsChart');
    if (metricsCtx) {
        new Chart(metricsCtx, {
            type: 'radar',
            data: {
                labels: ['Sequence Recovery', 'Success Rate', 'Generation Speed', 'Confidence Score', 'Structural Accuracy'],
                datasets: [
                    {
                        label: 'Our Method',
                        data: [49.2, 73.0, 85.0, 91.0, 88.5],
                        backgroundColor: `rgba(243, 111, 33, 0.2)`,
                        borderColor: chartColors.primary,
                        borderWidth: 2,
                        pointBackgroundColor: chartColors.primary,
                        pointBorderColor: chartColors.background,
                        pointBorderWidth: 2
                    },
                    {
                        label: 'ProteinMPNN',
                        data: [52.4, 68.0, 70.0, 85.0, 82.0],
                        backgroundColor: `rgba(114, 99, 92, 0.2)`,
                        borderColor: chartColors.secondary,
                        borderWidth: 2,
                        pointBackgroundColor: chartColors.secondary,
                        pointBorderColor: chartColors.background,
                        pointBorderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: chartColors.secondary,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            color: chartColors.grid
                        },
                        grid: {
                            color: chartColors.grid
                        },
                        pointLabels: {
                            color: chartColors.secondary,
                            font: {
                                size: 10
                            }
                        },
                        ticks: {
                            color: chartColors.light,
                            font: {
                                size: 9
                            }
                        }
                    }
                }
            }
        });
    }

    // Analysis Chart
    const analysisCtx = document.getElementById('analysisChart');
    if (analysisCtx) {
        // Generate sample data for protein analysis
        const proteinData = [];
        for (let i = 0; i < 100; i++) {
            proteinData.push({
                x: Math.random() * 100 + 50, // Structural confidence
                y: Math.random() * 10 + 85,  // Stability prediction
                r: Math.random() * 15 + 5    // Designability metric
            });
        }

        new Chart(analysisCtx, {
            type: 'bubble',
            data: {
                datasets: [{
                    label: 'Generated Proteins',
                    data: proteinData,
                    backgroundColor: `rgba(243, 111, 33, 0.6)`,
                    borderColor: chartColors.primary,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: chartColors.secondary,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Confidence: ${context.parsed.x.toFixed(1)}%, Stability: ${context.parsed.y.toFixed(1)}, Designability: ${context.raw.r.toFixed(1)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Structural Confidence (%)',
                            color: chartColors.secondary,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        },
                        ticks: {
                            color: chartColors.secondary,
                            font: {
                                size: 11
                            }
                        },
                        grid: {
                            color: chartColors.grid
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Stability Prediction',
                            color: chartColors.secondary,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        },
                        ticks: {
                            color: chartColors.secondary,
                            font: {
                                size: 11
                            }
                        },
                        grid: {
                            color: chartColors.grid
                        }
                    }
                }
            }
        });
    }
}

// DNA Animation Controls
function initializeDNAAnimations() {
    const dnaStrand = document.querySelector('.dna-strand');
    const dnaContainer = document.querySelector('.dna-container');
    
    if (!dnaStrand || !dnaContainer) return;

    // Add loading state initially
    dnaStrand.classList.add('loading');
    
    // Simulate loading completion
    setTimeout(() => {
        dnaStrand.classList.remove('loading');
        dnaStrand.classList.add('loaded');
    }, 500);

    // Enhanced hover controls
    let isHovered = false;
    let touchTimeout;

    dnaContainer.addEventListener('mouseenter', function() {
        isHovered = true;
        dnaStrand.style.animationPlayState = 'paused';
        
        // Add orange glow effect
        dnaStrand.style.transform = 'scale(1.05)';
        dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.4) brightness(1.2) drop-shadow(0 15px 40px rgba(243, 111, 33, 0.5))';
    });

    dnaContainer.addEventListener('mouseleave', function() {
        isHovered = false;
        dnaStrand.style.animationPlayState = 'running';
        
        // Reset scale and glow
        dnaStrand.style.transform = '';
        dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.2) brightness(1.1)';
    });

    // Touch support for mobile devices
    dnaContainer.addEventListener('touchstart', function(e) {
        e.preventDefault();
        
        if (touchTimeout) clearTimeout(touchTimeout);
        
        // Pause animation and apply hover effects
        dnaStrand.style.animationPlayState = 'paused';
        dnaStrand.style.transform = 'scale(1.02)';
        dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.3) brightness(1.15)';
        
        // Resume after 2 seconds
        touchTimeout = setTimeout(() => {
            if (!isHovered) {
                dnaStrand.style.animationPlayState = 'running';
                dnaStrand.style.transform = '';
                dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.2) brightness(1.1)';
            }
        }, 2000);
    });

    // Click interaction for DNA strand
    dnaContainer.addEventListener('click', function() {
        // Add a quick pulse effect with orange theme
        dnaStrand.style.animation = 'none';
        dnaStrand.style.transform = 'scale(1.1)';
        dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.5) brightness(1.3)';
        
        setTimeout(() => {
            dnaStrand.style.animation = 'rotateDNA 20s linear infinite';
            dnaStrand.style.transform = isHovered ? 'scale(1.05)' : '';
            dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.2) brightness(1.1)';
        }, 200);
        
        // Show a notification with orange theme
        showNotification('3D DNA visualization - Click and drag to explore!', 'info');
    });

    // Intersection observer for DNA entrance animation
    const dnaObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0) scale(1)';
            }
        });
    }, { threshold: 0.3 });

    dnaObserver.observe(dnaContainer);

    // Performance optimization: Pause animation when not visible
    const visibilityObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (!entry.isIntersecting && !isHovered) {
                dnaStrand.style.animationPlayState = 'paused';
            } else if (entry.isIntersecting && !isHovered) {
                dnaStrand.style.animationPlayState = 'running';
            }
        });
    }, { threshold: 0.1 });

    visibilityObserver.observe(dnaContainer);

    // Respect user's motion preferences
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        dnaStrand.style.animation = 'none';
        const dnaGlow = document.querySelector('.dna-glow');
        if (dnaGlow) {
            dnaGlow.style.animation = 'none';
        }
    }
}

// Navigation functionality
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');
    
    // Smooth scrolling for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - 80; // Account for fixed navbar
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Active navigation highlighting with orange theme
    window.addEventListener('scroll', function() {
        let current = '';
        const scrollPos = window.scrollY + 100;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            
            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });

    // Navbar background on scroll with orange theme
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = '0 2px 10px rgba(243, 111, 33, 0.1)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = 'none';
        }
    });
}

// Tab functionality
function initializeTabs() {
    // Technical details tabs
    const detailTabs = document.querySelectorAll('.tab-btn');
    const detailPanels = document.querySelectorAll('.tab-panel');

    detailTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const targetId = this.onclick ? this.onclick.toString().match(/'([^']+)'/)[1] : null;
            if (targetId) showTab(targetId);
        });
    });

    // Code tabs
    const codeTabs = document.querySelectorAll('.code-tab-btn');
    const codePanels = document.querySelectorAll('.code-panel');

    codeTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const targetId = this.onclick ? this.onclick.toString().match(/'([^']+)'/)[1] : null;
            if (targetId) showCodeTab(targetId);
        });
    });
}

// Tab switching functions
function showTab(tabId) {
    // Hide all panels
    const panels = document.querySelectorAll('.tab-panel');
    panels.forEach(panel => {
        panel.classList.remove('active');
    });

    // Remove active class from all buttons
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });

    // Show target panel
    const targetPanel = document.getElementById(tabId);
    if (targetPanel) {
        targetPanel.classList.add('active');
    }

    // Add active class to clicked button
    const activeButton = document.querySelector(`[onclick="showTab('${tabId}')"]`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

function showCodeTab(tabId) {
    // Hide all code panels
    const panels = document.querySelectorAll('.code-panel');
    panels.forEach(panel => {
        panel.classList.remove('active');
    });

    // Remove active class from all code buttons
    const buttons = document.querySelectorAll('.code-tab-btn');
    buttons.forEach(btn => {
        btn.classList.remove('active');
    });

    // Show target panel
    const targetPanel = document.getElementById(tabId);
    if (targetPanel) {
        targetPanel.classList.add('active');
    }

    // Add active class to clicked button
    const activeButton = document.querySelector(`[onclick="showCodeTab('${tabId}')"]`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

// Scroll effects and animations
function initializeScrollEffects() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                
                // Add staggered animation for cards in the same container
                const siblingCards = entry.target.parentElement.querySelectorAll('.feature-card, .application-card, .result-card, .metric-card');
                siblingCards.forEach((card, index) => {
                    if (card === entry.target) {
                        setTimeout(() => {
                            card.style.opacity = '1';
                            card.style.transform = 'translateY(0)';
                        }, index * 100);
                    }
                });
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animatedElements = document.querySelectorAll('.feature-card, .application-card, .result-card, .metric-card');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'all 0.6s ease-out';
        observer.observe(el);
    });
}

// Interactive elements with orange theme
function initializeInteractiveElements() {
    // Metric card hover effects with enhanced DNA interaction
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach((card, index) => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-4px) scale(1.02)';
            this.style.boxShadow = '0 8px 25px rgba(243, 111, 33, 0.15)';
            
            // Subtle DNA pulse when metrics are hovered
            const dnaGlow = document.querySelector('.dna-glow');
            if (dnaGlow) {
                dnaGlow.style.animationDuration = '2s';
            }
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            this.style.boxShadow = '0 2px 8px rgba(243, 111, 33, 0.1)';
            
            // Reset DNA glow animation
            const dnaGlow = document.querySelector('.dna-glow');
            if (dnaGlow) {
                dnaGlow.style.animationDuration = '4s';
            }
        });
    });

    // Feature card interactions
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('click', function() {
            // Add a subtle pulse effect
            this.style.animation = 'pulse 0.3s ease-in-out';
            setTimeout(() => {
                this.style.animation = '';
            }, 300);
        });

        // Enhanced hover effect with DNA synchronization
        card.addEventListener('mouseenter', function() {
            this.style.borderColor = 'rgba(243, 111, 33, 0.3)';
            const dnaStrand = document.querySelector('.dna-strand');
            if (dnaStrand && !dnaStrand.matches(':hover')) {
                dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.3) brightness(1.15) drop-shadow(0 12px 35px rgba(243, 111, 33, 0.3))';
            }
        });

        card.addEventListener('mouseleave', function() {
            this.style.borderColor = '';
            const dnaStrand = document.querySelector('.dna-strand');
            if (dnaStrand && !dnaStrand.matches(':hover')) {
                dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.2) brightness(1.1)';
            }
        });
    });

    // Application card interactions
    const appCards = document.querySelectorAll('.application-card');
    appCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            const icon = this.querySelector('.app-icon');
            if (icon) {
                icon.style.transform = 'scale(1.1) rotate(5deg)';
                icon.style.transition = 'transform 0.3s ease';
            }
            this.style.borderColor = 'rgba(243, 111, 33, 0.3)';
        });

        card.addEventListener('mouseleave', function() {
            const icon = this.querySelector('.app-icon');
            if (icon) {
                icon.style.transform = 'scale(1) rotate(0deg)';
            }
            this.style.borderColor = '';
        });
    });

    // Add keyboard navigation for DNA interaction
    const dnaContainer = document.querySelector('.dna-container');
    if (dnaContainer) {
        dnaContainer.setAttribute('tabindex', '0');
        dnaContainer.setAttribute('role', 'button');
        dnaContainer.setAttribute('aria-label', 'Interactive 3D DNA visualization - Press Enter to pause/resume rotation');
        
        dnaContainer.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
    }
}

// Metric animations with DNA synchronization and orange theme
function initializeMetricAnimations() {
    const metricValues = document.querySelectorAll('.metric-value');
    
    const animateValue = (element, start, end, duration, suffix = '') => {
        const startTime = performance.now();
        const dnaStrand = document.querySelector('.dna-strand');
        
        const update = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * easeOut;
            
            // Sync DNA glow intensity with animation progress using orange theme
            if (dnaStrand && progress < 1) {
                const glowIntensity = 0.2 + (progress * 0.3);
                dnaStrand.style.filter = `hue-rotate(15deg) saturate(1.2) brightness(1.1) drop-shadow(0 10px 30px rgba(243, 111, 33, ${glowIntensity}))`;
            }
            
            if (suffix === '%') {
                element.textContent = current.toFixed(1) + suffix;
            } else if (suffix === 's') {
                element.textContent = current.toFixed(1) + suffix;
            } else {
                element.textContent = current.toFixed(2);
            }
            
            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                // Reset DNA glow after animation
                if (dnaStrand) {
                    setTimeout(() => {
                        dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.2) brightness(1.1)';
                    }, 500);
                }
            }
        };
        requestAnimationFrame(update);
    };

    // Animate metrics when they come into view
    const metricsObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const text = entry.target.textContent;
                
                if (text.includes('%')) {
                    const value = parseFloat(text);
                    animateValue(entry.target, 0, value, 1500, '%');
                } else if (text.includes('s')) {
                    const value = parseFloat(text);
                    animateValue(entry.target, 0, value, 1500, 's');
                } else {
                    const value = parseFloat(text);
                    animateValue(entry.target, 0, value, 1500);
                }
                
                metricsObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    metricValues.forEach(metric => {
        metricsObserver.observe(metric);
    });
}

// Utility functions
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        const offsetTop = section.offsetTop - 80;
        window.scrollTo({
            top: offsetTop,
            behavior: 'smooth'
        });
    }
}

function downloadCode() {
    // Simulate code download
    showNotification('Code download would be available in the full implementation', 'info');
}

function openContact() {
    // Simulate contact form opening
    showNotification('Contact form would open in the full implementation', 'info');
}

// Enhanced notification system with orange theme
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification--${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;

    // Add styles with orange theme
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: #ffffff;
        border: 1px solid rgba(172, 166, 162, 0.4);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 8px 25px rgba(243, 111, 33, 0.15);
        z-index: 1001;
        max-width: 300px;
        animation: slideInRight 0.3s ease-out;
    `;

    const notificationContent = notification.querySelector('.notification-content');
    notificationContent.style.cssText = `
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    `;

    const messageSpan = notification.querySelector('.notification-message');
    messageSpan.style.color = '#72635c';

    const closeButton = notification.querySelector('.notification-close');
    closeButton.style.cssText = `
        background: none;
        border: none;
        font-size: 18px;
        cursor: pointer;
        color: #aca6a2;
        padding: 0;
        line-height: 1;
    `;

    // Add to page
    document.body.appendChild(notification);

    // Subtle DNA interaction when notification appears
    const dnaStrand = document.querySelector('.dna-strand');
    if (dnaStrand) {
        dnaStrand.style.animationDuration = '15s';
        setTimeout(() => {
            dnaStrand.style.animationDuration = '20s';
        }, 1000);
    }

    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }
    }, 5000);
}

// Add CSS animations for notifications
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
`;
document.head.appendChild(notificationStyles);

// Performance optimization: Debounce scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Enhanced scroll performance with DNA optimization
const debouncedScrollHandler = debounce(function() {
    // Optimize DNA animation based on scroll speed
    const dnaStrand = document.querySelector('.dna-strand');
    if (dnaStrand) {
        const scrollSpeed = Math.abs(window.scrollY - (window.lastScrollY || 0));
        window.lastScrollY = window.scrollY;
        
        if (scrollSpeed > 10) {
            // Pause DNA animation during fast scrolling for performance
            dnaStrand.style.animationPlayState = 'paused';
            clearTimeout(window.dnaScrollTimeout);
            window.dnaScrollTimeout = setTimeout(() => {
                dnaStrand.style.animationPlayState = 'running';
            }, 150);
        }
    }
}, 16); // ~60fps

window.addEventListener('scroll', debouncedScrollHandler);

// Keyboard navigation support
document.addEventListener('keydown', function(e) {
    // Allow keyboard navigation for tabs
    if (e.key === 'Enter' || e.key === ' ') {
        const activeElement = document.activeElement;
        if (activeElement.classList.contains('tab-btn') || activeElement.classList.contains('code-tab-btn')) {
            e.preventDefault();
            activeElement.click();
        }
    }
});

// DNA Animation Control API for external use
window.DNAControls = {
    pause: function() {
        const dnaStrand = document.querySelector('.dna-strand');
        if (dnaStrand) {
            dnaStrand.style.animationPlayState = 'paused';
        }
    },
    
    resume: function() {
        const dnaStrand = document.querySelector('.dna-strand');
        if (dnaStrand) {
            dnaStrand.style.animationPlayState = 'running';
        }
    },
    
    setSpeed: function(duration) {
        const dnaStrand = document.querySelector('.dna-strand');
        if (dnaStrand && duration > 0) {
            dnaStrand.style.animationDuration = duration + 's';
        }
    },
    
    reset: function() {
        const dnaStrand = document.querySelector('.dna-strand');
        if (dnaStrand) {
            dnaStrand.style.animation = 'none';
            setTimeout(() => {
                dnaStrand.style.animation = 'rotateDNA 20s linear infinite';
            }, 10);
        }
    }
};

// Export functions for global access
window.scrollToSection = scrollToSection;
window.showTab = showTab;
window.showCodeTab = showCodeTab;
window.downloadCode = downloadCode;
window.openContact = openContact;

// Initialize page performance monitoring
window.addEventListener('load', function() {
    // Log performance metrics for optimization
    const loadTime = performance.now();
    console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
    
    // Ensure DNA animation starts smoothly with orange filter
    const dnaStrand = document.querySelector('.dna-strand');
    if (dnaStrand) {
        setTimeout(() => {
            dnaStrand.style.opacity = '1';
            dnaStrand.style.filter = 'hue-rotate(15deg) saturate(1.2) brightness(1.1)';
        }, 100);
    }
});