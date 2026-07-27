/**
 * AuraScan Theme Toggle System
 * Handles dark/light theme switching with localStorage persistence
 */

(function() {
    'use strict';

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTheme);
    } else {
        initTheme();
    }

    function initTheme() {
        const toggleBtn = document.getElementById('theme-toggle');
        const body = document.body;
        
        // If no toggle button exists on this page, exit
        if (!toggleBtn) return;

        // Load saved theme preference
        const savedTheme = localStorage.getItem('aura-theme');
        if (savedTheme === 'light') {
            body.classList.add('light-theme');
            updateToggleIcon(toggleBtn, true);
        }

        // Toggle theme on button click
        toggleBtn.addEventListener('click', function() {
            const isLight = body.classList.toggle('light-theme');
            
            // Update icon
            updateToggleIcon(this, isLight);
            
            // Save preference
            localStorage.setItem('aura-theme', isLight ? 'light' : 'dark');
        });
    }

    /**
     * Update the toggle button icon
     * @param {HTMLElement} button - The toggle button element
     * @param {boolean} isLight - Whether light theme is active
     */
    function updateToggleIcon(button, isLight) {
        const icon = button.querySelector('i');
        if (!icon) return;

        // Update Lucide icon
        const iconName = isLight ? 'moon' : 'sun';
        icon.setAttribute('data-lucide', iconName);
        
        // Re-render Lucide icons if available
        if (typeof lucide !== 'undefined' && lucide.createIcons) {
            lucide.createIcons();
        }
    }

})();